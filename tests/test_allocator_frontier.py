from itertools import product
import unittest

from allocator import build_tensor_specs, estimate_size
from allocator_frontier import (
    build_objective_tables,
    build_result,
    expand_allocations,
    filter_rd_by_live_modules,
    pareto_prune,
    prune_local_configs,
    resolve_moe_top_k,
    select_frontier_candidate,
    selection_signature,
    search_frontier,
)
from tensor_aliases import tensor_aliases


def rd_tensor(num_params, nrmse_by_cfg=None, *, is_1d=False, shape=None, layer_idx=None):
    if is_1d:
        return {
            "shape": list(shape or [num_params]),
            "num_params": num_params,
            "layer_idx": layer_idx,
            "is_1d": True,
            "rd_curve": {},
            "sqnr": {},
        }

    nrmse_by_cfg = nrmse_by_cfg or {}
    sqnr = {cfg: 12.0 for cfg in nrmse_by_cfg}
    sqnr["16_0"] = 100.0
    rd_curve = dict(nrmse_by_cfg)
    rd_curve["16_0"] = 0.0
    return {
        "shape": list(shape or [num_params // 4, 4]),
        "num_params": num_params,
        "layer_idx": layer_idx,
        "is_1d": False,
        "rd_curve": rd_curve,
        "sqnr": sqnr,
    }


class TensorAliasTests(unittest.TestCase):
    def test_standard_weight_aliases_include_module_path(self):
        aliases = tensor_aliases("model.layers.0.self_attn.q_proj.weight")
        self.assertIn("model.layers.0.self_attn.q_proj", aliases)

    def test_gate_up_proj_aliases_split_to_gate_and_up(self):
        aliases = tensor_aliases("model.layers.0.mlp.gate_up_proj.weight")
        self.assertIn("model.layers.0.mlp.gate_proj", aliases)
        self.assertIn("model.layers.0.mlp.up_proj", aliases)

    def test_packed_expert_aliases_map_to_switch_mlp(self):
        aliases = tensor_aliases("model.layers.1.mlp.experts.gate_up_proj.weight")
        self.assertIn("model.layers.1.mlp.switch_mlp.gate_proj", aliases)
        self.assertIn("model.layers.1.mlp.switch_mlp.up_proj", aliases)

    def test_sanitize_remap_aliases_cover_language_model_swap(self):
        aliases = tensor_aliases("model.language_model.layers.0.self_attn.q_proj.weight")
        self.assertIn("language_model.model.layers.0.self_attn.q_proj", aliases)


class LiveScopeFilterTests(unittest.TestCase):
    def test_filter_rd_by_live_modules_excludes_non_live_tensors(self):
        rd_data = {
            "model": "synthetic",
            "total_layers": 2,
            "num_2d_tensors": 4,
            "num_1d_tensors": 0,
            "tensors": {
                "model.layers.0.self_attn.q_proj.weight": rd_tensor(64, {"2_32": 0.4, "4_64": 0.1}),
                "model.layers.0.mlp.down_proj.weight": rd_tensor(64, {"2_32": 0.4, "4_64": 0.1}),
                "model.visual.blocks.0.attn.q_proj.weight": rd_tensor(64, {"2_32": 0.4, "4_64": 0.1}),
                "model.language_model.layers.1.self_attn.k_proj.weight": rd_tensor(64, {"2_32": 0.4, "4_64": 0.1}),
            },
        }
        live_modules = {
            "model.layers.0.self_attn.q_proj",
            "language_model.model.layers.1.self_attn.k_proj",
        }

        filtered, reason_counts, excluded_count = filter_rd_by_live_modules(rd_data, live_modules)

        self.assertEqual(
            set(filtered["tensors"]),
            {
                "model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.1.self_attn.k_proj.weight",
            },
        )
        self.assertEqual(reason_counts, {"not_live_in_mlxlm": 2})
        self.assertEqual(excluded_count, 2)
        self.assertEqual(filtered["num_2d_tensors"], 2)


class FrontierSearchTests(unittest.TestCase):
    def make_rd_data(self):
        return {
            "model": "synthetic",
            "total_layers": 2,
            "num_2d_tensors": 5,
            "num_1d_tensors": 1,
            "tensors": {
                "model.layers.0.self_attn.q_proj.weight": rd_tensor(
                    100,
                    {"2_64": 0.4, "4_64": 0.1},
                ),
                "model.layers.0.mlp.up_proj.weight": rd_tensor(
                    200,
                    {"2_64": 0.3, "4_64": 0.12},
                ),
                "model.layers.0.input_layernorm.weight": rd_tensor(50, is_1d=True),
                "model.layers.1.mlp.experts.0.down_proj.weight": rd_tensor(
                    120,
                    {"2_64": 0.6, "4_64": 0.2},
                ),
                "model.layers.1.mlp.experts.1.down_proj.weight": rd_tensor(
                    120,
                    {"2_64": 0.6, "4_64": 0.2},
                ),
                "model.visual.blocks.0.attn.q_proj.weight": rd_tensor(
                    300,
                    {"2_64": 0.5, "4_64": 0.2},
                ),
            },
        }

    def build_manual_problem(self, table_defs):
        tensor_specs = []
        objective_tables = []
        tensor_meta = {}

        for idx, table_def in enumerate(table_defs):
            name = table_def["name"]
            cfg_rows = table_def["configs"]
            tensor_specs.append({
                "name": name,
                "num_params": table_def.get("num_params", 100 + idx),
                "valid_configs": [cfg for cfg, _, _, _ in cfg_rows],
                "rd_curve": {cfg: loss for cfg, loss, _, _ in cfg_rows},
                "prior": 1.0,
                "alpha": 1.0,
                "layer_idx": idx,
            })
            objective_tables.append({
                "name": name,
                "configs": [
                    {
                        "cfg": cfg,
                        "loss": loss,
                        "runtime": runtime,
                        "size": size,
                    }
                    for cfg, loss, runtime, size in cfg_rows
                ],
            })
            tensor_meta[name] = {
                "num_params": table_def.get("num_params", 100 + idx),
                "layer_idx": idx,
            }

        return tensor_specs, objective_tables, {}, tensor_meta

    def brute_force_frontier(self, tensor_specs, objective_tables, expert_members, tensor_meta):
        candidates = []
        for cfgs in product(*(table["configs"] for table in objective_tables)):
            selection = {
                table["name"]: cfg
                for table, cfg in zip(objective_tables, cfgs)
            }
            allocations, bits_dist, total_loss, total_size, total_runtime, average_bits = expand_allocations(
                tensor_specs,
                selection,
                expert_members,
                tensor_meta,
            )
            candidates.append({
                "signature": selection_signature(selection),
                "allocations": allocations,
                "bits_distribution": bits_dist,
                "total_loss": total_loss,
                "total_size_bytes": total_size,
                "runtime_proxy_bytes": total_runtime,
                "average_bits": average_bits,
                "num_tensors": len(allocations),
            })
        return pareto_prune(candidates)

    def frontier_rows(self, frontier):
        return [
            (
                candidate["signature"],
                round(candidate["total_loss"], 10),
                round(candidate["runtime_proxy_bytes"], 10),
                candidate["total_size_bytes"],
            )
            for candidate in frontier
        ]

    def test_dp_matches_bruteforce_on_tiny_problem(self):
        specs, objective_tables, expert_members, tensor_meta = self.build_manual_problem([
            {
                "name": "tensor_a",
                "configs": [
                    ((2, 32), 9.0, 4.0, 6),
                    ((4, 64), 4.0, 6.0, 9),
                    ((16, 0), 0.0, 9.0, 14),
                ],
            },
            {
                "name": "tensor_b",
                "configs": [
                    ((2, 32), 7.0, 3.0, 5),
                    ((4, 64), 3.0, 5.0, 7),
                    ((16, 0), 0.0, 7.0, 11),
                ],
            },
            {
                "name": "tensor_c",
                "configs": [
                    ((4, 64), 5.0, 2.0, 4),
                    ((8, 64), 2.0, 4.0, 6),
                    ((16, 0), 0.0, 6.0, 8),
                ],
            },
            {
                "name": "tensor_d",
                "configs": [
                    ((4, 64), 4.0, 3.0, 4),
                    ((16, 0), 0.0, 5.0, 7),
                ],
            },
        ])

        brute_force = self.brute_force_frontier(specs, objective_tables, expert_members, tensor_meta)
        brute_selected, _ = select_frontier_candidate(brute_force)
        frontier, selected, _ = search_frontier(
            specs,
            objective_tables,
            expert_members,
            tensor_meta,
            runtime_buckets=64,
            runtime_bucket_bytes=1.0,
        )

        self.assertEqual(self.frontier_rows(frontier), self.frontier_rows(brute_force))
        self.assertEqual(selected["signature"], brute_selected["signature"])

    def test_dp_bucket_collisions_keep_best_state(self):
        cases = [
            (
                "lower_loss_first",
                [
                    ((2, 32), 1.0, 40.0, 20),
                    ((4, 64), 2.0, 10.0, 10),
                ],
                (2, 32),
            ),
            (
                "lower_runtime_second",
                [
                    ((2, 32), 1.0, 40.0, 20),
                    ((4, 64), 1.0, 10.0, 30),
                ],
                (4, 64),
            ),
            (
                "smaller_size_third",
                [
                    ((2, 32), 1.0, 10.0, 30),
                    ((4, 64), 1.0, 10.0, 20),
                ],
                (4, 64),
            ),
        ]

        for label, configs, expected_cfg in cases:
            with self.subTest(label=label):
                specs, objective_tables, expert_members, tensor_meta = self.build_manual_problem([
                    {
                        "name": "tensor_bucket",
                        "configs": configs,
                    }
                ])
                frontier, selected, _ = search_frontier(
                    specs,
                    objective_tables,
                    expert_members,
                    tensor_meta,
                    runtime_buckets=1,
                    runtime_bucket_bytes=100.0,
                )
                expected_signature = (("tensor_bucket", expected_cfg[0], expected_cfg[1]),)
                self.assertEqual(len(frontier), 1)
                self.assertEqual(frontier[0]["signature"], expected_signature)
                self.assertEqual(selected["signature"], expected_signature)

    def test_fixed_tensor_factoring_matches_bruteforce(self):
        specs, objective_tables, expert_members, tensor_meta = self.build_manual_problem([
            {
                "name": "tensor_fixed",
                "configs": [
                    ((16, 0), 0.0, 8.0, 16),
                ],
            },
            {
                "name": "tensor_var_a",
                "configs": [
                    ((4, 64), 5.0, 4.0, 8),
                    ((16, 0), 0.0, 9.0, 16),
                ],
            },
            {
                "name": "tensor_var_b",
                "configs": [
                    ((4, 64), 4.0, 3.0, 7),
                    ((8, 64), 2.0, 5.0, 10),
                    ((16, 0), 0.0, 7.0, 14),
                ],
            },
        ])

        brute_force = self.brute_force_frontier(specs, objective_tables, expert_members, tensor_meta)
        brute_selected, _ = select_frontier_candidate(brute_force)
        frontier, selected, selection_meta = search_frontier(
            specs,
            objective_tables,
            expert_members,
            tensor_meta,
            runtime_buckets=32,
            runtime_bucket_bytes=1.0,
        )

        self.assertEqual(self.frontier_rows(frontier), self.frontier_rows(brute_force))
        self.assertEqual(selected["signature"], brute_selected["signature"])
        self.assertEqual(selection_meta["num_fixed_tensors"], 1)
        self.assertEqual(selection_meta["num_variable_tensors"], 2)

    def test_local_prune_keeps_fast_frontier_and_smallest_anchor(self):
        configs = [
            {"cfg": (2, 32), "loss": 0.6, "runtime": 10.0, "size": 10},
            {"cfg": (4, 64), "loss": 0.4, "runtime": 8.0, "size": 12},
            {"cfg": (8, 64), "loss": 0.2, "runtime": 6.0, "size": 18},
            {"cfg": (3, 64), "loss": 0.8, "runtime": 12.0, "size": 6},
        ]

        pruned = prune_local_configs(configs)

        self.assertEqual({cfg["cfg"] for cfg in pruned}, {(8, 64), (3, 64)})

    def test_packed_expert_weight_resolves_top_k_and_active_factor(self):
        specs = [
            {
                "name": "model.layers.1.mlp.experts.gate_up_proj.weight",
            }
        ]
        moe_top_k = resolve_moe_top_k({"text_config": {"num_experts_per_tok": 2}}, specs)
        self.assertEqual(moe_top_k, 2.0)

        table = build_objective_tables(
            [
                {
                    "name": "model.layers.1.mlp.experts.gate_up_proj.weight",
                    "num_params": 8 * 16,
                    "valid_configs": [(4, 64), (16, 0)],
                    "rd_curve": {(4, 64): 0.25, (16, 0): 0.0},
                    "prior": 1.0,
                    "alpha": 1.0,
                    "layer_idx": 1,
                }
            ],
            {},
            {
                "model.layers.1.mlp.experts.gate_up_proj.weight": {
                    "shape": [8, 4, 4],
                    "num_params": 8 * 16,
                }
            },
            moe_top_k,
        )
        cfgs = {cfg["cfg"]: cfg for cfg in table[0]["configs"]}
        self.assertAlmostEqual(cfgs[(4, 64)]["active_factor"], 0.25)
        self.assertAlmostEqual(cfgs[(16, 0)]["active_factor"], 0.25)

        frontier, selected, _ = search_frontier(
            [
                {
                    "name": "model.layers.1.mlp.experts.gate_up_proj.weight",
                    "num_params": 8 * 16,
                    "valid_configs": [(4, 64), (16, 0)],
                    "rd_curve": {(4, 64): 0.25, (16, 0): 0.0},
                    "prior": 1.0,
                    "alpha": 1.0,
                    "layer_idx": 1,
                }
            ],
            table,
            {},
            {
                "model.layers.1.mlp.experts.gate_up_proj.weight": {
                    "shape": [8, 4, 4],
                    "num_params": 8 * 16,
                }
            },
            runtime_buckets=32,
            runtime_bucket_bytes=1.0,
        )
        self.assertEqual(
            sorted(round(candidate["runtime_proxy_bytes"], 10) for candidate in frontier),
            sorted([
                round(estimate_size(8 * 16, 4, 64) * 0.25, 10),
                round(estimate_size(8 * 16, 16, 0) * 0.25, 10),
            ]),
        )
        self.assertIn(selected["signature"], {candidate["signature"] for candidate in frontier})

    def test_knee_selector_picks_fastest_point_under_loss_cap(self):
        frontier = [
            {
                "signature": (("a", 4, 32),),
                "total_loss": 10.0,
                "total_size_bytes": 100.0,
                "runtime_proxy_bytes": 100.0,
            },
            {
                "signature": (("b", 4, 32),),
                "total_loss": 6.0,
                "total_size_bytes": 140.0,
                "runtime_proxy_bytes": 160.0,
            },
            {
                "signature": (("c", 4, 32),),
                "total_loss": 4.0,
                "total_size_bytes": 200.0,
                "runtime_proxy_bytes": 240.0,
            },
            {
                "signature": (("d", 4, 32),),
                "total_loss": 3.7,
                "total_size_bytes": 260.0,
                "runtime_proxy_bytes": 330.0,
            },
            {
                "signature": (("e", 4, 32),),
                "total_loss": 3.6,
                "total_size_bytes": 320.0,
                "runtime_proxy_bytes": 430.0,
            },
        ]

        selected, selection_meta = select_frontier_candidate(frontier)

        self.assertEqual(selection_meta["selection_method"], "knee_loss_cap_fastest_under_cap")
        self.assertEqual(selection_meta["loss_cap"], 4.0)
        self.assertEqual(selected["signature"], (("c", 4, 32),))

    def test_size_guardrail_filters_final_selection(self):
        frontier = [
            {
                "signature": (("fast", 4, 32),),
                "total_loss": 5.0,
                "total_size_bytes": 220.0,
                "runtime_proxy_bytes": 120.0,
            },
            {
                "signature": (("guardrail", 4, 32),),
                "total_loss": 5.2,
                "total_size_bytes": 180.0,
                "runtime_proxy_bytes": 150.0,
            },
            {
                "signature": (("small", 4, 32),),
                "total_loss": 7.0,
                "total_size_bytes": 100.0,
                "runtime_proxy_bytes": 210.0,
            },
        ]

        selected, selection_meta = select_frontier_candidate(frontier, size_guardrail_bytes=190.0)

        self.assertEqual(selection_meta["size_guardrail_bytes"], 190.0)
        self.assertEqual(selected["signature"], (("guardrail", 4, 32),))

    def test_frontier_search_selects_balanced_point(self):
        rd_data = self.make_rd_data()
        live_modules = {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.input_layernorm",
            "model.layers.1.mlp.switch_mlp.down_proj",
        }
        filtered, reason_counts, _ = filter_rd_by_live_modules(rd_data, live_modules)
        specs, expert_members = build_tensor_specs(filtered, filtered["total_layers"])
        moe_top_k = resolve_moe_top_k({"text_config": {"num_experts_per_tok": 1}}, specs)
        objective_tables = build_objective_tables(specs, expert_members, filtered["tensors"], moe_top_k)

        q_proj_table = next(table for table in objective_tables if table["name"] == "model.layers.0.self_attn.q_proj.weight")
        q_proj_cfgs = {cfg["cfg"]: cfg for cfg in q_proj_table["configs"]}
        self.assertAlmostEqual(q_proj_cfgs[(2, 64)]["norm_loss"], 1.0)
        self.assertAlmostEqual(q_proj_cfgs[(16, 0)]["norm_size"], 1.0)
        self.assertGreater(q_proj_cfgs[(4, 64)]["norm_loss"], 0.0)
        self.assertLess(q_proj_cfgs[(4, 64)]["norm_loss"], 1.0)
        self.assertIn("scaled_runtime", q_proj_cfgs[(4, 64)])

        frontier, selected, selection_meta = search_frontier(specs, objective_tables, expert_members, filtered["tensors"])
        result = build_result(selected, frontier, reason_counts, selection_meta)

        self.assertGreater(len(frontier), 1)
        self.assertIn(selected["signature"], {candidate["signature"] for candidate in frontier})
        self.assertEqual(result["selection_method"], "knee_loss_cap_fastest_under_cap")
        self.assertEqual(result["frontier_size"], len(frontier))
        self.assertEqual(result["excluded_tensor_count"], 1)
        self.assertEqual(result["excluded_reason_counts"]["not_live_in_mlxlm"], 1)
        self.assertEqual(len([point for point in result["frontier"] if point["selected"]]), 1)
        self.assertIn("selection_loss_cap", result)
        self.assertIn("selection_curve_size", result)

        self.assertEqual(result["allocations"]["model.layers.0.self_attn.q_proj.weight"]["bits"], 4)
        self.assertEqual(result["allocations"]["model.layers.0.mlp.up_proj.weight"]["bits"], 4)
        self.assertEqual(result["allocations"]["model.layers.1.mlp.experts.0.down_proj.weight"]["bits"], 4)
        self.assertEqual(result["allocations"]["model.layers.1.mlp.experts.1.down_proj.weight"]["bits"], 4)
        self.assertEqual(result["allocations"]["model.layers.0.input_layernorm.weight"]["bits"], 16)
        self.assertNotIn("model.visual.blocks.0.attn.q_proj.weight", result["allocations"])

        expected_runtime = (
            estimate_size(100, 4, 64)
            + estimate_size(200, 4, 64)
            + estimate_size(50, 16, 0)
            + estimate_size(240, 4, 64) * 0.5
        )
        self.assertAlmostEqual(result["runtime_proxy_bytes"], expected_runtime)
        self.assertLess(result["runtime_proxy_bytes"], result["total_size_bytes"])

        self.assertEqual(result["budget_bytes"], result["total_size_bytes"])
        self.assertEqual(result["budget_gb"], result["total_size_gb"])
        self.assertEqual(result["budget_utilization"], 1.0)
        self.assertEqual(result["objective"], "quality_speed_frontier")
        self.assertEqual(result["solver"], "frontier_dp_2d")
        self.assertEqual(result["selection_axes"], ["total_loss", "runtime_proxy_bytes"])
        self.assertEqual(result["size_role"], "guardrail_tiebreaker")
        self.assertIn("runtime_bucket_count", result)
        self.assertIn("runtime_bucket_bytes", result)
        self.assertIn("dp_peak_live_states", result)
        self.assertIn("num_fixed_tensors", result)
        self.assertIn("num_variable_tensors", result)
        self.assertLessEqual(result["dp_peak_live_states"], result["runtime_bucket_count"])
        self.assertIn("solver_runtime_ms", result)
        self.assertIn("sqnr_floor_db", result)
        self.assertIn("average_bits", result)
        self.assertIn("bits_distribution", result)
        self.assertEqual(result["bits_distribution"]["4"]["params"], 540)
        self.assertEqual(result["bits_distribution"]["16"]["params"], 50)


if __name__ == "__main__":
    unittest.main()
