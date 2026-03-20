import json
import tempfile
import unittest
from pathlib import Path

from allocator import (
    allocate_greedy,
    build_tensor_specs,
    compute_min_safe_size,
    estimate_size,
    extract_num_experts_per_tok,
    load_model_config,
)
from build_manifest import build_manifest_doc


def make_tensor(shape, rd_curve, sqnr, layer_idx=0):
    num_params = 1
    for dim in shape:
        num_params *= dim
    return {
        "shape": list(shape),
        "num_params": num_params,
        "layer_idx": layer_idx,
        "is_1d": False,
        "rd_curve": rd_curve,
        "sqnr": sqnr,
    }


class SpeedAwareAllocatorTests(unittest.TestCase):
    def test_speed_bias_zero_keeps_size_only_allocation(self):
        specs = [
            {
                "name": "model.layers.0.mlp.up_proj.weight",
                "shape": [32, 4],
                "num_params": 128,
                "valid_configs": [(4, 32), (8, 64), (16, 0)],
                "rd_curve": {
                    (4, 32): 0.7,
                    (8, 64): 0.1,
                    (16, 0): 0.0,
                },
                "prior": 1.0,
                "alpha": 1.0,
                "layer_idx": 0,
                "active_factor": 1.0,
                "runtime_family": None,
            },
            {
                "name": "model.layers.1.mlp.up_proj.weight",
                "shape": [32, 4],
                "num_params": 128,
                "valid_configs": [(4, 32), (8, 64), (16, 0)],
                "rd_curve": {
                    (4, 32): 0.4,
                    (8, 64): 0.2,
                    (16, 0): 0.0,
                },
                "prior": 1.0,
                "alpha": 1.0,
                "layer_idx": 1,
                "active_factor": 1.0,
                "runtime_family": None,
            },
        ]
        min_safe = compute_min_safe_size(specs)
        one_upgrade = estimate_size(128, 8, 64) - estimate_size(128, 4, 32)

        result = allocate_greedy(specs, min_safe + one_upgrade, speed_bias=0.0)

        self.assertEqual(result["objective"], "size_only")
        self.assertEqual(result["allocations"]["model.layers.0.mlp.up_proj.weight"]["bits"], 8)
        self.assertEqual(result["allocations"]["model.layers.1.mlp.up_proj.weight"]["bits"], 4)
        self.assertEqual(result["total_size_bytes"], min_safe + one_upgrade)

    def test_speed_bias_prefers_grouped_sparse_moe_upgrade(self):
        rd_curve = {
            "4_32": 0.6,
            "8_64": 0.2,
            "16_0": 0.0,
        }
        sqnr = {
            "4_32": 12.0,
            "8_64": 18.0,
            "16_0": 100.0,
        }
        rd_data = {
            "total_layers": 1,
            "model_config": {"num_experts_per_tok": 1},
            "tensors": {
                "model.layers.0.self_attn.k_proj.weight": make_tensor((32, 4), rd_curve, sqnr),
                "model.layers.0.mlp.experts.0.down_proj": make_tensor((16, 4), rd_curve, sqnr),
                "model.layers.0.mlp.experts.1.down_proj": make_tensor((16, 4), rd_curve, sqnr),
            },
        }

        specs, expert_members = build_tensor_specs(rd_data, total_layers=1)
        spec_map = {spec["name"]: spec for spec in specs}
        moe_name = "model.layers.0.mlp.experts.*.down_proj"

        self.assertAlmostEqual(spec_map[moe_name]["active_factor"], 0.5)
        self.assertEqual(spec_map[moe_name]["runtime_family"], "moe_experts")

        min_safe = compute_min_safe_size(specs)
        one_upgrade = estimate_size(128, 8, 64) - estimate_size(128, 4, 32)

        zero_bias = allocate_greedy(
            specs,
            min_safe + one_upgrade,
            expert_members=expert_members,
            speed_bias=0.0,
        )
        speed_bias = allocate_greedy(
            specs,
            min_safe + one_upgrade,
            expert_members=expert_members,
            speed_bias=0.5,
        )

        self.assertEqual(
            zero_bias["allocations"]["model.layers.0.self_attn.k_proj.weight"]["bits"],
            8,
        )
        self.assertEqual(
            speed_bias["allocations"]["model.layers.0.self_attn.k_proj.weight"]["bits"],
            4,
        )
        self.assertEqual(
            speed_bias["allocations"]["model.layers.0.mlp.experts.0.down_proj"]["bits"],
            8,
        )
        self.assertEqual(
            speed_bias["allocations"]["model.layers.0.mlp.experts.1.down_proj"]["bits"],
            8,
        )
        self.assertLess(speed_bias["runtime_proxy_bytes"], zero_bias["runtime_proxy_bytes"])

    def test_packed_expert_active_factor_uses_shape(self):
        rd_data = {
            "total_layers": 1,
            "model_config": {"num_experts_per_tok": 2},
            "tensors": {
                "model.layers.0.mlp.experts.gate_up_proj": make_tensor(
                    (8, 4, 4),
                    {"4_32": 0.5, "16_0": 0.0},
                    {"4_32": 12.0, "16_0": 100.0},
                ),
            },
        }

        specs, _ = build_tensor_specs(rd_data, total_layers=1)

        self.assertEqual(len(specs), 1)
        self.assertAlmostEqual(specs[0]["active_factor"], 0.25)
        self.assertEqual(specs[0]["runtime_family"], "moe_experts")

    def test_model_config_helpers_and_provenance(self):
        self.assertEqual(
            extract_num_experts_per_tok(
                {
                    "num_experts_per_tok": 4,
                    "text_config": {"num_experts_per_tok": 2},
                }
            ),
            2,
        )
        self.assertEqual(
            extract_num_experts_per_tok({"num_experts_per_tok": 4}),
            4,
        )
        self.assertIsNone(extract_num_experts_per_tok({}))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"text_config": {"num_experts_per_tok": 3}}))
            self.assertEqual(
                load_model_config(tmpdir),
                {"num_experts_per_tok": 3},
            )

        manifest = build_manifest_doc(
            allocation={
                "budget_gb": 19.0,
                "sqnr_floor_db": 9.0,
                "speed_bias": 0.4,
                "objective": "size_runtime_tax",
                "runtime_proxy_gb": 17.5,
                "total_params": 128,
                "bits_distribution": {"4": {"params": 128, "percentage": 100.0}},
                "total_size_gb": 18.0,
                "average_bits": 4.0,
                "solver": "greedy",
                "solver_runtime_ms": 1.5,
                "total_loss": 0.2,
            },
            model_name="demo",
            total_layers=1,
            shards={"model-00001.safetensors": {"file": "model-00001.safetensors", "tensors": {}}},
        )

        self.assertEqual(manifest["config"]["speed_bias"], 0.4)
        self.assertEqual(manifest["config"]["objective"], "size_runtime_tax")
        self.assertEqual(manifest["summary"]["runtime_proxy_gb"], 17.5)


if __name__ == "__main__":
    unittest.main()
