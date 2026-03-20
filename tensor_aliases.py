"""Shared tensor-to-module alias helpers for MINT + MLX."""

import re
from typing import Iterable, Set

PROJ_ALIASES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
REVERSE_PROJ_ALIASES = {value: key for key, value in PROJ_ALIASES.items()}

_SUFFIX_PATTERN = re.compile(r"(.+)\.(weight|bias)$")
_EXPERT_PATTERN = re.compile(r"(.+)\.experts\.(\d+)\.(.+)")
_PACKED_EXPERT_PATTERN = re.compile(r"(.+)\.experts\.(gate_up_proj|down_proj)$")


def _switch_proj_aliases(proj: str) -> Set[str]:
    aliases = {proj}
    mapped = PROJ_ALIASES.get(proj)
    if mapped:
        aliases.add(mapped)
    mapped = REVERSE_PROJ_ALIASES.get(proj)
    if mapped:
        aliases.add(mapped)
    return aliases


def iter_tensor_aliases(name: str) -> Iterable[str]:
    """Yield tensor and module aliases used by bridge + frontier filtering."""
    seen = set()
    queue = [name]

    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        yield current

        match = _SUFFIX_PATTERN.match(current)
        if match:
            queue.append(match.group(1))

        if current.startswith("model.language_model."):
            queue.append("language_model.model." + current[len("model.language_model."):])
        elif current.startswith("language_model.model."):
            queue.append("model.language_model." + current[len("language_model.model."):])

        if "gate_up_proj" in current:
            queue.append(current.replace("gate_up_proj", "gate_proj"))
            queue.append(current.replace("gate_up_proj", "up_proj"))

        match = _EXPERT_PATTERN.match(current)
        if match:
            prefix, _, proj = match.groups()
            for alias in _switch_proj_aliases(proj):
                queue.append(f"{prefix}.switch_mlp.{alias}")

        match = _PACKED_EXPERT_PATTERN.match(current)
        if match:
            prefix, proj = match.groups()
            if proj == "gate_up_proj":
                queue.append(f"{prefix}.switch_mlp.gate_proj")
                queue.append(f"{prefix}.switch_mlp.up_proj")
                queue.append(f"{prefix}.switch_mlp.w1")
                queue.append(f"{prefix}.switch_mlp.w3")
            else:
                queue.append(f"{prefix}.switch_mlp.down_proj")
                queue.append(f"{prefix}.switch_mlp.w2")

        parts = current.rsplit(".", 1)
        if len(parts) == 2:
            stem, tail = parts
            for alias in _switch_proj_aliases(tail):
                if alias != tail:
                    queue.append(f"{stem}.{alias}")


def tensor_aliases(name: str) -> Set[str]:
    return set(iter_tensor_aliases(name))
