"""Named debate protocols as switch combinations."""

from typing import Any

from madcal.debate.base import DebateConfig

PRESETS: dict[str, dict[str, Any]] = {
    "best_of_n": {
        "n_rounds": 0,
        "confidence_mode": "verbalized",
        "aggregation": "argmax_confidence",
    },
    "vanilla": {
        "n_agents": 3,
        "n_rounds": 2,
        "confidence_mode": "none",
        "aggregation": "majority",
    },
}


def preset(name: str, **settings: Any) -> DebateConfig:
    """Return the named protocol completed with settings it does not fix."""
    if name not in PRESETS:
        raise KeyError(f"unknown protocol {name!r}; known: {sorted(PRESETS)}")
    structure = PRESETS[name]
    fixed = sorted(structure.keys() & settings.keys())
    if fixed:
        raise ValueError(f"protocol {name!r} fixes {fixed}; overriding them changes the protocol")
    return DebateConfig.model_validate({**structure, **settings})
