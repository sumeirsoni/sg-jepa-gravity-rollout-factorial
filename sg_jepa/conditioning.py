"""Apply controlled gravity inputs for world-model experiments."""

from __future__ import annotations

import torch

GRAVITY_CONDITIONING_MODES = frozenset({"correct", "constant"})


def resolve_gravity_action_index(action_dim: int, configured: int | None = None) -> int:
    """Return the gravity coordinate for a task action vector."""

    action_dim = int(action_dim)
    if configured is not None:
        index = int(configured)
        if index < 0 or index >= action_dim:
            raise ValueError(
                f"gravity_action_index={index} is outside action dimension {action_dim}"
            )
        return index
    if action_dim == 3:
        return 2
    if action_dim in {1, 4, 6}:
        return 0
    raise ValueError(
        "gravity_action_index is required for an unsupported action dimension: "
        f"{action_dim}"
    )


def condition_actions(
    actions: torch.Tensor,
    *,
    mode: str,
    gravity_action_index: int | None = None,
) -> torch.Tensor:
    """Return action vectors with either correct or constant gravity conditioning.

    Action vectors are already normalized with training-split statistics. The
    normalized training mean is therefore zero.
    """

    if mode not in GRAVITY_CONDITIONING_MODES:
        raise ValueError(
            f"gravity conditioning must be one of {sorted(GRAVITY_CONDITIONING_MODES)}, "
            f"got {mode!r}"
        )
    if actions.ndim < 1:
        raise ValueError(f"actions must have at least one dimension, got {actions.ndim}")
    if mode == "correct":
        return actions
    index = resolve_gravity_action_index(actions.shape[-1], gravity_action_index)
    conditioned = actions.clone()
    conditioned[..., index] = 0
    return conditioned


__all__ = [
    "GRAVITY_CONDITIONING_MODES",
    "condition_actions",
    "resolve_gravity_action_index",
]
