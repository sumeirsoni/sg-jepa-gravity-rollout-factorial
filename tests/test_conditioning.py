import pytest
import torch

from sg_jepa.conditioning import condition_actions, resolve_gravity_action_index


def test_square_gravity_defaults_to_last_coordinate() -> None:
    assert resolve_gravity_action_index(3) == 2


def test_constant_conditioning_zeros_normalized_gravity() -> None:
    actions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    conditioned = condition_actions(actions, mode="constant")
    assert torch.equal(conditioned, torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]))
    assert torch.equal(actions, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_constant_conditioning_accepts_explicit_index() -> None:
    actions = torch.tensor([[1.0, 2.0, 3.0]])
    conditioned = condition_actions(actions, mode="constant", gravity_action_index=0)
    assert torch.equal(conditioned, torch.tensor([[0.0, 2.0, 3.0]]))


def test_unknown_conditioning_mode_fails() -> None:
    with pytest.raises(ValueError, match="gravity conditioning"):
        condition_actions(torch.zeros(1, 3), mode="shuffled")
