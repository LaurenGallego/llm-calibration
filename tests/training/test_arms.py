import math

import pytest
import torch

from madcal.models.vae_head import CalibratedHead, load_vae
from madcal.storage.states import AnswerStates
from madcal.training import (
    OutputHead,
    batched_brier,
    brier_loss,
    build_arm,
    in_selection,
    split,
    train,
)

HIDDEN, VOCAB = 4, 5


def head() -> OutputHead:
    generator = torch.Generator().manual_seed(0)
    return OutputHead(torch.randn(VOCAB, HIDDEN, generator=generator), None, "cpu")


def answers(n: int, seed: int = 0, positions: int = 2) -> AnswerStates:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(n * positions, HIDDEN, generator=generator)
    return AnswerStates(
        question_ids=tuple(f"q{i}" for i in range(n)),
        roles=("train",) * n,
        correct=states[positions - 1 :: positions, 0] > 0,
        states=states,
        offsets=torch.arange(0, n * positions + 1, positions),
        targets=torch.randint(0, VOCAB, (n * positions,), generator=generator),
    )


def base_confidence(output: OutputHead, data: AnswerStates) -> torch.Tensor:
    return output.base_log_confidence(data).exp()


@pytest.mark.parametrize("name", ["temperature", "hidden_temperature", "platt", "vae"])
def test_every_arm_starts_at_the_base_model(name):
    output, data = head(), answers(8)
    settings = {"latent_dim": 3} if name == "vae" else {}
    arm = build_arm(name, output, HIDDEN, 0, **settings)
    with torch.no_grad():
        assert arm.confidence(data).tolist() == pytest.approx(
            base_confidence(output, data).tolist(), abs=1e-5
        )


def test_the_probe_starts_at_one_half():
    arm = build_arm("probe", head(), HIDDEN, 0)
    with torch.no_grad():
        assert arm.confidence(answers(3)).tolist() == pytest.approx([0.5] * 3)


def test_the_temperature_grid_recovers_a_planted_temperature():
    weight = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    output = OutputHead(weight, None, "cpu")
    planted, rows = 2.0, []
    for gap, n in ((1.0, 400), (3.0, 400)):
        n_correct = round(n / (1 + math.exp(-gap / planted)))
        rows += [(gap, True)] * n_correct + [(gap, False)] * (n - n_correct)
    data = AnswerStates(
        question_ids=tuple(f"q{i}" for i in range(len(rows))),
        roles=("train",) * len(rows),
        correct=torch.tensor([correct for _, correct in rows]),
        states=torch.tensor([[gap, 0.0] for gap, _ in rows]),
        offsets=torch.arange(len(rows) + 1),
        targets=torch.zeros(len(rows), dtype=torch.long),
    )
    arm = build_arm("temperature", output, 2, 0, grid_points=251)
    arm.fit_directly(data, 64)
    assert arm.diagnostics(data)["temperature"] == pytest.approx(planted, abs=0.05)
    capped = build_arm("temperature", output, 2, 0, max_temperature=1.5)
    with pytest.raises(ValueError, match="edge of the grid"):
        capped.fit_directly(data, 64)


def test_training_the_probe_on_a_learnable_task_beats_its_start():
    output, data = head(), answers(400)
    fit, selection = split(data, 0.25, 0)
    arm = build_arm("probe", output, HIDDEN, 0)
    start = batched_brier(arm, selection, 64)
    result = train(arm, fit, selection, 0.05, 200, 64, 50, 0)
    assert result.selection_brier < start - 0.05
    assert result.selection_brier == pytest.approx(batched_brier(arm, selection, 64))
    assert result.curve[0] == (0.0, pytest.approx(start))


def test_the_vae_keeps_its_mean_and_output_bias_at_zero_and_loads_as_an_evaluation_head(tmp_path):
    output, data = head(), answers(64)
    arm = build_arm("vae", output, HIDDEN, 0, latent_dim=3, mc_samples=4)
    optimizer = torch.optim.AdamW(arm.parameters(), lr=0.05, weight_decay=0.0)
    for _ in range(5):
        loss = brier_loss(arm.confidence(data), data.correct)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    state = arm.checkpoint()
    assert torch.count_nonzero(state["enc.0.0.weight"][:3]) == 0
    assert torch.count_nonzero(state["enc.0.0.bias"][:3]) == 0
    assert torch.count_nonzero(state["dec.0.0.bias"]) == 0
    assert torch.count_nonzero(state["dec.0.0.weight"]) > 0
    torch.save(state, tmp_path / "vae.pt")
    loaded = load_vae(tmp_path / "vae.pt", output.weight)
    evaluation = CalibratedHead(output.weight, None, 1.0, loaded, 4, 0)
    replayed = build_arm("vae", output, HIDDEN, 0, latent_dim=3, mc_samples=4)
    replayed.load_checkpoint(state)
    with torch.no_grad():
        replayed_log_probs = replayed.log_probs(data)
        assert replayed_log_probs is not None
        assert torch.allclose(evaluation(data.states), replayed_log_probs, atol=1e-6)


def test_a_wide_noise_lowers_the_vae_confidence():
    output, data = head(), answers(16)
    arm = build_arm("vae", output, HIDDEN, 0, latent_dim=3, mc_samples=8)
    state = arm.checkpoint()
    state["enc.0.0.bias"][3:] = 4.0
    state["dec.0.0.weight"] = torch.ones_like(state["dec.0.0.weight"])
    arm.load_checkpoint(state)
    with torch.no_grad():
        lowered = arm.confidence(data)
    assert float(lowered.mean()) != pytest.approx(float(base_confidence(output, data).mean()))
    assert arm.diagnostics(data)["sigma_mean"] == pytest.approx(math.exp(2.0), rel=1e-5)


def test_the_selection_split_is_a_stable_function_of_the_id():
    ids = [f"q{i}" for i in range(2000)]
    first, again = in_selection(ids, 0.1, 0), in_selection(ids, 0.1, 0)
    assert first == again
    assert 150 < sum(first) < 250
    assert in_selection(ids, 0.1, 1) != first


def test_an_empty_side_of_the_split_is_refused():
    with pytest.raises(ValueError, match="empty"):
        split(answers(1), 0.5, 0)
