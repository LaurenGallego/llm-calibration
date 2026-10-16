import pytest
import torch

from madcal.storage.states import AnswerStates, NullAnswer, StateChunk, StateStore

HIDDEN = 3


def open_answers(
    ids: list[str], lengths: list[int], roles: list[str] | None = None
) -> AnswerStates:
    total = sum(lengths)
    return AnswerStates(
        question_ids=tuple(ids),
        roles=tuple(roles or ["train"] * len(ids)),
        correct=torch.tensor([i % 2 == 0 for i in range(len(ids))]),
        states=torch.arange(total * HIDDEN, dtype=torch.float32).reshape(total, HIDDEN),
        offsets=torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
        targets=torch.arange(total),
    )


def begin(store: StateStore, run_id: str = "run-a") -> None:
    store.begin({"config_hash": "abc"}, run_id, torch.eye(4, HIDDEN), None)


def test_a_cache_round_trips_answers_and_nulls(tmp_path):
    store = StateStore(tmp_path)
    begin(store)
    store.write_chunk("run-a", 0, StateChunk(open_answers(["q1", "q2"], [2, 1]), ()))
    store.write_chunk(
        "run-a", 1, StateChunk(open_answers(["q3"], [3]), (NullAnswer("q4", "train", "x"),))
    )
    store.finish({"n_train": 4})
    cache = store.load()
    assert cache.answers.question_ids == ("q1", "q2", "q3")
    assert cache.answers.offsets.tolist() == [0, 2, 3, 6]
    assert cache.answers.states.shape == (6, HIDDEN)
    assert cache.nulls == (NullAnswer("q4", "train", "x"),)
    assert cache.manifest["n_train"] == 4
    assert torch.equal(cache.weight, torch.eye(4, HIDDEN))


def test_an_unfinished_cache_is_unreadable(tmp_path):
    store = StateStore(tmp_path)
    begin(store)
    store.write_chunk("run-a", 0, StateChunk(open_answers(["q1"], [1]), ()))
    with pytest.raises(ValueError, match="incomplete"):
        store.load()


def test_a_resumed_run_sees_every_stored_question_and_reopens_the_cache(tmp_path):
    store = StateStore(tmp_path)
    begin(store)
    store.write_chunk(
        "run-a", 0, StateChunk(open_answers(["q1"], [1]), (NullAnswer("q2", "train", "x"),))
    )
    store.finish({})
    begin(store, "run-b")
    assert store.stored_ids() == {"q1", "q2"}
    assert store.manifest()["runs"] == ["run-a", "run-b"]
    with pytest.raises(ValueError, match="incomplete"):
        store.load()


def test_a_chunk_from_a_run_that_never_began_is_refused(tmp_path):
    store = StateStore(tmp_path)
    begin(store)
    with pytest.raises(ValueError, match="never began"):
        store.write_chunk("run-z", 0, StateChunk(None, (NullAnswer("q", "train", "x"),)))


def test_a_tampered_head_is_refused(tmp_path):
    store = StateStore(tmp_path)
    begin(store)
    store.write_chunk("run-a", 0, StateChunk(open_answers(["q1"], [1]), ()))
    store.finish({})
    torch.save({"weight": torch.zeros(4, HIDDEN), "bias": None}, tmp_path / "head.pt")
    with pytest.raises(ValueError, match="sha256"):
        store.load()


def test_select_gathers_the_positions_of_each_question_in_order():
    answers = open_answers(["a", "b", "c"], [2, 1, 3])
    picked = answers.select([2, 0])
    assert picked.question_ids == ("c", "a")
    assert picked.offsets.tolist() == [0, 3, 5]
    assert picked.targets.tolist() == [3, 4, 5, 0, 1]
    assert picked.last_positions().tolist() == [2, 4]
    assert picked.positions().tolist() == [0, 0, 0, 1, 1]


def test_with_role_keeps_only_that_role():
    answers = open_answers(["a", "b", "c"], [1, 1, 1], ["train", "evaluation", "train"])
    assert answers.with_role("train").question_ids == ("a", "c")


def test_a_multiple_choice_target_must_be_the_chosen_option():
    with pytest.raises(ValueError, match="chosen index"):
        AnswerStates(
            question_ids=("q",),
            roles=("train",),
            correct=torch.tensor([True]),
            states=torch.zeros(1, HIDDEN),
            offsets=torch.tensor([0, 1]),
            targets=torch.tensor([7]),
            options=torch.tensor([[5, 6, 7, 8]]),
            chosen=torch.tensor([1]),
        )


def test_a_question_with_no_position_is_refused():
    with pytest.raises(ValueError, match="every question a position"):
        open_answers(["a", "b"], [1, 0])
