import re
from collections.abc import Sequence

import pytest

from madcal.debate import (
    PRESETS,
    VERBALIZED_INSTRUCTION,
    DebateQuestion,
    Message,
    SystemNull,
    preset,
    round_seed,
    run_debate,
)
from madcal.models import Generation, StubAdapter
from madcal.signals import SignalNull

TEMPLATE = "Answer: {letter}\nConfidence: {confidence}%"
QUESTIONS = [DebateQuestion(f"q{index}", f"Question number {index}?", 4) for index in range(4)]
SETTINGS = {"best_of_n": {"n_agents": 3}}


def render(messages: Sequence[Message]) -> str:
    return "\n".join(f"[{message.role}] {message.content}" for message in messages)


def extract(text: str, n_choices: int | None) -> str | None:
    assert n_choices == 4
    matches = re.findall(r"Answer: ([A-D])", text)
    return matches[-1] if matches else None


class RecordingAdapter(StubAdapter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[tuple[list[str], int]] = []
        self.seeds: list[int | None] = []

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, seed=None):
        self.calls.append((list(prompts), n))
        self.seeds.append(seed)
        return super().generate(prompts, n, temperature, max_tokens, seed)


def debate(name: str, adapter: StubAdapter | None = None, questions=QUESTIONS):
    config = preset(name, temperature=1.0, **SETTINGS.get(name, {}))
    adapter = adapter or StubAdapter(response_template=TEMPLATE)
    return run_debate(questions, config, adapter, render, extract, seed=0)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_runs_and_is_deterministic(name):
    first = debate(name)
    assert first == debate(name)
    assert [t.question_id for t in first] == [q.question_id for q in QUESTIONS]


def test_vanilla_stores_round_zero_plus_each_debate_round():
    transcripts = debate("vanilla")
    assert all(len(t.turns) == 3 * (2 + 1) for t in transcripts)
    assert {turn.round for turn in transcripts[0].turns} == {0, 1, 2}


def test_round_zero_is_one_call_sampling_every_agent_from_the_shared_prompt():
    adapter = RecordingAdapter(response_template=TEMPLATE)
    debate("vanilla", adapter)
    prompts, n = adapter.calls[0]
    assert n == 3
    assert len(prompts) == len(QUESTIONS)
    assert [call[1] for call in adapter.calls[1:]] == [1, 1]
    assert [len(call[0]) for call in adapter.calls[1:]] == [len(QUESTIONS) * 3] * 2


def test_round_zero_agents_are_independent_samples():
    transcripts = debate("vanilla", questions=[DebateQuestion("q", "Some question?", 4)] * 1)
    texts = [turn.text for turn in transcripts[0].round_turns(0)]
    samples = StubAdapter(response_template=TEMPLATE).generate(
        ["[user] Some question?"], n=3, temperature=1.0, seed=round_seed(0, 0)
    )[0]
    assert texts == [sample.text for sample in samples]


class ScriptedAdapter(StubAdapter):
    def __init__(self, script: list[list[str]]) -> None:
        super().__init__()
        self.script = script
        self.calls: list[tuple[list[str], int]] = []

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, seed=None):
        texts = self.script[len(self.calls)]
        self.calls.append((list(prompts), n))

        def make(text: str) -> Generation:
            return Generation(text=text, token_logprobs=None, tokens=None, finish_reason="stop")

        if n > 1:
            return [[make(text) for text in texts]]
        return [[make(text)] for text in texts]


SCRIPT = [
    ["Answer: A r0a0", "Answer: B r0a1", "Answer: C r0a2"],
    ["Answer: A r1a0", "Answer: B r1a1", "Answer: B r1a2"],
    ["Answer: C r2a0", "Answer: B r2a1", "Answer: B r2a2"],
]


def scripted_vanilla(script=SCRIPT):
    adapter = ScriptedAdapter(script)
    question = DebateQuestion("q", "Some question?", 4)
    transcript = run_debate(
        [question], preset("vanilla", temperature=1.0), adapter, render, extract, seed=0
    )[0]
    return transcript, adapter


def test_each_agent_sees_its_own_history_and_its_peers_latest_responses():
    _, adapter = scripted_vanilla()
    round_two_prompts, _ = adapter.calls[2]
    agent_one = round_two_prompts[1]
    assert agent_one.startswith("[user] Some question?")
    assert "[assistant] Answer: B r0a1" in agent_one
    assert "[assistant] Answer: B r1a1" in agent_one
    assert "```Answer: A r1a0```" in agent_one
    assert "```Answer: B r1a2```" in agent_one
    assert "```Answer: B r1a1```" not in agent_one
    assert "```Answer: A r0a0```" in agent_one
    assert agent_one.index("r0a0") < agent_one.index("[assistant] Answer: B r1a1")


def test_system_answer_aggregates_the_final_round_only():
    transcript, _ = scripted_vanilla()
    assert [turn.answer for turn in transcript.round_turns(2)] == ["C", "B", "B"]
    assert transcript.system.answer == "B"
    assert transcript.system.confidence == pytest.approx(2 / 3)


def test_three_way_split_in_the_final_round_is_a_null_tie():
    script = [*SCRIPT[:2], ["Answer: A x", "Answer: B y", "Answer: C z"]]
    transcript, _ = scripted_vanilla(script)
    assert transcript.system.reason is SystemNull.TIE


def test_verbalized_confidence_is_parsed_from_every_turn():
    transcript = debate("best_of_n")[0]
    assert all(turn.confidence.value is not None for turn in transcript.turns)
    best = max(turn.confidence.value or 0.0 for turn in transcript.turns)
    assert transcript.system.confidence == best


def test_unparseable_responses_are_nulls_all_the_way_up():
    adapter = StubAdapter(response_template="I would rather not say.")
    transcript = debate("best_of_n", adapter)[0]
    assert all(turn.answer is None for turn in transcript.turns)
    assert all(turn.confidence.reason is SignalNull.PARSE_FAILED for turn in transcript.turns)
    assert transcript.system.reason is SystemNull.NO_VALID_ANSWERS


def test_confidence_mode_none_records_not_applicable_and_asks_for_nothing():
    adapter = RecordingAdapter(response_template=TEMPLATE)
    transcript = debate("vanilla", adapter)[0]
    assert all(turn.confidence.reason is SignalNull.NOT_APPLICABLE for turn in transcript.turns)
    prompts = [prompt for call_prompts, _ in adapter.calls for prompt in call_prompts]
    assert all(VERBALIZED_INSTRUCTION not in prompt for prompt in prompts)


class ShortAdapter(StubAdapter):
    def generate(self, prompts, n=1, temperature=0.0, max_tokens=512, seed=None):
        results = super().generate(prompts, n, temperature, max_tokens, seed)
        return [list(samples)[:-1] for samples in results]


def test_adapter_returning_too_few_samples_raises():
    with pytest.raises(RuntimeError, match="samples"):
        debate("vanilla", ShortAdapter(response_template=TEMPLATE))


def test_duplicate_question_ids_are_refused():
    with pytest.raises(ValueError, match="unique"):
        debate("vanilla", questions=[QUESTIONS[0], QUESTIONS[0]])


def test_empty_batch_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        debate("vanilla", questions=[])


def test_preset_structure_cannot_be_overridden():
    with pytest.raises(ValueError, match="fixes"):
        preset("vanilla", n_agents=5, temperature=1.0)


def test_unknown_preset_is_refused():
    with pytest.raises(KeyError, match="unknown protocol"):
        preset("nonexistent", temperature=1.0)


def test_verbalized_mode_asks_for_confidence_in_every_request():
    adapter = RecordingAdapter(response_template=TEMPLATE)
    config = preset("best_of_n", n_agents=3, temperature=1.0)
    run_debate(QUESTIONS, config, adapter, render, extract, seed=0)
    prompts = [prompt for call_prompts, _ in adapter.calls for prompt in call_prompts]
    assert all(VERBALIZED_INSTRUCTION in prompt for prompt in prompts)


def test_each_round_samples_with_its_own_seed():
    adapter = RecordingAdapter(response_template=TEMPLATE)
    debate("vanilla", adapter)
    assert adapter.seeds == [round_seed(0, 0), round_seed(0, 1), round_seed(0, 2)]
    assert len(set(adapter.seeds)) == 3


def test_round_seed_is_stable_and_absent_without_a_debate_seed():
    assert round_seed(0, 1) == 4011020074
    assert round_seed(None, 1) is None
