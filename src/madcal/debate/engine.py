"""The debate round loop."""

import hashlib
from collections.abc import Sequence

from madcal.debate.base import (
    AgentTurn,
    AnswerExtractor,
    ConfidenceMode,
    DebateConfig,
    DebateQuestion,
    DebateTranscript,
    Message,
    Renderer,
    Role,
    agent_id,
    aggregation_registry,
    confidence_mode_registry,
)
from madcal.models import Generation, ModelAdapter

PEER_HEADER = "These are the responses to the question from other agents:"
PEER_TEMPLATE = "One agent response: ```{text}```"
UPDATE_REQUEST = (
    "Using the reasoning from other agents as additional advice, give an updated answer. "
    "Examine your response and the other agents' responses step by step, and answer in "
    "the same format as before."
)


def run_debate(
    questions: Sequence[DebateQuestion],
    config: DebateConfig,
    adapter: ModelAdapter,
    render: Renderer,
    extract_answer: AnswerExtractor,
    seed: int | None = None,
) -> list[DebateTranscript]:
    """Run one debate per question, all questions advancing one round at a time."""
    if not questions:
        raise ValueError("run_debate needs at least one question")
    ids = [question.question_id for question in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("question ids must be unique within a debate batch")

    mode = confidence_mode_registry.get(config.confidence_mode)
    rule = aggregation_registry.get(config.aggregation)
    agents = [agent_id(index) for index in range(config.n_agents)]

    histories = [
        [[Message(Role.USER, opening_request(question, mode))] for _ in agents]
        for question in questions
    ]
    opening = _generate(
        adapter,
        [render(per_agent[0]) for per_agent in histories],
        config.n_agents,
        config,
        round_seed(seed, 0),
    )
    latest = [
        [
            _turn(agents[index], 0, sample, question, extract_answer, mode)
            for index, sample in enumerate(samples)
        ]
        for question, samples in zip(questions, opening, strict=True)
    ]
    turns = [list(round_turns) for round_turns in latest]

    for round_ in range(1, config.n_rounds + 1):
        prompts = []
        for question_index, per_agent in enumerate(histories):
            previous = latest[question_index]
            for index, history in enumerate(per_agent):
                peers = [turn.text for turn in previous if turn.agent_id != agents[index]]
                history.append(Message(Role.ASSISTANT, previous[index].text))
                history.append(Message(Role.USER, update_request(peers, mode)))
                prompts.append(render(history))
        samples = _generate(adapter, prompts, 1, config, round_seed(seed, round_))
        latest = [
            [
                _turn(
                    agents[index],
                    round_,
                    samples[question_index * config.n_agents + index][0],
                    questions[question_index],
                    extract_answer,
                    mode,
                )
                for index in range(config.n_agents)
            ]
            for question_index in range(len(questions))
        ]
        for question_index, round_turns in enumerate(latest):
            turns[question_index].extend(round_turns)

    return [
        DebateTranscript(
            question_id=question.question_id,
            n_agents=config.n_agents,
            n_rounds=config.n_rounds,
            turns=tuple(turns[question_index]),
            system=rule.aggregate(latest[question_index]),
        )
        for question_index, question in enumerate(questions)
    ]


def opening_request(question: DebateQuestion, mode: ConfidenceMode) -> str:
    """Return the round-0 request an agent receives."""
    if mode.instruction is None:
        return question.text
    return f"{question.text}\n\n{mode.instruction}"


def update_request(peer_texts: Sequence[str], mode: ConfidenceMode) -> str:
    """Return the request an agent receives after seeing its peers' previous responses."""
    parts = [PEER_HEADER, *(PEER_TEMPLATE.format(text=text) for text in peer_texts), UPDATE_REQUEST]
    if mode.instruction is not None:
        parts.append(mode.instruction)
    return "\n\n".join(parts)


def round_seed(seed: int | None, round_: int) -> int | None:
    """Return an independent seed for one round derived from the debate seed."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}:{round_}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _generate(
    adapter: ModelAdapter,
    prompts: Sequence[str],
    n: int,
    config: DebateConfig,
    seed: int | None,
) -> list[list[Generation]]:
    results = adapter.generate(
        prompts,
        n=n,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=seed,
    )
    if len(results) != len(prompts):
        raise RuntimeError(f"adapter returned {len(results)} results for {len(prompts)} prompts")
    for samples in results:
        if len(samples) != n:
            raise RuntimeError(f"adapter returned {len(samples)} samples, expected {n}")
    return [list(samples) for samples in results]


def _turn(
    agent: str,
    round_: int,
    generation: Generation,
    question: DebateQuestion,
    extract_answer: AnswerExtractor,
    mode: ConfidenceMode,
) -> AgentTurn:
    return AgentTurn(
        agent_id=agent,
        round=round_,
        text=generation.text,
        answer=extract_answer(generation.text, question.n_choices),
        confidence=mode.parse(generation.text),
        finish_reason=generation.finish_reason,
    )
