import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from madcal.benchmarks import MMLU, TriviaQA
from madcal.config import MMLUBenchmarkConfig, TriviaQABenchmarkConfig
from madcal.debate import Message
from madcal.models import Generation, TransformersAdapter
from madcal.orchestration.harvest import PARSE_FAILED, HarvestItem, harvest_chunk, harvest_items
from tests.models.test_transformers import TINY_MODEL, _is_cached

FIXTURES = Path(__file__).parent.parent / "fixtures"
TRIVIA = FIXTURES / "trivia_qa_sample.jsonl"
MMLU_SAMPLE = FIXTURES / "mmlu_sample.jsonl"
needs_model = pytest.mark.skipif(not _is_cached(), reason=f"{TINY_MODEL} is not cached")


def train_file(tmp_path: Path) -> Path:
    rows = [
        {
            "question_id": "tr_1",
            "question": "Which Lloyd-Webber musical premiered in the US on 10th December 1993",
            "answer": {"value": "Sunset Boulevard", "aliases": ["Sunset Boulevard"]},
        },
        {
            "question_id": "tr_2",
            "question": "What is the capital of France?",
            "answer": {"value": "Paris", "aliases": ["Paris"]},
        },
        {
            "question_id": "tr_3",
            "question": "What is the capital of Spain?",
            "answer": {"value": "Madrid", "aliases": ["Madrid"]},
        },
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def render(messages: Sequence[Message]) -> str:
    return messages[-1].content


def test_training_questions_sharing_text_with_an_evaluation_question_are_excluded(tmp_path):
    train = TriviaQA(source=train_file(tmp_path), split="train")
    evaluation = TriviaQA(source=TRIVIA)
    items, excluded = harvest_items(
        train, TriviaQABenchmarkConfig(split="train"), evaluation, TriviaQABenchmarkConfig()
    )
    assert excluded == 1
    assert [item.question.id for item in items if item.role == "train"] == [
        "trivia_qa/train/tr_2",
        "trivia_qa/train/tr_3",
    ]
    assert {item.question.id for item in items if item.role == "evaluation"} == {
        q.id for q in evaluation.load()
    }


def test_the_same_split_is_split_by_question_id():
    mmlu = MMLU(source=MMLU_SAMPLE, reasoning=False)
    items, excluded = harvest_items(
        mmlu, MMLUBenchmarkConfig(), mmlu, MMLUBenchmarkConfig(n_questions=4, sample_seed=1)
    )
    train = {item.question.id for item in items if item.role == "train"}
    evaluation = {item.question.id for item in items if item.role == "evaluation"}
    assert len(evaluation) == 4 and not train & evaluation
    assert len(train) + len(evaluation) + excluded - 4 == len(mmlu.load())
    assert excluded >= 4


@pytest.fixture(scope="module")
def adapter() -> TransformersAdapter:
    return TransformersAdapter(model_id=TINY_MODEL, dtype="float32", device="cpu", batch_size=2)


def scripted(adapter: TransformersAdapter, texts: list[str]):
    tokenizer = adapter._tokenizer

    def generate(prompts, n=1, temperature=0.0, max_tokens=16, seed=None):
        assert len(prompts) == len(texts)
        out = []
        for text in texts:
            tokens = tuple(tokenizer.tokenize(text)) if text else ("<|endoftext|>",)
            out.append([Generation(text, tuple(-0.1 for _ in tokens), tokens, "stop")])
        return out

    return generate


@needs_model
def test_open_answers_keep_every_generated_position_and_parse_failures_are_nulls(
    adapter, monkeypatch
):
    trivia = TriviaQA(source=TRIVIA)
    questions = trivia.load()[:3]
    texts = [" Sunset Boulevard", " Paris", ""]
    monkeypatch.setattr(adapter, "generate", scripted(adapter, texts))
    items = [HarvestItem(q, "train", trivia) for q in questions]
    chunk = harvest_chunk(items, adapter, render, 16)
    assert chunk.answers is not None
    assert [n.reason for n in chunk.nulls] == [PARSE_FAILED]
    assert chunk.nulls[0].question_id == questions[2].id
    lengths = chunk.answers.offsets.diff().tolist()
    assert lengths == [len(adapter._tokenizer.tokenize(t)) for t in texts[:2]]
    assert chunk.answers.correct.tolist() == [
        trivia.grade(questions[0], "Sunset Boulevard"),
        trivia.grade(questions[1], "Paris"),
    ]
    tokenizer = adapter._tokenizer
    expected = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(texts[0]))
    assert chunk.answers.targets[: lengths[0]].tolist() == expected


@needs_model
def test_a_multiple_choice_answer_is_the_letter_position_with_its_options(adapter, monkeypatch):
    mmlu = MMLU(source=MMLU_SAMPLE, reasoning=False)
    questions = mmlu.load()[:2]
    monkeypatch.setattr(adapter, "generate", scripted(adapter, ["Answer: C", "I am not sure"]))
    chunk = harvest_chunk(
        [HarvestItem(q, "evaluation", mmlu) for q in questions], adapter, render, 16
    )
    assert chunk.answers is not None and chunk.answers.options is not None
    assert chunk.answers.chosen is not None
    assert chunk.answers.chosen.tolist() == [2]
    assert chunk.answers.correct.tolist() == [questions[0].answer == "C"]
    assert chunk.answers.options.shape == (1, 4)
    assert len(set(chunk.answers.options[0].tolist())) == 4
    assert [n.question_id for n in chunk.nulls] == [questions[1].id]
    assert torch.equal(chunk.answers.targets, chunk.answers.options[0, 2:3])
