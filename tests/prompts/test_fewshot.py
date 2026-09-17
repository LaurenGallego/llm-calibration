"""Few-shot completion rendering.

Three silent failures live in this package:

- The target's own answer leaking into its prompt. Accuracy jumps toward 100%,
  ECE goes strange, and nothing raises.
- The template drifting from lm-evaluation-harness'. A moved space or separator
  retokenises the answer, shifting every logprob in the project and quietly
  breaking comparability with published MMLU numbers.
- Continuations falling out of `choices` order. ChoiceScores aligns logprobs to
  this tuple by position, so a reorder attaches confidences to the wrong option.
"""

from pathlib import Path

import pytest

from madcal.benchmarks import MMLU
from madcal.prompts import ANSWER_CUE, BLOCK_SEPARATOR, FewShotCompletion

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mmlu_sample.jsonl"

# Transcribed from lm-evaluation-harness: the per-subject `description`, then
# `doc_to_text` = "{{question.strip()}}\nA. ...\nD. ...\nAnswer:" per block,
# blocks separated by a blank line, exemplars carrying their answer.
EXPECTED_ONE_SHOT = """\
The following are multiple choice questions (with answers) about anatomy.

A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral
A. paralysis of the facial muscles.
B. paralysis of the facial muscles and loss of taste.
C. paralysis of the facial muscles, loss of taste and lacrimation.
D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.
Answer: A

A "dished face" profile is often associated with
A. a protruding mandible due to reactivation of the condylar cartilage by acromegaly.
B. a recessive maxilla due to failure of elongation of the cranial base.
C. an enlarged frontal bone due to hydrocephaly.
D. defective development of the maxillary air sinus.
Answer:"""


@pytest.fixture
def anatomy():
    """The four anatomy questions from the committed fixture, in id order."""
    return MMLU(source=FIXTURE, subjects=["anatomy"]).load()


def render(n_shots, question, examples=()):
    """Shorthand so each test states only what it is varying."""
    return FewShotCompletion(n_shots=n_shots).render(question, examples)


def test_target_answer_does_not_leak_into_its_own_prompt(anatomy):
    """The highest-value test in this package.

    Exemplars must show their answers; the target must not. One wrong
    `include_answer` inverts that and the model is handed the answer key.
    """
    target = anatomy[3]
    examples = anatomy[:2]
    blocks = render(2, target, examples).text.split(BLOCK_SEPARATOR)

    # Scoped to the target block on purpose: exemplar answers legitimately appear
    # elsewhere in the prompt, so searching the whole text would either fail for
    # the wrong reason or pass by luck when the letters happen to differ.
    target_block = blocks[-1]
    assert target_block.endswith(ANSWER_CUE), (
        f"target block must end at the bare cue, got:\n{target_block!r}"
    )
    assert f"{ANSWER_CUE} {target.answer}" not in target_block, (
        f"target answer {target.answer!r} leaked into its own block:\n{target_block}"
    )

    # The mirror image: if include_answer were wired False everywhere, few-shot
    # would silently degrade to zero-shot with extra questions attached.
    for example, block in zip(examples, blocks[1:-1], strict=True):
        assert block.endswith(f"{ANSWER_CUE} {example.answer}"), (
            f"exemplar {example.id} should show its answer, got:\n{block!r}"
        )


def test_rendered_text_matches_the_harness_template(anatomy):
    """Golden text. Pins the exact string the model sees.

    Not hand-computed from first principles -- it is a transcription of
    lm-evaluation-harness' MMLU template. Its job is to make any drift in
    separators, the answer cue, option lettering, or the subject wording fail
    loudly rather than silently changing every logprob in the run.
    """
    assert render(1, anatomy[1], anatomy[:1]).text == EXPECTED_ONE_SHOT


def test_subject_is_written_without_underscores(anatomy):
    """The harness writes "about world religions", not "about world_religions".

    The metadata value keeps the underscore -- it is the HF config name and the
    topic-drift partition key -- so only the wording the model sees changes.
    """
    question = MMLU(source=FIXTURE, subjects=["world_religions"]).load()[0]
    assert question.metadata["subject"] == "world_religions"
    assert render(0, question).text.startswith(
        "The following are multiple choice questions (with answers) about world religions."
    )


def test_continuations_align_with_choices(anatomy):
    """Position is the only link between a logprob and an option."""
    question = anatomy[0]
    rendered = render(0, question)
    assert rendered.continuations is not None
    assert rendered.continuations == (" A", " B", " C", " D")
    assert len(rendered.continuations) == len(question.choices)


@pytest.mark.parametrize(
    ("n_shots", "n_examples"),
    [(5, 2), (0, 1), (2, 3)],
    ids=["too-few", "unwanted-example", "too-many"],
)
def test_exemplar_count_must_match_n_shots(anatomy, n_shots, n_examples):
    """A 3-shot render recorded under a 5-shot condition name is a wrong number
    with a correct-looking label."""
    with pytest.raises(ValueError, match="examples"):
        render(n_shots, anatomy[3], anatomy[:n_examples])


def test_exemplars_must_share_the_targets_subject(anatomy):
    """lm-eval draws exemplars from the same subject's dev split, so a mismatch
    means the caller wired it wrong rather than that a new condition was wanted."""
    astronomy = MMLU(source=FIXTURE, subjects=["astronomy"]).load()
    with pytest.raises(ValueError, match="subject"):
        render(1, anatomy[0], astronomy[:1])
