"""Few-shot completion prompting, following lm-evaluation-harness' MMLU template.

The format this produces, for n_shots=1 on an MMLU question:

    The following are multiple choice questions (with answers) about anatomy.

    What is the largest organ in the human body?
    A. Liver
    B. Skin
    C. Heart
    D. Brain
    Answer: B

    A lesion causing compression of the facial nerve will cause ipsilateral
    A. paralysis of the facial muscles.
    B. paralysis of the facial muscles and loss of taste.
    C. ...
    D. ...
    Answer:

Exemplar blocks end with their answer; the target block ends at the bare cue.
Continuations are then (" A", " B", " C", " D") -- the space lives on the
continuation, never on the end of `text`.

This is the base-model-compatible protocol: plain text completion, no chat
template, so it can be applied unchanged to base, SFT and DPO checkpoints. That
is the point -- holding the protocol fixed is what stops prompt format from
being confounded with alignment stage.
"""

from collections.abc import Sequence

from caldrift.benchmarks import LETTERS, Question
from caldrift.prompts.base import CONTINUATION_PREFIX, RenderedPrompt, register_prompt

# lm-evaluation-harness prepends this per subject. {subject} is filled from
# Question.metadata.
DEFAULT_DESCRIPTION = "The following are multiple choice questions (with answers) about {subject}."

# Between the description, each exemplar, and the target.
BLOCK_SEPARATOR = "\n\n"

# Ends every block. Exemplars continue past it with their answer; the target does not.
ANSWER_CUE = "Answer:"


@register_prompt("fewshot_completion")
class FewShotCompletion:
    name = "fewshot_completion"

    def __init__(
        self,
        n_shots: int = 5,
        description: str | None = DEFAULT_DESCRIPTION,
    ) -> None:
        if n_shots < 0:
            raise ValueError(f"FewShotCompletion.n_shots must be non-negative, got {n_shots}")

        self.n_shots = n_shots
        self.description = description

    def render(
        self,
        question: Question,
        examples: Sequence[Question] = (),
    ) -> RenderedPrompt:
        if len(examples) != self.n_shots:
            raise ValueError(
                f"FewShotCompletion.render got {len(examples)} examples, "
                f"but config asked for {self.n_shots}"
            )

        for example in examples:
            if example.metadata.get("subject") != question.metadata.get("subject"):
                raise ValueError(
                    f"FewShotCompletion.render got an example with subject "
                    f"{example.metadata.get('subject')!r} but target has "
                    f"{question.metadata.get('subject')!r}"
                )

        description_block = (
            self.description.format(**self._display_metadata(question))
            if self.description is not None
            else None
        )

        blocks = []
        if description_block is not None:
            blocks.append(description_block)

        blocks.extend(self._block(example, include_answer=True) for example in examples)
        blocks.append(self._block(question, include_answer=False))
        text = BLOCK_SEPARATOR.join(blocks)
        return RenderedPrompt(text=text, continuations=self._continuations(question))

    def _block(self, question: Question, *, include_answer: bool) -> str:
        body = question.body.strip()
        if question.choices is not None:
            choices_lines = "\n".join(
                f"{LETTERS[i]}. {choice}" for i, choice in enumerate(question.choices)
            )
            body += "\n" + choices_lines
        answer_line = ANSWER_CUE + (f" {question.answer}" if include_answer else "")
        return body + "\n" + answer_line

    def _continuations(self, question: Question) -> tuple[str, ...] | None:
        if question.choices is not None:
            return tuple(CONTINUATION_PREFIX + LETTERS[i] for i in range(len(question.choices)))
        else:
            return None

    @staticmethod
    def _display_metadata(question: Question) -> dict[str, str]:
        return {
            key: value.replace("_", " ") if key == "subject" else value
            for key, value in question.metadata.items()
        }
