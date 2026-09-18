"""Chat-template rendering of debate messages."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from madcal.debate import Message, Renderer, Role
from madcal.registry import Registry

type RendererFactory = Callable[[str, str | None], Renderer]


class ChatTokenizer(Protocol):
    chat_template: str | None
    bos_token_id: int | None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str) -> Any: ...


class ChatTemplate:
    """Render messages with a tokenizer's chat template, ending at an open assistant turn."""

    def __init__(self, tokenizer: ChatTokenizer) -> None:
        if tokenizer.chat_template is None:
            raise ValueError("tokenizer has no chat template; it cannot render a conversation")
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str | None) -> "ChatTemplate":
        """Load the tokenizer for `model_id` at `revision` from the local cache only."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        return cls(tokenizer)

    def __call__(self, messages: Sequence[Message]) -> str:
        if not messages:
            raise ValueError("cannot render an empty conversation")
        if messages[-1].role is not Role.USER:
            raise ValueError("the last message must be from the user so the model has a turn")
        conversation = [
            {"role": str(message.role), "content": message.content} for message in messages
        ]
        text = self._tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(text, str):
            raise TypeError(f"chat template returned {type(text).__name__}, expected str")
        self._refuse_duplicate_bos(text)
        return text

    def _refuse_duplicate_bos(self, text: str) -> None:
        bos = self._tokenizer.bos_token_id
        if bos is None:
            return
        ids = list(self._tokenizer(text)["input_ids"])
        if ids[:2] == [bos, bos]:
            raise ValueError(
                "the rendered chat already starts with BOS and the tokenizer adds another; "
                "the model would see a duplicated BOS token"
            )


class PlainChat:
    """Render messages as labelled turns, for adapters with no chat template."""

    def __call__(self, messages: Sequence[Message]) -> str:
        if not messages:
            raise ValueError("cannot render an empty conversation")
        if messages[-1].role is not Role.USER:
            raise ValueError("the last message must be from the user so the model has a turn")
        turns = "\n\n".join(f"{message.role}: {message.content}" for message in messages)
        return f"{turns}\n\nassistant:"


def _chat_template(model_id: str, revision: str | None) -> Renderer:
    return ChatTemplate.from_pretrained(model_id, revision)


def _plain_chat(model_id: str, revision: str | None) -> Renderer:
    return PlainChat()


renderer_registry: Registry[RendererFactory] = Registry("renderer")
renderer_registry.register("chat_template")(_chat_template)
renderer_registry.register("plain_chat")(_plain_chat)
