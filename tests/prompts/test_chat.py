from typing import Any

import pytest

from madcal.debate import Message, Role
from madcal.prompts import ChatTemplate

INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class FakeTokenizer:
    chat_template: str | None = "fake"
    bos_token_id: int | None = None

    def __init__(self, prefix: str = "", ids: list[int] | None = None) -> None:
        self.prefix = prefix
        self.ids = ids or [7, 8]
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        self.calls.append((conversation, kwargs))
        body = "".join(f"<{turn['role']}>{turn['content']}" for turn in conversation)
        return f"{self.prefix}{body}<assistant>"

    def __call__(self, text: str) -> Any:
        return {"input_ids": self.ids}


CONVERSATION = [
    Message(Role.USER, "Question?"),
    Message(Role.ASSISTANT, "Answer: A"),
    Message(Role.USER, "Update?"),
]


def test_messages_are_passed_in_order_with_an_open_assistant_turn():
    tokenizer = FakeTokenizer()
    text = ChatTemplate(tokenizer)(CONVERSATION)
    assert text == "<user>Question?<assistant>Answer: A<user>Update?<assistant>"
    conversation, kwargs = tokenizer.calls[0]
    assert conversation == [
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": "Answer: A"},
        {"role": "user", "content": "Update?"},
    ]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}


def test_tokenizer_without_a_chat_template_is_refused():
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="no chat template"):
        ChatTemplate(tokenizer)


def test_empty_conversation_is_refused():
    with pytest.raises(ValueError, match="empty"):
        ChatTemplate(FakeTokenizer())([])


def test_conversation_ending_on_the_assistant_is_refused():
    with pytest.raises(ValueError, match="last message"):
        ChatTemplate(FakeTokenizer())(CONVERSATION[:2])


def test_duplicated_bos_is_refused():
    tokenizer = FakeTokenizer(prefix="<s>", ids=[1, 1, 7])
    tokenizer.bos_token_id = 1
    with pytest.raises(ValueError, match="duplicated BOS"):
        ChatTemplate(tokenizer)(CONVERSATION)


def test_single_bos_is_accepted():
    tokenizer = FakeTokenizer(prefix="<s>", ids=[1, 7])
    tokenizer.bos_token_id = 1
    assert ChatTemplate(tokenizer)(CONVERSATION).startswith("<s>")


def _instruct_model_is_cached() -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(INSTRUCT_MODEL, local_files_only=True)
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not _instruct_model_is_cached(),
    reason=f"{INSTRUCT_MODEL} is not in the local HF cache",
)
def test_real_chat_template_ends_at_the_assistant_turn():
    text = ChatTemplate.from_pretrained(INSTRUCT_MODEL, None)(CONVERSATION)
    assert text.endswith("<|im_start|>assistant\n")
    assert text.count("<|im_start|>user") == 2
