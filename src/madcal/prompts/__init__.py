"""Prompt protocols -- how a Question becomes the string a model sees.

Importing this package registers every protocol, so `prompt_registry` is fully
populated after `import madcal.prompts`. Each new implementation must be
imported here or its decorator never runs.
"""

from madcal.prompts.base import (
    CONTINUATION_PREFIX,
    PromptProtocol,
    RenderedPrompt,
    prompt_registry,
    register_prompt,
)
from madcal.prompts.fewshot import (
    ANSWER_CUE,
    BLOCK_SEPARATOR,
    FewShotCompletion,
)

__all__ = [
    "ANSWER_CUE",
    "BLOCK_SEPARATOR",
    "CONTINUATION_PREFIX",
    "FewShotCompletion",
    "PromptProtocol",
    "RenderedPrompt",
    "prompt_registry",
    "register_prompt",
]
