"""Prompt protocols -- how a Question becomes the string a model sees.

Importing this package registers every protocol, so `prompt_registry` is fully
populated after `import caldrift.prompts`. Each new implementation must be
imported here or its decorator never runs.
"""

from caldrift.prompts.base import (
    CONTINUATION_PREFIX,
    PromptProtocol,
    RenderedPrompt,
    prompt_registry,
    register_prompt,
)

__all__ = [
    "CONTINUATION_PREFIX",
    "PromptProtocol",
    "RenderedPrompt",
    "prompt_registry",
    "register_prompt",
]
