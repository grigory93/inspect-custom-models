from __future__ import annotations

from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ToolChoice,
    ToolInfo,
    modelapi,
)


@modelapi(name="endpoint")
class Endpoint(ModelAPI):
    def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            config: GenerateConfig = GenerateConfig(),  # noqa: B008
            **model_args: dict[str, Any],
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, config=config)
        self.model_args = model_args


    async def generate(
            self,
            input: list[ChatMessage],  # noqa: A002
            tools: list[ToolInfo], 
            tool_choice: ToolChoice,
            config: GenerateConfig,
    ) -> ModelOutput:
        message = endpoint_messagee(input[-1])

        return None


def endpoint_message(message: ChatMessage) -> str:
    return message.text
