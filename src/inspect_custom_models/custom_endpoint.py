from __future__ import annotations

from typing import Any

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    StopReason,
    ToolChoice,
    ToolInfo,
    modelapi,
)

ENDPOINT_MODELS = ["eai-stream"]

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
        message = endpoint_message(input[-1])

        response = await call_endpoint(message) # call endpoint here 
        choices = endpoint_choice_from_response(response)
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            usage=None,
        )


def endpoint_message(message: ChatMessage) -> str:
    return message.text


async def call_endpoint(message: str) -> str:
    return message


def endpoint_choice_from_response(answer: str) -> list[ChatCompletionChoice]:
    return [ChatCompletionChoice(
        message=ChatMessageAssistant(
            content=answer, source="generate",
        ),
    )]
