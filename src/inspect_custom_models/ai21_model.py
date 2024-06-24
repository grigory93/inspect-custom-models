from __future__ import annotations

import os
from typing import Any

from ai21 import AsyncAI21Client
from ai21.models.chat import ChatCompletionResponse as AI21ChatCompletionResponse
from ai21.models.chat import (
    ChatCompletionResponseChoice as AI21ChatCompletionResponseChoice,
)
from ai21.models.chat import ChatMessage as AI21ChatMessage
from ai21.types import NOT_GIVEN
from inspect_ai._util.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
)
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

AI21_API_KEY = "AI21_API_KEY"
AI21_MODELS = ["jamba-instruct-preview"]


@modelapi(name="ai21")
class AI21API(ModelAPI):
    def _init_(
        self,
        model_name: str,
        base_url: str | None = None,
        config: GenerateConfig = GenerateConfig(),  # noqa: B008
        **model_args: dict[str, Any],
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, config=config)
        self.model_args = model_args

        if model_name not in AI21_MODELS:
            message = f"Invalid AI21 model name: {model_name}"
            raise ValueError(message)

        api_key = os.environ.get(AI21_API_KEY, None)
        if api_key is None:
            message = f"{AI21_API_KEY} env variable not found"
            raise ValueError(message)

        self.api_key = api_key
        self.client = AsyncAI21Client(
            api_key=api_key,
            num_retries=(
                config.max_retries if config.max_retries else DEFAULT_MAX_RETRIES
            ),
            timeout_sec=config.timeout if config.timeout else DEFAULT_TIMEOUT,
        )

    async def generate(
        self,
        input: list[ChatMessage],  # noqa: A002
        tools: list[ToolInfo],  # noqa: ARG002
        tool_choice: ToolChoice,  # noqa: ARG002
        config: GenerateConfig,
    ) -> ModelOutput:
        messages = [
            ai21_chat_message(message)
            for message in input
            if message.role in ["user", "assistant"]
        ]
        system_message = next(
            (message.text for message in input if message.role == "system"), None,
        )

        # send request to AI21 API
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            system=system_message,
            temperature=config.temperature if config.temperature else NOT_GIVEN,
            top_p=config.top_p if config.top_p else NOT_GIVEN,
            max_tokens=config.max_tokens if config.max_tokens else DEFAULT_MAX_TOKENS,
            # random_seed=config.seed,  # noqa: ERA001
        )

        choices = completion_choices_from_response(response)
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            usage=ModelUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=(
                    response.usage.completion_tokens
                    if response.usage.completion_tokens
                    else response.usage.total_tokens - response.usage.prompt_tokens
                ),
                total_tokens=response.usage.total_tokens,
            ),
        )


def ai21_chat_message(message: ChatMessage) -> AI21ChatMessage:
    return AI21ChatMessage(
        role=message.role,
        content=message.text,
    )

def completion_choice(choice: AI21ChatCompletionResponseChoice) -> ChatCompletionChoice:
    message = choice.message
    completion = message.content
    if isinstance(completion, list):
        completion = " ".join(completion)
    return ChatCompletionChoice(
        message=ChatMessageAssistant(
            content=completion, source="generate",
        ),
        stop_reason=(
            choice_stop_reason(choice)
            if choice.finish_reason is not None
            else "unknown"
        ),
    )


def completion_choices_from_response(
    response: AI21ChatCompletionResponse,
) -> list[ChatCompletionChoice]:
    return [completion_choice(choice) for choice in response.choices]

def choice_stop_reason(choice: AI21ChatCompletionResponseChoice) -> StopReason:
    match choice.finish_reason:
        case "stop":
            return "stop"
        case "length":
            return "length"
        case "tool_calls":
            return "tool_calls"
        case _:
            return "unknown"
