from __future__ import annotations

import os
from typing import Any
import logging
import websockets
import time
import json
import requests

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ToolChoice,
    ToolInfo,
    modelapi,
)

ENDPOINT_MODELS = ["eai-stream"]

@modelapi(name="eai-endpoint")
class EAIEndpoint(ModelAPI):
    def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            config: GenerateConfig = GenerateConfig(),  # noqa: B008
            **model_args: dict[str, Any],
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, config=config)
        self.model_args = model_args

        if model_name not in ENDPOINT_MODELS:
            raise ValueError(f"Invalid endpoint name: {model_name}")
        
        api_token = os.environ.get("VITE_API_TOKEN")
        if api_token is None:
            raise ValueError("VITE_API_TOKEN env variable not found")
        self.api_token = api_token

        rest_url = os.environ.get("VITE_API_REST_URL")
        if rest_url is None:
            raise ValueError("VITE_API_REST_URL env variable not found")
        self.rest_url = rest_url

        ws_url = os.environ.get("VITE_API_WS_URL")
        if ws_url is None:
            raise ValueError("VITE_API_WS_URL env variable not found")
        self.ws_url = ws_url

        gpt_version = os.environ.get("VITE_API_GPT")
        if gpt_version is None:
            raise ValueError("VITE_API_GPT env variable not found")
        self.gpt_version = gpt_version 

        self.client = EAIClient(api_token, rest_url, ws_url, gpt_version)
        
    
    async def generate(
            self,
            input: list[ChatMessage],  # noqa: A002
            tools: list[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
    ) -> ModelOutput:
        message = endpoint_message(input[-1])
        logging.debug(f"Message: {message}")
        conversation_id = await self.client.new_conversation_id()
        response = await self.client.stream(message, conversation_id)
        choices = endpoint_choice_from_response(response)
        return ModelOutput(
            model=self.model_name,
            choices=choices,
            usage=None,
        )

def endpoint_message(message: ChatMessage) -> str:
    return message.text

def endpoint_choice_from_response(answer: str) -> list[ChatCompletionChoice]:
    return [ChatCompletionChoice(
        message=ChatMessageAssistant(
            content=answer, source="generate",
        ),
    )]

class EAIClient:
    def __init__(self, api_token: str, rest_url: str, ws_url: str, gpt_version: str):
        self.api_token = api_token
        self.rest_url = rest_url
        self.ws_url = ws_url
        self.gpt_version = gpt_version

    async def stream(self, question, conversation_id):
        start_time = time.time()
        response = ""
        try:
            async with websockets.connect(f"{self.ws_url}/ws?token={self.api_token}") as websocket:
                request = json.dumps({  
                    "conversation_id": conversation_id,
                    "bot": self.gpt_version,
                    "user_content": question,
                    "command": "STREAMING",
                })
                await websocket.send(request)
                while True:
                    r = await websocket.recv()
                    d = json.loads(r)
                    status = d["status"]
                    if status == "done":
                        break
                    content = d["content"]
                    response += content
                return response
        except websockets.ConnectionClosedError as e:
            logging.error(f"Connection closed with error: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            await websocket.close()
            logging.debug(f"Total duration was {time.time() - start_time:.2f} seconds")
            logging.debug(f"Response: {response}")


    async def new_conversation_id(self):
        response = requests.post(f"{self.rest_url}/conversations/new", headers={"Authorization": self.api_token})
        conversation_id = response.json()["conversation_id"]
        logging.debug(f"New conversation_id = {conversation_id}")
        return conversation_id
