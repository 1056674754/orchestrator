import json
from typing import Any, Dict, Optional, Union

import openai

from ..utils.exception import MissingAPIKeyException
from .openai_memory_client import OpenAIMemoryClient


class QwenMemoryClient(OpenAIMemoryClient):
    """Alibaba DashScope memory client using the OpenAI-compatible API."""

    def __init__(
        self,
        name: str,
        db_client,
        qwen_model_name: str = "qwen-turbo-latest",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        input_token_number_histogram=None,
        output_token_number_histogram=None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            db_client=db_client,
            openai_model_name=qwen_model_name,
            proxy_url=proxy_url,
            timeout=timeout,
            conversation_char_threshold=conversation_char_threshold,
            conversation_char_target=conversation_char_target,
            short_term_length_threshold=short_term_length_threshold,
            short_term_target_size=short_term_target_size,
            medium_term_length_threshold=medium_term_length_threshold,
            input_token_number_histogram=input_token_number_histogram,
            output_token_number_histogram=output_token_number_histogram,
            logger_cfg=logger_cfg,
        )
        self.qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _get_completion_extra_body(self) -> Dict[str, Any]:
        """Disable DashScope thinking mode for memory jobs."""
        return {"enable_thinking": False}

    async def call_llm(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        try:
            if not api_keys:
                raise ValueError("api_keys is required for Qwen LLM calls")
            qwen_api_key = api_keys.get("qwen_api_key", "")
            if not qwen_api_key:
                msg = "Qwen API key is not found in the API keys."
                self.logger.error(msg)
                raise MissingAPIKeyException(msg)

            qwen_client = openai.AsyncOpenAI(
                api_key=qwen_api_key,
                base_url=self.qwen_base_url,
                http_client=self.http_client,
                timeout=self.timeout,
            )

            qwen_model_name = model_override if model_override else self.openai_model_name
            response = await qwen_client.chat.completions.create(
                model=qwen_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=max_tokens,
                response_format=response_format,  # type: ignore
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned None content")
            if self.input_token_number_histogram:
                input_token_number = response.usage.prompt_tokens if response.usage else 0
                self.input_token_number_histogram.labels(adapter=self.name).observe(input_token_number)
            if self.output_token_number_histogram:
                output_token_number = response.usage.completion_tokens if response.usage else 0
                self.output_token_number_histogram.labels(adapter=self.name).observe(output_token_number)
            output = json.loads(content)["output"]
            return output
        except Exception as e:
            exception_type = type(e).__name__
            error_msg = f"Qwen LLM call failed: {exception_type}: {e}"
            if "response" in locals() and response is not None:
                try:
                    response_content = response.choices[0].message.content if response.choices else None
                    if response_content:
                        error_msg += f" | LLM response content: {response_content[:500]}"
                except Exception:
                    pass
            self.logger.error(error_msg)
            raise e
