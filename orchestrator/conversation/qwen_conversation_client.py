from typing import Any, Dict, Union

import openai

from ..utils.exception import MissingAPIKeyException
from .openai_conversation_client import OpenAIConversationClient


class QwenConversationClient(OpenAIConversationClient):
    """Alibaba DashScope conversation client using the OpenAI-compatible API."""

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        qwen_model_name: str = "qwen-turbo-latest",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor=None,
        latency_histogram=None,
        input_token_number_histogram=None,
        output_token_number_histogram=None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
        enable_bracket_filter: bool = True,
        bracket_pairs: list[tuple[str, str]] = [("*", "*"), ("(", ")"), ("[", "]"), ("{", "}"), ("「", "」"), ("（", "）")],
    ):
        super().__init__(
            name=name,
            agent_prompts_file=agent_prompts_file,
            openai_model_name=qwen_model_name,
            proxy_url=proxy_url,
            request_timeout=request_timeout,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            max_workers=max_workers,
            thread_pool_executor=thread_pool_executor,
            latency_histogram=latency_histogram,
            input_token_number_histogram=input_token_number_histogram,
            output_token_number_histogram=output_token_number_histogram,
            logger_cfg=logger_cfg,
            enable_bracket_filter=enable_bracket_filter,
            bracket_pairs=bracket_pairs,
        )
        self.qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    async def _init_llm_client(self, request_id: str) -> None:
        qwen_api_key = self.input_buffer[request_id]["chat_task"].get("api_keys", {}).get("qwen_api_key", "")
        if not qwen_api_key:
            msg = "Qwen API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        self.input_buffer[request_id]["chat_task"]["llm_client"] = openai.AsyncOpenAI(
            api_key=qwen_api_key,
            base_url=self.qwen_base_url,
            http_client=self.http_client,
            timeout=self.request_timeout,
        )
        self.input_buffer[request_id]["reject_task"]["llm_client"] = openai.AsyncOpenAI(
            api_key=qwen_api_key,
            base_url=self.qwen_base_url,
            http_client=self.http_client,
            timeout=self.request_timeout,
        )

    def _get_completion_extra_body(self) -> Dict[str, Any]:
        """Disable DashScope thinking mode for realtime conversation."""
        return {"enable_thinking": False}
