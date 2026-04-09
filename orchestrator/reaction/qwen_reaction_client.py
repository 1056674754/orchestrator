from typing import Any, Dict, Union

import openai

from ..utils.exception import MissingAPIKeyException
from .openai_reaction_client import OpenAIReactionClient


class QwenReactionClient(OpenAIReactionClient):
    """Alibaba DashScope reaction client using the OpenAI-compatible API."""

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        qwen_model_name: str = "qwen-turbo-latest",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        latency_histogram=None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            motion_keywords=motion_keywords,
            openai_model_name=qwen_model_name,
            proxy_url=proxy_url,
            timeout=timeout,
            latency_histogram=latency_histogram,
            logger_cfg=logger_cfg,
        )
        self.qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    async def _init_llm_client(self, request_id: str) -> None:
        qwen_api_key = self.input_buffer[request_id]["api_keys"].get("qwen_api_key", "")
        if not qwen_api_key:
            msg = "Qwen API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        self.input_buffer[request_id]["llm_client"] = openai.AsyncOpenAI(
            api_key=qwen_api_key,
            base_url=self.qwen_base_url,
            http_client=self.http_client,
            timeout=self.timeout,
        )
