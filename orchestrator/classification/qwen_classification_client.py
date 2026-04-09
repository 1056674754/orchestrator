import asyncio
import json
import re
from typing import Any, Dict, Optional, Union

import httpx
import openai
from prometheus_client import Histogram

from ..data_structures.classification import ClassificationType
from ..utils.exception import MissingAPIKeyException
from .classification_adapter import ClassificationAdapter


class QwenClassificationClient(ClassificationAdapter):
    """Classification client for Alibaba Cloud DashScope / Qwen."""

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        qwen_model_name: str = "qwen-turbo-latest",
        proxy_url: Union[None, str] = None,
        timeout: float = 3.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            motion_keywords=motion_keywords,
            proxy_url=proxy_url,
            latency_histogram=latency_histogram,
            logger_cfg=logger_cfg,
        )
        self.qwen_model_name = qwen_model_name
        self.qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.timeout = timeout

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url)
        else:
            self.http_client = None

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

    async def classify(
        self,
        request_id: str,
        prompt: str,
        text: str,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
    ) -> ClassificationType:
        llm_client = self.input_buffer[request_id].get("llm_client", None)
        while llm_client is None:
            await asyncio.sleep(self.sleep_time)
            llm_client = self.input_buffer[request_id].get("llm_client", None)

        model_name_override = self.input_buffer[request_id]["classification_model_override"]
        qwen_model_name = model_name_override if model_name_override else self.qwen_model_name
        system_content = prompt + "\n" + tag_prompt if tag_prompt else prompt

        try:
            response = await llm_client.chat.completions.create(
                model=qwen_model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"<user_input>: {text}"},
                ],
                temperature=0,
                max_tokens=64,
            )
            response_text = response.choices[0].message.content or ""  # type: ignore
            classification_result = self._parse_classification_result(response_text)
            self.logger.debug(f"Classification response: {classification_result}")
            return ClassificationType(classification_result)
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            raise e

    def _parse_classification_result(self, response_text: str) -> str:
        stripped_response = response_text.strip()

        try:
            parsed = json.loads(stripped_response)
            response_type = parsed.get("type", "")
            if response_type in {"accept", "reject", "leave"}:
                return response_type
        except json.JSONDecodeError:
            pass

        tag_match = re.search(r"<type>\s*(accept|reject|leave)\s*</type>", stripped_response, flags=re.IGNORECASE)
        if tag_match:
            return tag_match.group(1).lower()

        lowered_response = stripped_response.lower()
        if "reject" in lowered_response:
            return "reject"
        if "leave" in lowered_response:
            return "leave"
        return "accept"
