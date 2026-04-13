import asyncio
import json
import re
import time
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
        timeout: float = 6.0,
        max_retries: int = 0,
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
        self.max_retries = max_retries

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url)
        else:
            self.http_client = None

    def _get_completion_extra_body(self) -> Dict[str, Any]:
        """Disable DashScope thinking mode for latency-sensitive classification."""
        return {"enable_thinking": False}

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
            max_retries=self.max_retries,
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
            request_start_time = time.time()
            response = await llm_client.chat.completions.create(
                model=qwen_model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"<user_input>: {text}"},
                ],
                temperature=0,
                max_tokens=64,
                extra_body=self._get_completion_extra_body(),
            )
            request_end_time = time.time()
            response_text = response.choices[0].message.content or ""  # type: ignore
            classification_result = self._parse_classification_result(response_text)
            self.logger.info(
                "Qwen classification response for request %s: model=%s, elapsed=%.3fs, response_chars=%d, result=%s",
                request_id,
                qwen_model_name,
                request_end_time - request_start_time,
                len(response_text),
                classification_result,
            )
            return ClassificationType(classification_result)
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            request_info = getattr(e, "request", None)
            response_info = getattr(e, "response", None)
            response_text = None
            if response_info is not None:
                response_text = getattr(response_info, "text", None)
                if callable(response_text):
                    try:
                        response_text = response_text()
                    except Exception:
                        response_text = None
            self.logger.error(
                "Classification error for request %s with model %s: type=%s, status=%s, request_url=%s, response_preview=%s, error=%s",
                request_id,
                qwen_model_name,
                type(e).__name__,
                status_code,
                getattr(request_info, "url", None),
                response_text[:300] if isinstance(response_text, str) and response_text else None,
                e,
            )
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
