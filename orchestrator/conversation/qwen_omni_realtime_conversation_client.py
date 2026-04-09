import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union
from urllib.parse import quote

import websockets
from prometheus_client import Histogram

from ..utils.audio import resample_pcm
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .openai_audio_client import OpenAIAudioClient, _json_dumps_not_ensure_ascii


class QwenOmniRealtimeConversationClient(OpenAIAudioClient):
    """Alibaba Bailian / DashScope realtime omni conversation client."""

    INPUT_FRAME_RATE: int = 16000
    FRAME_RATE: int = 24000
    ExecutorRegistry.register_class("QwenOmniRealtimeConversationClient")

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        wss_url: str,
        qwen_model_name: str = "qwen3.5-omni-plus-realtime",
        input_transcription_model_name: str = "gummy-realtime-v1",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        latency_histogram: Histogram | None = None,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            agent_prompts_file=agent_prompts_file,
            wss_url=wss_url,
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
            token_number_histogram=token_number_histogram,
            logger_cfg=logger_cfg,
        )
        self.qwen_model_name = qwen_model_name
        self.input_transcription_model_name = input_transcription_model_name

    async def _create_session(
        self,
        request_id: str,
        cascade_memories: Union[None, Dict[str, Any]],
        voice_name: str = "Chelsie",
    ) -> None:
        conversation_model_override = self.input_buffer[request_id]["conversation_model_override"]
        qwen_model_name = conversation_model_override if conversation_model_override is not None else self.qwen_model_name
        wss_url = self.wss_url + "?model=" + quote(qwen_model_name)

        qwen_api_key = self.input_buffer[request_id]["api_keys"].get("qwen_api_key", "")
        if not qwen_api_key:
            msg = "Qwen API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        headers = {
            "Authorization": f"Bearer {qwen_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        ws = await websockets.connect(
            wss_url,
            proxy=self.proxy_url,
            additional_headers=headers,
            ping_interval=self.request_timeout,
            ping_timeout=self.request_timeout,
        )
        self.input_buffer[request_id]["ws_client"] = ws
        self.logger.info(f"Qwen Omni realtime client created for request {request_id}")
        memory_adapter = self.input_buffer[request_id]["memory_adapter"]
        conversation_history = await memory_adapter.build_chat_history(cascade_memories)
        loop = asyncio.get_event_loop()
        conversation_history = await loop.run_in_executor(
            self.executor,
            _json_dumps_not_ensure_ascii,
            {
                "CHAT_HISTORY": conversation_history,
            },
        )
        conversation_prompt = self.input_buffer[request_id]["user_prompt"]
        prompt = conversation_prompt + "\n" + conversation_history
        data = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "turn_detection": None,
                "instructions": prompt,
                "voice": voice_name or "Chelsie",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": self.input_transcription_model_name},
            },
        }
        data = await loop.run_in_executor(self.executor, json.dumps, data)
        await ws.send(data)
        await self._send_stream_start_task(request_id)
        asyncio.create_task(self._receive_pcm(request_id))

    async def _send_audio(self, request_id: str, audio_bytes: bytes, seq_number: int) -> None:
        loop = asyncio.get_event_loop()
        start_time = asyncio.get_running_loop().time()
        audio_type = self.input_buffer[request_id]["audio_type"]
        if audio_type == "pcm":
            pcm_bytes = audio_bytes
            frame_rate = self.input_buffer[request_id]["frame_rate"]
        elif audio_type == "wav":
            with io.BytesIO(audio_bytes) as wav_io:
                raise NotImplementedError("WAV format is not supported for Qwen Omni realtime conversation.")
        else:
            raise NotImplementedError(f"Unknown audio type: {audio_type} for request {request_id}")
        while "ws_client" not in self.input_buffer[request_id]:
            await asyncio.sleep(self.sleep_time)
            if asyncio.get_running_loop().time() - start_time > self.request_timeout:
                self.logger.error(
                    "Qwen Omni realtime client not connected after %s seconds for request %s",
                    self.request_timeout,
                    request_id,
                )
                return
        ws = self.input_buffer[request_id]["ws_client"]
        if frame_rate != self.__class__.INPUT_FRAME_RATE:
            pcm_bytes = await loop.run_in_executor(
                self.executor,
                resample_pcm,
                pcm_bytes,
                frame_rate,
                self.__class__.INPUT_FRAME_RATE,
            )
        audio_buffer = await loop.run_in_executor(self.executor, self._encode_pcm, pcm_bytes)
        message = await loop.run_in_executor(
            self.executor,
            json.dumps,
            {"type": "input_audio_buffer.append", "audio": audio_buffer},
        )
        while self.input_buffer[request_id]["input_chunk_sent"] < seq_number:
            await asyncio.sleep(self.sleep_time)
        await ws.send(message)
        self.input_buffer[request_id]["input_chunk_sent"] += 1
