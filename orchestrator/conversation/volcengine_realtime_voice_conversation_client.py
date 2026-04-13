import asyncio
import gzip
import io
import json
import os
import random
import re
import struct
import time
import traceback
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import websockets
from prometheus_client import Histogram

from ..data_structures.process_flow import DAGStatus
from ..utils.audio import resample_pcm
from ..utils.exception import MissingAPIKeyException, failure_callback
from ..utils.executor_registry import ExecutorRegistry
from .audio_conversation_adapter import AudioConversationAdapter
from .openai_audio_client import OpenAIAudioClient


class VolcengineRealtimeVoiceConversationClient(OpenAIAudioClient):
    """Volcengine realtime dialogue client.

    Official docs:
    - Realtime dialogue protocol:
      https://www.volcengine.com/docs/6561/1594356?lang=zh
    - Realtime dialogue product overview:
      https://www.volcengine.com/docs/6561/1594348?lang=zh
    """

    AVAILABLE_FOR_STREAM = True
    INPUT_FRAME_RATE: int = 16000
    FRAME_RATE: int = 24000
    N_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2

    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_ERROR_RESPONSE = 0b1111

    JSON_SERIALIZATION = 0b0001
    NO_SERIALIZATION = 0b0000
    GZIP_COMPRESSION = 0b0001
    MSG_WITH_EVENT = 0b0100

    EVENT_START_CONNECTION = 1
    EVENT_STOP_CONNECTION = 2
    EVENT_START_SESSION = 100
    EVENT_STOP_SESSION = 102
    EVENT_CLIENT_AUDIO = 200

    EVENT_CONNECTION_STARTED = 50
    EVENT_CONNECTION_FINISHED = 52
    EVENT_SESSION_STARTED = 150
    EVENT_SESSION_FINISHED = 152
    EVENT_SESSION_FAILED = 153
    EVENT_USAGE = 154
    EVENT_TTS_SENTENCE_START = 350
    EVENT_TTS_SENTENCE_END = 351
    EVENT_TTS_AUDIO = 352
    EVENT_TTS_ENDED = 359
    EVENT_ASR_INFO = 450
    EVENT_ASR_RESULT = 451
    EVENT_ASR_ENDED = 459
    EVENT_CHAT_RESULT = 550
    EVENT_CHAT_ENDED = 559

    VOLCENGINE_REALTIME_APP_KEY = "PlgvMymc7f3tQnJ6"
    DEFAULT_RESOURCE_ID = "volc.speech.dialog"
    TRAILING_SILENCE_DURATION = 1.0
    AUDIO_CHUNK_N_BYTES = 1600
    MIN_UPSTREAM_AUDIO_SECONDS = 0.5
    ASR_RESULT_TIMEOUT = 8.0
    ASR_FALLBACK_TIMEOUT = 2.0
    ASR_FALLBACK_MAX_UPSTREAM_SECONDS = 1.0
    AUDIO_RETENTION_DIR_ENV = "VOLCENGINE_AUDIO_RETENTION_DIR"
    VENDOR_CLOSE_TIMEOUT = 1.0

    _FALLBACK_TEXTS = [
        "嗯……我没太听清，能再说一遍吗？",
        "抱歉，刚刚没听清楚呢。",
        "你刚才说什么？风太大了我没听见。",
        "嗯？能再重复一下吗？",
        "哎呀，走神了，再说一次好不好？",
        "等一下，我没听明白，再说一次？",
        "不好意思，刚才没听清呢。",
        "嗯，能再说一次吗？我刚才没听清。",
        "哈？你说啥？我刚才在发呆。",
        "抱歉抱歉，再说一遍呗？",
        "嗯……什么？再说一次好不好？",
        "刚刚走神了，再讲一遍嘛。",
        "我没听清哎，再说一次？",
        "啊？能再说一遍吗？",
        "嗯，信号好像不太好，刚才说什么了？",
        "不好意思，刚没注意听，再来一次？",
        "哎呀，我刚才没听清楚，再说一遍吧。",
        "嗯？你说了什么？我走神了。",
        "抱歉，再说一次好不好？我没听清。",
        "等等，我好像漏掉了，再说一遍？",
    ]

    ExecutorRegistry.register_class("VolcengineRealtimeVoiceConversationClient")

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        wss_url: str,
        volcengine_bot_name: str = "",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        trailing_silence_duration: float = TRAILING_SILENCE_DURATION,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        latency_histogram: Histogram | None = None,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        AudioConversationAdapter.__init__(
            self,
            name=name,
            agent_prompts_file=agent_prompts_file,
            proxy_url=proxy_url,
            request_timeout=request_timeout,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            latency_histogram=latency_histogram,
            input_token_number_histogram=input_token_number_histogram,
            output_token_number_histogram=output_token_number_histogram,
            logger_cfg=logger_cfg,
        )
        self.wss_url = wss_url
        self.volcengine_bot_name = volcengine_bot_name
        self.trailing_silence_duration = trailing_silence_duration
        self.executor = (
            thread_pool_executor
            if thread_pool_executor is not None
            else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = thread_pool_executor is not None
        audio_retention_dir = os.environ.get(self.__class__.AUDIO_RETENTION_DIR_ENV, "")
        self.audio_retention_dir = audio_retention_dir.strip() or None
        _ = token_number_histogram

    def __del__(self) -> None:
        if not getattr(self, "executor_external", True):
            self.executor.shutdown(wait=True)

    DIALOG_CONTEXT_MAX_TURNS = 20

    async def _create_session(
        self,
        request_id: str,
        cascade_memories: Union[None, Dict[str, Any]],
        voice_name: str = "",
    ) -> None:
        _ = voice_name
        request_state = self.input_buffer[request_id]
        api_keys = request_state["api_keys"]
        app_id = api_keys.get("volcengine_app_id", "") or ""
        access_token = api_keys.get("volcengine_token", "") or ""
        if not app_id or not access_token:
            msg = "Volcengine realtime dialogue requires volcengine_app_id and volcengine_token."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        connect_id = str(uuid.uuid4())
        headers = {
            "X-Api-App-ID": app_id,
            "X-Api-Access-Key": access_token,
            "X-Api-Resource-Id": self.__class__.DEFAULT_RESOURCE_ID,
            "X-Api-App-Key": self.__class__.VOLCENGINE_REALTIME_APP_KEY,
            "X-Api-Connect-Id": connect_id,
        }

        connect_start_time = time.time()
        ws = await websockets.connect(
            self.wss_url,
            proxy=self.proxy_url,
            additional_headers=headers,
            ping_interval=5,
            max_size=100 * 1024 * 1024,
        )
        request_state["ws_client"] = ws
        request_state["volcengine_session_id"] = uuid.uuid4().hex
        request_state["user_input"] = ""
        request_state["assistant_output"] = ""
        request_state["response_done"] = False
        request_state["first_audio_received"] = False
        request_state["closed"] = False
        request_state["asr_final_received"] = False
        request_state["audio_end_sent_time"] = None
        request_state["short_fallback_task"] = None
        request_state["asr_timeout_task"] = None
        request_state["completion_reason"] = None
        request_state["audio_retention_saved"] = False
        if self.audio_retention_dir is not None:
            request_state["retained_upstream_audio"] = bytearray()
        request_state["connection_logid"] = self._safe_response_header(ws, "X-Tt-Logid")

        await ws.send(
            self._encode_connection_event(self.__class__.EVENT_START_CONNECTION, {})
        )
        connection_started = await self._recv_until_event(
            request_id,
            expected_events={self.__class__.EVENT_CONNECTION_STARTED},
            stage="start_connection",
        )
        self.logger.info(
            "Volcengine realtime connection ready for request %s: connect_id=%s, logid=%s, connect_elapsed=%.3fs",
            request_id,
            connect_id,
            request_state["connection_logid"],
            time.time() - connect_start_time,
        )

        bot_name = self._resolve_bot_name(request_state)
        dialog_config: Dict[str, Any] = {
            "extra": {"input_mod": "keep_alive"},
        }
        if bot_name:
            dialog_config["bot_name"] = bot_name
            self.logger.info(
                "Volcengine realtime bot_name selected for request %s: %s",
                request_id,
                bot_name,
            )

        system_role = await self._build_system_role(request_state, cascade_memories)
        if system_role:
            dialog_config["system_role"] = system_role

        dialog_context = self._build_dialog_context(request_state, cascade_memories)
        if dialog_context:
            dialog_config["dialog_context"] = dialog_context

        session_payload: Dict[str, Any] = {
            "tts": {
                "audio_config": {
                    "format": "pcm",
                    "sample_rate": self.__class__.FRAME_RATE,
                    "channels": self.__class__.N_CHANNELS,
                    "bit_size": "f32",
                }
            },
            "dialog": dialog_config,
        }

        await ws.send(
            self._encode_session_event(
                self.__class__.EVENT_START_SESSION,
                request_state["volcengine_session_id"],
                session_payload,
            )
        )
        session_started = await self._recv_until_event(
            request_id,
            expected_events={self.__class__.EVENT_SESSION_STARTED},
            stage="start_session",
        )
        dialog_id = ""
        if isinstance(session_started.get("payload_msg"), dict):
            dialog_id = session_started["payload_msg"].get("dialog_id", "")
        request_state["dialog_id"] = dialog_id
        request_state["last_update_time"] = time.time()
        await self._send_stream_start_task(request_id)
        asyncio.create_task(self._receive_pcm(request_id))

    async def _send_audio(
        self, request_id: str, audio_bytes: bytes, seq_number: int
    ) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            self.logger.warning(
                "Request %s not found before sending realtime audio", request_id
            )
            return
        start_time = time.time()
        loop = asyncio.get_event_loop()
        audio_type = request_state["audio_type"]
        if audio_type != "pcm":
            msg = f"Unknown audio type: {audio_type} for request {request_id}"
            self.logger.error(msg)
            request_state["dag"].set_status(DAGStatus.FAILED)
            return
        pcm_bytes = audio_bytes
        frame_rate = request_state["frame_rate"]
        while "ws_client" not in request_state:
            await asyncio.sleep(self.sleep_time)
            request_state = self.input_buffer.get(request_id)
            if request_state is None:
                return
            if time.time() - start_time > self.request_timeout:
                self.logger.error(
                    "Volcengine realtime client not connected after %.3fs for request %s",
                    self.request_timeout,
                    request_id,
                )
                return
        ws = request_state["ws_client"]
        if frame_rate != self.__class__.INPUT_FRAME_RATE:
            pcm_bytes = await loop.run_in_executor(
                self.executor,
                resample_pcm,
                pcm_bytes,
                frame_rate,
                self.__class__.INPUT_FRAME_RATE,
            )
        retained_audio = request_state.get("retained_upstream_audio")
        if retained_audio is not None:
            retained_audio.extend(pcm_bytes)
        while request_state["input_chunk_sent"] < seq_number:
            await asyncio.sleep(self.sleep_time)
            request_state = self.input_buffer.get(request_id)
            if request_state is None or request_state.get("closed", False):
                return
        await ws.send(
            self._encode_audio_event(
                self.__class__.EVENT_CLIENT_AUDIO,
                request_state["volcengine_session_id"],
                pcm_bytes,
            )
        )
        request_state["input_chunk_sent"] += 1

    async def _commit_audio(self, request_id: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            return
        request_state["commit_time"] = time.time()
        dag = request_state["dag"]
        while "ws_client" not in request_state:
            await asyncio.sleep(self.sleep_time)
            request_state = self.input_buffer.get(request_id)
            if request_state is None:
                return
            if request_state["dag"].status != DAGStatus.RUNNING:
                return
        while request_state["input_chunk_sent"] < request_state["input_chunk_received"]:
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
                request_state = self.input_buffer.get(request_id)
                if request_state is None:
                    return
            else:
                self.logger.warning(
                    "Volcengine realtime commit interrupted by DAG status %s for request %s",
                    dag.status,
                    request_id,
                )
                return

        silence_frames = max(
            1,
            int(
                self.trailing_silence_duration
                * self.__class__.INPUT_FRAME_RATE
                * self.__class__.SAMPLE_WIDTH
                * self.__class__.N_CHANNELS
                / self.__class__.AUDIO_CHUNK_N_BYTES
            ),
        )
        silence_chunk = bytes(self.__class__.AUDIO_CHUNK_N_BYTES)
        ws = request_state["ws_client"]
        session_id = request_state["volcengine_session_id"]
        sample_width = request_state.get("sample_width", self.__class__.SAMPLE_WIDTH)
        n_channels = request_state.get("n_channels", self.__class__.N_CHANNELS)
        frame_rate = request_state.get("frame_rate", self.__class__.INPUT_FRAME_RATE)
        upstream_bytes = request_state.get("input_audio_bytes_received", 0)
        upstream_seconds = 0.0
        denominator = sample_width * n_channels * frame_rate
        if denominator > 0:
            upstream_seconds = upstream_bytes / denominator
        for _ in range(silence_frames):
            await ws.send(
                self._encode_audio_event(
                    self.__class__.EVENT_CLIENT_AUDIO, session_id, silence_chunk
                )
            )
            await asyncio.sleep(
                self.__class__.AUDIO_CHUNK_N_BYTES
                / (
                    self.__class__.INPUT_FRAME_RATE
                    * self.__class__.SAMPLE_WIDTH
                    * self.__class__.N_CHANNELS
                )
            )
        request_state["audio_end_sent_time"] = time.time()
        dag_start_time = request_state.get("dag_start_time")
        since_dag_start = (
            request_state["audio_end_sent_time"] - dag_start_time
            if dag_start_time is not None
            else None
        )
        self.logger.info(
            "Volcengine realtime audio end sent for request %s: upstream_chunks=%s, upstream_bytes=%s, upstream_audio=%.3fs, vendor_chunks_sent=%s%s",
            request_id,
            request_state.get("input_chunk_received", 0),
            upstream_bytes,
            upstream_seconds,
            request_state.get("input_chunk_sent", 0),
            f", since_dag_start={since_dag_start:.3f}s"
            if since_dag_start is not None
            else "",
        )
        self._schedule_asr_watchdogs(request_id)

    async def _receive_pcm(self, request_id: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            return
        ws = request_state["ws_client"]
        dag_start_time = request_state["dag_start_time"]
        input_token_number = 0
        output_token_number = 0
        while True:
            try:
                message = await asyncio.wait_for(
                    ws.recv(), timeout=self.request_timeout
                )
                cur_time = time.time()
                request_state = self.input_buffer.get(request_id)
                if request_state is None:
                    return
                request_state["last_update_time"] = cur_time
                result = self._parse_response(message)
                if result.get("message_type") == self.__class__.SERVER_ERROR_RESPONSE:
                    vendor_reason = self._format_server_error_reason(result)
                    self.logger.error(
                        "Volcengine realtime server error for request %s: code=%s body=%s",
                        request_id,
                        result.get("code"),
                        result.get("payload_msg"),
                    )
                    await self._fail_request(request_id, vendor_reason)
                    return

                event = result.get("event")
                payload_msg = result.get("payload_msg")
                if event == self.__class__.EVENT_ASR_INFO:
                    self.logger.debug(
                        "Volcengine realtime ASR info for request %s: %s",
                        request_id,
                        payload_msg,
                    )
                elif event == self.__class__.EVENT_ASR_RESULT:
                    text, is_final = self._extract_asr_text(payload_msg)
                    if text:
                        request_state["user_input"] = text
                    if is_final:
                        request_state["asr_final_received"] = True
                        self._cancel_asr_watchdogs(request_state)
                        commit_time = self._get_audio_end_reference_time(request_state)
                        if commit_time is not None:
                            since_commit = cur_time - commit_time
                            since_dag_start = (
                                cur_time - dag_start_time
                                if dag_start_time is not None
                                else None
                            )
                            self.logger.info(
                                "Volcengine realtime ASR final text ready for request %s: text_chars=%d, commit_to_final=%.3fs%s",
                                request_id,
                                len(text),
                                since_commit,
                                f", since_dag_start={since_dag_start:.3f}s"
                                if since_dag_start is not None
                                else "",
                            )
                elif event == self.__class__.EVENT_ASR_ENDED:
                    self.logger.debug(
                        "Volcengine realtime ASR ended for request %s: %s",
                        request_id,
                        payload_msg,
                    )
                elif event == self.__class__.EVENT_CHAT_RESULT:
                    content = ""
                    if isinstance(payload_msg, dict):
                        content = payload_msg.get("content", "") or ""
                    request_state["assistant_output"] += content
                elif event == self.__class__.EVENT_TTS_AUDIO:
                    seq_number = request_state["chunk_received"]
                    request_state["chunk_received"] += 1
                    pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._float32_pcm_to_int16,
                        payload_msg,
                    )
                    ret_dict = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.convert_tts_time,
                        pcm_bytes,
                    )
                    audio_io = io.BytesIO(pcm_bytes)
                    audio_io.seek(0)
                    ret_dict["audio"] = audio_io
                    if not request_state["first_audio_received"]:
                        request_state["first_audio_received"] = True
                        commit_time = self._get_audio_end_reference_time(request_state)
                        if commit_time is not None:
                            latency = cur_time - commit_time
                            msg = (
                                f"request {request_id} Volcengine realtime first audio latency from audio end: "
                                f"{latency:.2f} seconds"
                            )
                            if dag_start_time is not None:
                                msg += f", from dag start: {cur_time - dag_start_time:.2f} seconds"
                            self.logger.debug(msg)
                            if self.latency_histogram:
                                user_id = request_state["user_id"]
                                self.latency_histogram.labels(
                                    adapter=self.name, user_id=user_id
                                ).observe(latency)
                    await self._send_audio_to_downstream(
                        request_id, ret_dict, seq_number
                    )
                elif event == self.__class__.EVENT_TTS_SENTENCE_END:
                    self.logger.debug(
                        "Volcengine realtime TTS sentence end for request %s",
                        request_id,
                    )
                elif event == self.__class__.EVENT_USAGE:
                    usage = (
                        payload_msg.get("usage", {})
                        if isinstance(payload_msg, dict)
                        else {}
                    )
                    input_token_number = usage.get("input_text_tokens", 0) + usage.get(
                        "input_audio_tokens", 0
                    )
                    output_token_number = usage.get(
                        "output_text_tokens", 0
                    ) + usage.get("output_audio_tokens", 0)
                elif event == self.__class__.EVENT_CHAT_ENDED:
                    request_state["response_done"] = True
                    request_state["completion_reason"] = "chat_ended"
                    await self._append_history(request_id)
                    if self.input_token_number_histogram:
                        self.input_token_number_histogram.labels(
                            adapter=self.name,
                            user_id=request_state["user_id"],
                        ).observe(input_token_number)
                    if self.output_token_number_histogram:
                        self.output_token_number_histogram.labels(
                            adapter=self.name,
                            user_id=request_state["user_id"],
                        ).observe(output_token_number)
                    await self._close_session(request_id)
                    return
                elif event in {
                    self.__class__.EVENT_CONNECTION_STARTED,
                    self.__class__.EVENT_CONNECTION_FINISHED,
                    self.__class__.EVENT_SESSION_STARTED,
                    self.__class__.EVENT_SESSION_FINISHED,
                    self.__class__.EVENT_SESSION_FAILED,
                    self.__class__.EVENT_TTS_SENTENCE_START,
                    self.__class__.EVENT_TTS_ENDED,
                }:
                    self.logger.debug(
                        "Volcengine realtime event %s for request %s: %s",
                        event,
                        request_id,
                        payload_msg,
                    )
                else:
                    self.logger.debug(
                        "Volcengine realtime unknown event %s for request %s: %s",
                        event,
                        request_id,
                        payload_msg,
                    )
            except websockets.exceptions.ConnectionClosed as exc:
                if not request_state.get("closed", False):
                    self.logger.warning(
                        "Volcengine realtime websocket closed for request %s: code=%s reason=%s",
                        request_id,
                        exc.code,
                        exc.reason,
                    )
                    if not request_state.get("response_done", False):
                        await self._fail_request(
                            request_id,
                            f"实时语音连接中断: {exc.reason or exc.code}",
                        )
                    else:
                        await self._close_session(request_id)
                return
            except asyncio.TimeoutError:
                request_state = self.input_buffer.get(request_id)
                if request_state is not None and not request_state.get("closed", False):
                    self.logger.warning(
                        "Volcengine realtime websocket recv timeout for request %s after %.1fs without vendor frames",
                        request_id,
                        self.request_timeout,
                    )
                    if not request_state.get("response_done", False):
                        request_state["completion_reason"] = "vendor_recv_timeout"
                        await self._fail_request(
                            request_id,
                            "实时语音识别连接超时，服务端未能继续返回结果，请重试。",
                        )
                    else:
                        # Response was already sent (e.g. ASR fallback) but the
                        # session was never properly closed — clean up now so the
                        # DAG can complete instead of hanging until timeout.
                        await self._close_session(request_id)
                return
            except Exception as exc:
                msg = f"Error in Volcengine realtime audio conversation: {exc} for request {request_id}"
                msg += f"\n{traceback.format_exc()}"
                self.logger.error(msg)
                await self._fail_request(
                    request_id, f"volcengine realtime receive error: {exc}"
                )
                return

    async def _close_session(self, request_id: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            return
        if request_state.get("closed", False):
            return
        self._cancel_asr_watchdogs(request_state)
        self._persist_retained_upstream_audio(request_id, request_state)
        request_state["closed"] = True
        ws = request_state.get("ws_client")
        session_id = request_state.get("volcengine_session_id", "")
        if ws is not None:
            try:
                await asyncio.wait_for(
                    ws.send(
                        self._encode_session_event(
                            self.__class__.EVENT_STOP_SESSION, session_id, {}
                        )
                    ),
                    timeout=self.__class__.VENDOR_CLOSE_TIMEOUT,
                )
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    ws.send(
                        self._encode_connection_event(
                            self.__class__.EVENT_STOP_CONNECTION, {}
                        )
                    ),
                    timeout=self.__class__.VENDOR_CLOSE_TIMEOUT,
                )
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    ws.close(), timeout=self.__class__.VENDOR_CLOSE_TIMEOUT
                )
            except Exception:
                pass
        await self._send_stream_end_task(request_id)

    async def _recv_until_event(
        self,
        request_id: str,
        expected_events: set[int],
        stage: str,
    ) -> dict[str, Any]:
        ws = self.input_buffer[request_id]["ws_client"]
        start_time = time.time()
        while True:
            remaining = self.request_timeout - (time.time() - start_time)
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {stage} for request {request_id}"
                )
            raw_message = await asyncio.wait_for(ws.recv(), timeout=remaining)
            result = self._parse_response(raw_message)
            if result.get("message_type") == self.__class__.SERVER_ERROR_RESPONSE:
                raise RuntimeError(
                    f"Volcengine realtime {stage} failed for request {request_id}: "
                    f"code={result.get('code')} body={result.get('payload_msg')}"
                )
            event = result.get("event")
            if event in expected_events:
                return result
            self.logger.debug(
                "Volcengine realtime unexpected event while waiting for %s, request %s: event=%s payload=%s",
                stage,
                request_id,
                event,
                result.get("payload_msg"),
            )

    async def _append_history(self, request_id: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            return
        memory_db_client = request_state["memory_db_client"]
        if memory_db_client is None:
            return
        user_input = request_state.get("user_input", "")
        assistant_output = request_state.get("assistant_output", "")
        if not user_input or not assistant_output:
            return
        cur_time = time.time()
        timezone = request_state.get("timezone")
        await memory_db_client.append_chat_history(
            character_id=request_state["character_id"],
            unix_timestamp=cur_time,
            role="user",
            content=user_input,
            relationship=request_state["relationship"],
            timezone=timezone,
        )
        await memory_db_client.append_chat_history(
            character_id=request_state["character_id"],
            unix_timestamp=cur_time,
            role="assistant",
            content=assistant_output,
            timezone=timezone,
            **request_state["emotion"],
        )

    def _calc_upstream_seconds(self, request_state: dict[str, Any]) -> float:
        upstream_bytes = request_state.get("input_audio_bytes_received", 0)
        sample_width = request_state.get("sample_width", self.__class__.SAMPLE_WIDTH)
        n_channels = request_state.get("n_channels", self.__class__.N_CHANNELS)
        frame_rate = request_state.get("frame_rate", self.__class__.INPUT_FRAME_RATE)
        denominator = sample_width * n_channels * frame_rate
        if denominator <= 0:
            return 0.0
        return upstream_bytes / denominator

    def _get_audio_end_reference_time(
        self, request_state: dict[str, Any]
    ) -> float | None:
        return request_state.get("audio_end_sent_time") or request_state.get(
            "commit_time"
        )

    def _cancel_asr_watchdogs(self, request_state: dict[str, Any]) -> None:
        for key in ("short_fallback_task", "asr_timeout_task"):
            task = request_state.get(key)
            if task is not None and not task.done():
                task.cancel()
            request_state[key] = None

    def _schedule_asr_watchdogs(self, request_id: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None or request_state.get("closed", False):
            return
        reference_time = request_state.get("audio_end_sent_time")
        if reference_time is None:
            return
        self._cancel_asr_watchdogs(request_state)
        request_state["short_fallback_task"] = asyncio.create_task(
            self._run_short_fallback_watchdog(request_id, reference_time)
        )
        request_state["asr_timeout_task"] = asyncio.create_task(
            self._run_asr_timeout_watchdog(request_id, reference_time)
        )

    def _watchdog_can_fire(
        self, request_state: dict[str, Any] | None, reference_time: float
    ) -> bool:
        if request_state is None:
            return False
        if request_state.get("closed", False) or request_state.get(
            "response_done", False
        ):
            return False
        if request_state.get("asr_final_received", False):
            return False
        current_reference = request_state.get("audio_end_sent_time")
        if current_reference is None:
            return False
        return abs(current_reference - reference_time) < 1e-6

    async def _run_short_fallback_watchdog(
        self, request_id: str, reference_time: float
    ) -> None:
        try:
            await asyncio.sleep(self.__class__.ASR_FALLBACK_TIMEOUT)
            request_state = self.input_buffer.get(request_id)
            if not self._watchdog_can_fire(request_state, reference_time):
                return
            if (
                self._calc_upstream_seconds(request_state)
                <= self.__class__.ASR_FALLBACK_MAX_UPSTREAM_SECONDS
            ):
                await self._send_asr_fallback(request_id)
        except asyncio.CancelledError:
            return

    async def _run_asr_timeout_watchdog(
        self, request_id: str, reference_time: float
    ) -> None:
        try:
            await asyncio.sleep(self.__class__.ASR_RESULT_TIMEOUT)
            request_state = self.input_buffer.get(request_id)
            if not self._watchdog_can_fire(request_state, reference_time):
                return
            request_state["completion_reason"] = "asr_result_timeout"
            self.logger.warning(
                "Volcengine realtime ASR result timeout for request %s: no ASR result within %.1fs after audio end",
                request_id,
                self.__class__.ASR_RESULT_TIMEOUT,
            )
            await self._fail_request(
                request_id,
                "实时语音识别超时，服务端未能识别语音内容，请重试。",
            )
        except asyncio.CancelledError:
            return

    def _persist_retained_upstream_audio(
        self, request_id: str, request_state: dict[str, Any]
    ) -> None:
        if self.audio_retention_dir is None:
            return
        if request_state.get("audio_retention_saved", False):
            return
        retained_audio = request_state.get("retained_upstream_audio")
        if not retained_audio:
            return
        save_dir = Path(self.audio_retention_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        completion_reason = request_state.get("completion_reason") or "unknown"
        base_name = f"{request_id}_{completion_reason}"
        wav_path = save_dir / f"{base_name}.wav"
        metadata_path = save_dir / f"{base_name}.json"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(self.__class__.N_CHANNELS)
            wav_file.setsampwidth(self.__class__.SAMPLE_WIDTH)
            wav_file.setframerate(self.__class__.INPUT_FRAME_RATE)
            wav_file.writeframes(bytes(retained_audio))
        metadata = {
            "request_id": request_id,
            "completion_reason": completion_reason,
            "user_id": request_state.get("user_id"),
            "character_id": request_state.get("character_id"),
            "bytes": len(retained_audio),
            "duration_seconds": len(retained_audio)
            / (
                self.__class__.INPUT_FRAME_RATE
                * self.__class__.N_CHANNELS
                * self.__class__.SAMPLE_WIDTH
            ),
            "input_audio_bytes_received": request_state.get(
                "input_audio_bytes_received", 0
            ),
            "audio_end_sent_time": request_state.get("audio_end_sent_time"),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        request_state["audio_retention_saved"] = True

    async def _generate_fallback_tts_audio(
        self,
        request_id: str,
        request_state: dict[str, Any],
        fallback_text: str,
    ) -> dict[str, Any]:
        dag_conf = request_state["dag"].conf
        tts_adapter = dag_conf.get("fallback_tts_adapter")
        if tts_adapter is None:
            raise RuntimeError("fallback tts adapter is not configured")
        voice_name = dag_conf.get("voice_name") or request_state.get("voice_name")
        if not voice_name:
            raise RuntimeError("fallback tts voice_name is not configured")
        voice_speed = dag_conf.get("voice_speed", 1.0)
        language = dag_conf.get("language", "zh")
        tts_request_id = f"{request_id}_fallback_tts"
        tts_adapter.input_buffer[tts_request_id] = {
            "api_keys": request_state.get("api_keys", {}),
            "voice_name": voice_name,
            "voice_speed": voice_speed,
            "voice_style": None,
            "language": language,
        }
        try:
            ret_dict = await tts_adapter._generate_tts(
                request_id=tts_request_id,
                text=fallback_text,
                voice_name=voice_name,
                voice_speed=voice_speed,
                language=language,
            )
        finally:
            tts_adapter.input_buffer.pop(tts_request_id, None)
        pcm_bytes = self._normalize_fallback_tts_audio(ret_dict["audio"])
        normalized_ret_dict = {
            "audio": io.BytesIO(pcm_bytes),
            "speech_text": fallback_text,
            "speech_time": ret_dict.get("speech_time", [(0, 0.0)]),
            "duration": len(pcm_bytes)
            / (
                self.__class__.N_CHANNELS
                * self.__class__.SAMPLE_WIDTH
                * self.__class__.FRAME_RATE
            ),
        }
        normalized_ret_dict["audio"].seek(0)
        return normalized_ret_dict

    def _normalize_fallback_tts_audio(self, audio_io: io.BytesIO) -> bytes:
        audio_io.seek(0)
        audio_bytes = audio_io.read()
        if audio_bytes[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                pcm_bytes = wav_file.readframes(wav_file.getnframes())
        else:
            n_channels = self.__class__.N_CHANNELS
            sample_width = self.__class__.SAMPLE_WIDTH
            frame_rate = 16000
            pcm_bytes = audio_bytes
        if n_channels != self.__class__.N_CHANNELS:
            raise RuntimeError(f"fallback tts audio channel mismatch: {n_channels}")
        if sample_width != self.__class__.SAMPLE_WIDTH:
            raise RuntimeError(f"fallback tts sample width mismatch: {sample_width}")
        if frame_rate != self.__class__.FRAME_RATE:
            pcm_bytes = resample_pcm(pcm_bytes, frame_rate, self.__class__.FRAME_RATE)
        return pcm_bytes

    async def _send_asr_fallback(self, request_id: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            return
        upstream_seconds = self._calc_upstream_seconds(request_state)
        fallback_text = random.choice(self.__class__._FALLBACK_TEXTS)
        request_state["completion_reason"] = "asr_fallback"
        self.logger.info(
            "Volcengine realtime ASR fallback for request %s: upstream_audio=%.3fs, fallback_text=%s",
            request_id,
            upstream_seconds,
            fallback_text,
        )
        try:
            dag = request_state["dag"]
            if dag.status != DAGStatus.RUNNING:
                self.logger.warning(
                    "Volcengine realtime ASR fallback skipped for request %s: DAG status=%s",
                    request_id,
                    dag.status,
                )
                await self._close_session(request_id)
                return
            ret_dict = await self._generate_fallback_tts_audio(
                request_id=request_id,
                request_state=request_state,
                fallback_text=fallback_text,
            )
            request_state["chunk_received"] = request_state.get("chunk_received", 0) + 1
            await self._send_audio_to_downstream(request_id, ret_dict, 0)
            request_state["response_done"] = True
            request_state["assistant_output"] = fallback_text
            # Close the volcengine session to send stream-end signals to
            # downstream nodes (a2f, s2m, blendshapes, callback).  Without
            # this the main _receive_pcm loop will keep waiting for vendor
            # events that never arrive (because the audio was too short for
            # volcengine to produce a meaningful response), and the DAG will
            # hang until its 120 s timeout.
            await self._close_session(request_id)
        except Exception as exc:
            self.logger.error(
                "Volcengine realtime ASR fallback failed for request %s: %s",
                request_id,
                exc,
            )
            await self._fail_request(
                request_id,
                "实时语音识别超时，且兜底语音生成失败，请重试。",
            )
            return
        await self._close_session(request_id)

    async def _fail_request(self, request_id: str, reason: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            return
        request_state["completion_reason"] = (
            request_state.get("completion_reason") or "failed"
        )
        self.logger.error(
            "Volcengine realtime request %s failed: %s", request_id, reason
        )
        if not request_state.get("failure_callback_sent", False):
            callback_bytes_fn = request_state.get("callback_bytes_fn")
            if callback_bytes_fn is not None:
                try:
                    await failure_callback(reason, callback_bytes_fn)
                    request_state["failure_callback_sent"] = True
                except Exception as exc:
                    self.logger.error(
                        "Failed to send realtime failure callback for request %s: %s",
                        request_id,
                        exc,
                    )
        request_state["dag"].set_status(DAGStatus.FAILED)
        await self._close_session(request_id)

    @classmethod
    def _resolve_bot_name(cls, request_state: dict[str, Any]) -> str | None:
        configured_bot_name = (
            request_state.get("conversation_model_override") or ""
        ).strip()
        if configured_bot_name and not cls._looks_like_model_identifier(
            configured_bot_name
        ):
            return configured_bot_name

        prompt_name = cls._extract_name_from_prompt(
            request_state.get("user_prompt", "")
        )
        if prompt_name:
            return prompt_name

        character_name = (request_state.get("character_name") or "").strip()
        if character_name:
            return character_name

        default_bot_name = (request_state.get("voice_name") or "").strip()
        if default_bot_name:
            return default_bot_name

        return None

    async def _build_system_role(
        self,
        request_state: Dict[str, Any],
        cascade_memories: Union[None, Dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        user_prompt = request_state.get("user_prompt") or ""
        if user_prompt:
            parts.append(user_prompt.strip())
        if cascade_memories is not None:
            long_term = (cascade_memories.get("long_term_memory") or {}).get(
                "content", ""
            )
            if long_term:
                parts.append(f"<long_term_memory>: {long_term}")
            medium_term_entries = cascade_memories.get("medium_term_memory") or []
            medium_parts = [
                e.get("content", "") for e in medium_term_entries if e.get("content")
            ]
            if medium_parts:
                parts.append(f"<medium_term_memory>: " + "\n".join(medium_parts))
        profile_memory = request_state.get("profile_memory")
        if profile_memory is not None:
            profile_content = profile_memory.get("content", "")
            if profile_content:
                parts.append(f"<user_profile>: {profile_content}")
        return "\n".join(parts)

    @classmethod
    def _build_dialog_context(
        cls,
        request_state: Dict[str, Any],
        cascade_memories: Union[None, Dict[str, Any]],
    ) -> list[Dict[str, str]]:
        if cascade_memories is None:
            return []
        short_term = cascade_memories.get("short_term_memories") or []
        if not short_term:
            return []
        window = short_term[-cls.DIALOG_CONTEXT_MAX_TURNS :]
        context: list[Dict[str, str]] = []
        for entry in window:
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role and content:
                context.append({"role": role, "content": content})
        return context

    @staticmethod
    def _looks_like_model_identifier(value: str) -> bool:
        lowered = value.strip().lower()
        if not lowered:
            return False
        modelish_prefixes = (
            "ag-",
            "doubao",
            "qwen",
            "gpt",
            "claude",
            "gemini",
            "deepseek",
            "o1",
            "o3",
            "tts-",
            "speech-",
        )
        return lowered.startswith(modelish_prefixes) or "/" in lowered

    @staticmethod
    def _extract_name_from_prompt(prompt: str) -> str | None:
        if not prompt:
            return None
        patterns = (
            r"(?mi)^\s*(?:character\s*name|name|角色名|名字|名称)\s*[:：]\s*([^\n\r]{1,40})\s*$",
            r"(?mi)^\s*-\s*(?:character\s*name|name|角色名|名字|名称)\s*[:：]\s*([^\n\r]{1,40})\s*$",
        )
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                candidate = match.group(1).strip().strip("\"'`")
                candidate = re.sub(r"\s+", " ", candidate)
                if candidate:
                    return candidate
        return None

    @classmethod
    def _format_server_error_reason(cls, result: dict[str, Any]) -> str:
        code = result.get("code")
        payload_msg = result.get("payload_msg")
        error_text = ""
        if isinstance(payload_msg, dict):
            error_text = (
                payload_msg.get("error", "")
                or payload_msg.get("message", "")
                or json.dumps(payload_msg, ensure_ascii=False)
            )
        else:
            error_text = str(payload_msg)
        error_text = error_text.strip()
        if "DialogAudioIdleTimeoutError" in error_text:
            return "实时语音输入无有效语音内容，服务端未收到足够的连续音频，请重试。"
        if error_text:
            return f"火山实时语音服务错误(code={code}): {error_text}"
        return f"火山实时语音服务错误(code={code})"

    @classmethod
    def _generate_header(
        cls,
        *,
        message_type: int = CLIENT_FULL_REQUEST,
        serialization_method: int = JSON_SERIALIZATION,
        compression_type: int = GZIP_COMPRESSION,
        message_flags: int = MSG_WITH_EVENT,
        extension_header: bytes = b"",
    ) -> bytearray:
        header = bytearray()
        header_size = int(len(extension_header) / 4) + 1
        header.append((0b0001 << 4) | header_size)
        header.append((message_type << 4) | message_flags)
        header.append((serialization_method << 4) | compression_type)
        header.append(0x00)
        header.extend(extension_header)
        return header

    @classmethod
    def _encode_connection_event(cls, event: int, payload_obj: dict[str, Any]) -> bytes:
        payload_bytes = gzip.compress(
            json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
        )
        buf = bytearray(cls._generate_header())
        buf.extend(int(event).to_bytes(4, "big"))
        buf.extend((len(payload_bytes)).to_bytes(4, "big"))
        buf.extend(payload_bytes)
        return bytes(buf)

    @classmethod
    def _encode_session_event(
        cls, event: int, session_id: str, payload_obj: dict[str, Any]
    ) -> bytes:
        payload_bytes = gzip.compress(
            json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
        )
        session_id_bytes = session_id.encode("utf-8")
        buf = bytearray(cls._generate_header())
        buf.extend(int(event).to_bytes(4, "big"))
        buf.extend((len(session_id_bytes)).to_bytes(4, "big"))
        buf.extend(session_id_bytes)
        buf.extend((len(payload_bytes)).to_bytes(4, "big"))
        buf.extend(payload_bytes)
        return bytes(buf)

    @classmethod
    def _encode_audio_event(
        cls, event: int, session_id: str, audio_bytes: bytes
    ) -> bytes:
        payload_bytes = gzip.compress(audio_bytes)
        session_id_bytes = session_id.encode("utf-8")
        buf = bytearray(
            cls._generate_header(
                message_type=cls.CLIENT_AUDIO_ONLY_REQUEST,
                serialization_method=cls.NO_SERIALIZATION,
            )
        )
        buf.extend(int(event).to_bytes(4, "big"))
        buf.extend((len(session_id_bytes)).to_bytes(4, "big"))
        buf.extend(session_id_bytes)
        buf.extend((len(payload_bytes)).to_bytes(4, "big"))
        buf.extend(payload_bytes)
        return bytes(buf)

    @classmethod
    def _parse_response(cls, res: bytes) -> dict[str, Any]:
        if not isinstance(res, bytes) or len(res) < 4:
            return {}
        header_size = res[0] & 0x0F
        message_type = res[1] >> 4
        message_type_specific_flags = res[1] & 0x0F
        serialization_method = res[2] >> 4
        message_compression = res[2] & 0x0F
        payload = res[header_size * 4 :]
        result: dict[str, Any] = {"message_type": message_type}

        if message_type == cls.SERVER_ERROR_RESPONSE:
            code = int.from_bytes(payload[:4], "big")
            payload_size = int.from_bytes(payload[4:8], "big")
            payload_msg = payload[8 : 8 + payload_size]
            if message_compression == cls.GZIP_COMPRESSION:
                payload_msg = gzip.decompress(payload_msg)
            if serialization_method == cls.JSON_SERIALIZATION and payload_msg:
                payload_msg = json.loads(payload_msg.decode("utf-8"))
            else:
                payload_msg = payload_msg.decode("utf-8", errors="ignore")
            result["code"] = code
            result["payload_msg"] = payload_msg
            return result

        current_offset = 0
        if message_type_specific_flags & cls.MSG_WITH_EVENT > 0:
            result["event"] = int.from_bytes(
                payload[current_offset : current_offset + 4], "big"
            )
            current_offset += 4

        session_id_size = int.from_bytes(
            payload[current_offset : current_offset + 4], "big"
        )
        current_offset += 4
        result["session_id"] = payload[
            current_offset : current_offset + session_id_size
        ].decode("utf-8")
        current_offset += session_id_size

        payload_size = int.from_bytes(
            payload[current_offset : current_offset + 4], "big"
        )
        current_offset += 4
        payload_msg = payload[current_offset : current_offset + payload_size]
        event = result.get("event")
        if event == cls.EVENT_TTS_AUDIO:
            result["payload_msg"] = payload_msg
            return result
        if message_compression == cls.GZIP_COMPRESSION:
            payload_msg = gzip.decompress(payload_msg)
        if serialization_method == cls.JSON_SERIALIZATION and payload_msg:
            result["payload_msg"] = json.loads(payload_msg.decode("utf-8"))
        else:
            result["payload_msg"] = payload_msg
        return result

    @staticmethod
    def _extract_asr_text(payload_msg: Any) -> tuple[str, bool]:
        if not isinstance(payload_msg, dict):
            return "", False
        results = payload_msg.get("results") or []
        if not results:
            return "", False
        first_result = results[0] or {}
        text = first_result.get("text", "") or ""
        extra = payload_msg.get("extra") or {}
        is_final = bool(extra.get("endpoint")) or not first_result.get(
            "is_interim", True
        )
        return text, is_final

    @staticmethod
    def _float32_pcm_to_int16(pcm_bytes: bytes) -> bytes:
        float_audio = np.frombuffer(pcm_bytes, dtype=np.float32)
        float_audio = np.nan_to_num(float_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        int_audio = np.clip(float_audio, -1.0, 1.0)
        int_audio = (int_audio * 32767.0).astype(np.int16)
        return int_audio.tobytes()

    @staticmethod
    def _safe_response_header(ws: Any, header_name: str) -> str:
        response_headers = getattr(ws, "response_headers", None)
        if response_headers is None:
            return ""
        return response_headers.get(header_name, "")
