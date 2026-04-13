import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union
from urllib.parse import quote

import websockets

from ...data_structures.process_flow import DAGStatus
from ...data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ...utils.exception import MissingAPIKeyException
from ...utils.executor_registry import ExecutorRegistry
from .openai_realtime_asr_client import OpenAIRealtimeASRClient


class QwenRealtimeASRClient(OpenAIRealtimeASRClient):
    """Alibaba Bailian / DashScope realtime ASR client.

    DashScope's Qwen-ASR-Realtime follows the Realtime event model closely,
    but uses DashScope credentials, a model query parameter, and slightly
    different incremental transcript event names.
    """

    FRAME_RATE: int = 16000
    CHUNK_DURATION = 0.04
    CHUNK_N_BYTES = 1280
    ExecutorRegistry.register_class("QwenRealtimeASRClient")

    def __init__(
        self,
        name: str,
        wss_url: str,
        qwen_model_name: str = "qwen-asr-realtime",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        commit_timeout: float = 2.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            wss_url=wss_url,
            openai_model_name=qwen_model_name,
            proxy_url=proxy_url,
            request_timeout=request_timeout,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            commit_timeout=commit_timeout,
            max_workers=max_workers,
            thread_pool_executor=thread_pool_executor,
            logger_cfg=logger_cfg,
        )
        self.qwen_model_name = qwen_model_name

    async def _create_connection(self, request_id: str, cur_time: float) -> None:
        qwen_api_key = self.input_buffer[request_id].get("api_keys", {}).get("qwen_api_key", "")
        if not qwen_api_key:
            msg = "Qwen API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)
        language = self.input_buffer[request_id].get("language", "zh")
        model_name = self.qwen_model_name
        wss_url = f"{self.wss_url}?model={quote(model_name)}"
        additional_headers = {
            "Authorization": f"Bearer {qwen_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            connect_start_time = time.time()
            ws = await websockets.connect(
                wss_url,
                proxy=self.proxy_url,
                additional_headers=additional_headers,
                ping_interval=self.request_timeout,
                ping_timeout=self.request_timeout,
            )
            data = {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm",
                    "sample_rate": self.__class__.FRAME_RATE,
                    "input_audio_transcription": {
                        "language": language,
                    },
                    "turn_detection": None,
                },
            }
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(self.executor, json.dumps, data)
            await ws.send(data)
            self.input_buffer[request_id]["ws_client"] = ws
            self.input_buffer[request_id]["commit_time"] = None
            dag_start_time = self.input_buffer[request_id].get("dag_start_time", None)
            since_dag_start = time.time() - dag_start_time if dag_start_time is not None else None
            self.logger.info(
                "Qwen realtime ASR connection ready for request %s: model=%s, connect_elapsed=%.3fs%s",
                request_id,
                model_name,
                time.time() - connect_start_time,
                f", since_dag_start={since_dag_start:.3f}s" if since_dag_start is not None else "",
            )
        except Exception as e:
            self.input_buffer[request_id]["connection_failed"] = True
            self.logger.error(f"Failed to create connection to Qwen realtime ASR server: {e}")
            raise e

    async def _send_to_downstream_and_clean(self, request_id: str) -> None:
        dag = self.input_buffer[request_id]["dag"]
        ws_client = self.input_buffer[request_id].get("ws_client", None)
        while ws_client is None:
            if self.input_buffer[request_id]["connection_failed"]:
                self.input_buffer.pop(request_id, None)
                return
            await asyncio.sleep(self.sleep_time)
            ws_client = self.input_buffer[request_id].get("ws_client", None)
        while (
            self.input_buffer[request_id]["chunk_sent_to_server"]
            < self.input_buffer[request_id]["chunk_received_from_upstream"]
        ):
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                self.logger.warning(
                    "Streaming audio sending interrupted by DAG status %s for request %s",
                    dag.status,
                    request_id,
                )
                await ws_client.close()
                self.input_buffer.pop(request_id, None)
                return

        start_time = time.time()
        loop = asyncio.get_event_loop()
        await ws_client.send(
            await loop.run_in_executor(
                self.executor,
                json.dumps,
                {"type": "input_audio_buffer.commit"},
            )
        )
        await ws_client.send(
            await loop.run_in_executor(
                self.executor,
                json.dumps,
                {"type": "session.finish"},
            )
        )
        self.input_buffer[request_id]["commit_time"] = time.time()
        dag_start_time = self.input_buffer[request_id].get("dag_start_time", None)
        since_dag_start = (
            self.input_buffer[request_id]["commit_time"] - dag_start_time if dag_start_time is not None else None
        )
        self.logger.info(
            "Realtime ASR commit sent for request %s%s",
            request_id,
            f", since_dag_start={since_dag_start:.3f}s" if since_dag_start is not None else "",
        )

        dag = self.input_buffer[request_id]["dag"]
        node_name = self.input_buffer[request_id]["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        if len(downstream_nodes) == 0:
            self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
        for node in downstream_nodes:
            await node.payload.feed_stream(
                TextChunkStart(
                    request_id=request_id,
                    node_name=node.name,
                    dag=dag,
                )
            )

        asr_text = ""
        commit_time = self.input_buffer[request_id]["commit_time"]
        first_partial_logged = False
        while True:
            current_time = time.time()
            if current_time - commit_time > self.commit_timeout:
                break
            try:
                remaining_time = self.commit_timeout - (current_time - commit_time)
                if remaining_time <= 0:
                    break
                message = await asyncio.wait_for(ws_client.recv(), timeout=remaining_time)
                if not isinstance(message, str):
                    continue
                msg = json.loads(message)
                event_type = msg.get("type")
                if event_type == "conversation.item.input_audio_transcription.text":
                    text = msg.get("text", "") or ""
                    if text:
                        asr_text += text
                        if not first_partial_logged:
                            first_partial_logged = True
                            since_dag_partial = time.time() - dag_start_time if dag_start_time is not None else None
                            self.logger.info(
                                "Realtime ASR first partial ready for request %s: text_chars=%d, commit_to_partial=%.3fs%s",
                                request_id,
                                len(text),
                                time.time() - commit_time,
                                f", since_dag_start={since_dag_partial:.3f}s" if since_dag_partial is not None else "",
                            )
                        for node in downstream_nodes:
                            await node.payload.feed_stream(
                                TextChunkBody(
                                    request_id=request_id,
                                    text_segment=text,
                                )
                            )
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    completed_text = msg.get("transcript", "") or ""
                    if completed_text and completed_text != asr_text:
                        delta = completed_text[len(asr_text) :] if completed_text.startswith(asr_text) else completed_text
                        asr_text = completed_text
                        if delta:
                            for node in downstream_nodes:
                                await node.payload.feed_stream(
                                    TextChunkBody(
                                        request_id=request_id,
                                        text_segment=delta,
                                    )
                                )
                    since_dag_final = time.time() - dag_start_time if dag_start_time is not None else None
                    self.logger.info(
                        "Realtime ASR final text ready for request %s: text_chars=%d, commit_to_final=%.3fs%s",
                        request_id,
                        len(asr_text),
                        time.time() - commit_time,
                        f", since_dag_start={since_dag_final:.3f}s" if since_dag_final is not None else "",
                    )
                    break
                elif event_type == "session.finished":
                    break
                elif event_type == "error":
                    raise Exception(msg.get("error"))
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.error(f"ASR error for request {request_id}: {e}")
                break

        if not asr_text:
            self.logger.warning(f"ASR text is empty for request {request_id}, use default text.")
            for node in downstream_nodes:
                await node.payload.feed_stream(
                    TextChunkBody(
                        request_id=request_id,
                        text_segment="",
                    )
                )

        for node in downstream_nodes:
            await node.payload.feed_stream(TextChunkEnd(request_id=request_id))
        end_time = time.time()
        msg = f"Qwen realtime ASR finished in {end_time - start_time:.3f} seconds from audio stream end"
        dag_start_time = self.input_buffer[request_id].get("dag_start_time", None)
        if dag_start_time is not None:
            msg += f", in {end_time - dag_start_time:.3f} seconds from DAG start"
        msg += f", for request {request_id}, speech_text: {asr_text}"
        self.logger.debug(msg)
        self.input_buffer.pop(request_id, None)
        await ws_client.close()
