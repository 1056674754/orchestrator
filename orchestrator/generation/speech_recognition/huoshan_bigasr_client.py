import asyncio
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import websockets

from ...data_structures.process_flow import DAGStatus
from ...data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ...utils.executor_registry import ExecutorRegistry
from .huoshan_asr_client import (
    HuoshanASRClient,
    HuoshanASRClientError,
    _encode_request,
    _generate_header,
)


class HuoshanBigmodelASRClient(HuoshanASRClient):
    """Volcengine big-model duplex streaming ASR client.

    Uses the newer `/api/v3/sauc/bigmodel_async` line with earlier partial
    results and optional second-pass refinement support.
    """

    SIGNATURE_USER_AGENT = "DLP3D-VolcengineBigASR/1.0"
    REQUEST_PATH = "/api/v3/sauc/bigmodel_async"
    ExecutorRegistry.register_class("HuoshanBigmodelASRClient")

    def __init__(
        self,
        name: str,
        wss_url: str,
        default_language: str = "zh-CN",
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        commit_timeout: float = 4.0,
        enable_nonstream: bool = True,
        enable_itn: bool = True,
        enable_punc: bool = True,
        enable_ddc: bool = True,
        enable_accelerate_text: bool = True,
        accelerate_score: int = 8,
        end_window_size: int = 500,
        force_to_speech_time: int = 800,
        show_utterances: bool = True,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            wss_url=wss_url,
            cluster_id="unused",
            default_language=default_language,
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
        self.enable_nonstream = enable_nonstream
        self.enable_itn = enable_itn
        self.enable_punc = enable_punc
        self.enable_ddc = enable_ddc
        self.enable_accelerate_text = enable_accelerate_text
        self.accelerate_score = accelerate_score
        self.end_window_size = end_window_size
        self.force_to_speech_time = force_to_speech_time
        self.show_utterances = show_utterances

    def _construct_request(self, request_id: str, appid: str, token: str, language: str) -> dict[str, Any]:
        return {
            "app": {
                "appid": appid,
                "token": token,
            },
            "user": {"uid": self.__class__.HOSTNAME_FULL},
            "request": {
                "reqid": request_id,
                "sequence": 1,
                "model_name": "bigmodel",
                "show_utterances": self.show_utterances,
                "result_type": "single",
                "enable_nonstream": self.enable_nonstream,
                "enable_itn": self.enable_itn,
                "enable_punc": self.enable_punc,
                "enable_ddc": self.enable_ddc,
                "enable_accelerate_text": self.enable_accelerate_text,
                "accelerate_score": self.accelerate_score,
                "end_window_size": self.end_window_size,
                "force_to_speech_time": self.force_to_speech_time,
            },
            "audio": {
                "format": "raw",
                "rate": self.__class__.FRAME_RATE,
                "language": language,
                "bits": 16,
                "channel": 1,
            },
        }

    async def _create_connection(self, request_id: str, cur_time: float) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            self.logger.warning("Skip creating Huoshan big ASR connection for inactive request %s", request_id)
            return
        api_keys = request_state.get("api_keys", {})
        app_id = api_keys.get("volcengine_app_id", "")
        token = api_keys.get("volcengine_token", "")
        secret_key = api_keys.get("volcengine_secret_key", "")
        if not app_id or not token:
            from ...utils.exception import MissingAPIKeyException

            msg = "Volcengine app ID or access token is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)
        language = request_state.get("language", self.default_language)
        request_dict = self._construct_request(request_id, app_id, token, language)
        loop = asyncio.get_event_loop()
        request_bytes = await loop.run_in_executor(self.executor, _encode_request, request_dict)
        header = _generate_header(self.__class__.CLIENT_FULL_REQUEST_TYPE, self.__class__.NO_SEQUENCE_FLAG)
        full_client_request = bytearray(header)
        full_client_request.extend((len(request_bytes)).to_bytes(4, "big"))
        full_client_request.extend(request_bytes)
        auth_headers = {"Authorization": "Bearer; {}".format(token)}
        auth_mode = "token"
        if secret_key:
            auth_headers = self._build_signature_authorization(
                token=token,
                secret_key=secret_key,
                request_bytes=request_bytes,
                full_client_request=bytes(full_client_request),
                request_path=self.__class__.REQUEST_PATH,
            )
            auth_mode = "signature"
        try:
            connect_start_time = time.time()
            self.logger.info(
                "Connecting to Volcengine big ASR via %s auth for request %s",
                auth_mode,
                request_id,
            )
            ws = await websockets.connect(self.wss_url, additional_headers=auth_headers, max_size=100 * 1024 * 1024)
            await ws.send(full_client_request)
            res = await ws.recv()
            result = self._parse_response(res)
            if "payload_msg" in result and result["payload_msg"].get("code", 1000) != 1000:
                msg = f"Huoshan big ASR server returned error {result['payload_msg']} for request {request_id}"
                self.logger.error(msg)
                raise HuoshanASRClientError(msg)
            request_state = self.input_buffer.get(request_id)
            if request_state is None:
                self.logger.warning("Request %s disappeared before big ASR connection was stored", request_id)
                await ws.close()
                return
            request_state["ws_client"] = ws
            request_state["commit_time"] = None
            dag_start_time = request_state.get("dag_start_time", None)
            since_dag_start = time.time() - dag_start_time if dag_start_time is not None else None
            self.logger.info(
                "Volcengine big ASR connection ready for request %s: auth=%s, connect_elapsed=%.3fs%s",
                request_id,
                auth_mode,
                time.time() - connect_start_time,
                f", since_dag_start={since_dag_start:.3f}s" if since_dag_start is not None else "",
            )
        except Exception as e:
            request_state = self.input_buffer.get(request_id)
            if request_state is not None:
                request_state["connection_failed"] = True
            self.logger.error(f"Failed to create connection to Huoshan big ASR server: {e}")
            raise e

    @staticmethod
    def _extract_text(payload_msg: Dict[str, Any]) -> tuple[str, bool]:
        result = payload_msg.get("result", {})
        if isinstance(result, list) and result:
            text = result[0].get("text", "") or ""
            return text, payload_msg.get("sequence", 1) < 0
        if isinstance(result, dict):
            text = result.get("text", "") or ""
            utterances = result.get("utterances") or []
            definite = any(isinstance(item, dict) and item.get("definite") for item in utterances)
            return text, definite or payload_msg.get("sequence", 1) < 0
        return "", payload_msg.get("sequence", 1) < 0

    async def _send_to_downstream_and_clean(self, request_id: str) -> None:
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            self.logger.warning("Request %s not found before ASR finalize", request_id)
            return
        dag = request_state["dag"]
        ws_client = request_state.get("ws_client", None)
        while ws_client is None:
            request_state = self.input_buffer.get(request_id)
            if request_state is None:
                self.logger.warning("Request %s disappeared before big ASR websocket became ready", request_id)
                return
            if request_state["connection_failed"]:
                self.input_buffer.pop(request_id, None)
                return
            await asyncio.sleep(self.sleep_time)
            ws_client = request_state.get("ws_client", None)
        while True:
            request_state = self.input_buffer.get(request_id)
            if request_state is None:
                self.logger.warning("Request %s disappeared while waiting for queued PCM chunks", request_id)
                await ws_client.close()
                return
            if request_state["chunk_sent_to_server"] >= request_state["chunk_received_from_upstream"]:
                break
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
        mute_chunk_bytes = bytes(self.__class__.CHUNK_N_BYTES)
        payload_bytes = await loop.run_in_executor(self.executor, gzip.compress, mute_chunk_bytes)
        header = _generate_header(self.__class__.CLIENT_AUDIO_ONLY_REQUEST_TYPE, self.__class__.NEG_SEQUENCE_FLAG)
        audio_only_request = bytearray(header)
        audio_only_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        audio_only_request.extend(payload_bytes)
        await ws_client.send(audio_only_request)
        commit_time = time.time()
        request_state = self.input_buffer.get(request_id)
        if request_state is None:
            self.logger.warning("Request %s disappeared immediately after ASR commit", request_id)
            await ws_client.close()
            return
        request_state["commit_time"] = commit_time

        dag = request_state["dag"]
        node_name = request_state["node_name"]
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
        final_seen = False
        while True:
            current_time = time.time()
            if current_time - commit_time > self.commit_timeout:
                break
            try:
                remaining_time = self.commit_timeout - (current_time - commit_time)
                if remaining_time <= 0:
                    break
                message = await asyncio.wait_for(ws_client.recv(), timeout=remaining_time)
                result = self._parse_response(message)
                payload_msg = result.get("payload_msg")
                if isinstance(payload_msg, dict) and payload_msg.get("code", 1000) != 1000:
                    self.logger.error(
                        "Huoshan big ASR server returned error %s for request %s",
                        payload_msg,
                        request_id,
                    )
                    break
                if not isinstance(payload_msg, dict):
                    continue
                new_text, final_seen = self._extract_text(payload_msg)
                if new_text and new_text != asr_text:
                    delta = new_text[len(asr_text) :] if new_text.startswith(asr_text) else new_text
                    asr_text = new_text
                    if delta:
                        for node in downstream_nodes:
                            await node.payload.feed_stream(
                                TextChunkBody(
                                    request_id=request_id,
                                    text_segment=delta,
                                )
                            )
                if final_seen:
                    break
            except asyncio.TimeoutError:
                self.logger.error(
                    f"ASR timeout for request {request_id}: "
                    + f"spent {current_time - commit_time:.3f} seconds, "
                    + f"timeout {self.commit_timeout:.3f} seconds"
                )
                break
            except Exception as e:
                self.logger.error(f"ASR error for request {request_id}: {e}")
                break
        if not asr_text:
            self.logger.warning(f"ASR text is empty for request {request_id}, use default text.")
            for node in downstream_nodes:
                await node.payload.feed_stream(TextChunkBody(request_id=request_id, text_segment=""))
        for node in downstream_nodes:
            await node.payload.feed_stream(TextChunkEnd(request_id=request_id))
        end_time = time.time()
        msg = f"Streaming bigmodel ASR finished in {end_time - start_time:.3f} seconds from audio stream end"
        request_state = self.input_buffer.get(request_id)
        dag_start_time = request_state.get("dag_start_time", None) if request_state is not None else None
        if dag_start_time is not None:
            msg += f", in {end_time - dag_start_time:.3f} seconds from DAG start"
        msg += f", for request {request_id}, speech_text: {asr_text}"
        self.logger.debug(msg)
        self.input_buffer.pop(request_id, None)
        await ws_client.close()
