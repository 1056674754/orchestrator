import asyncio
import json
import os
import time
import uuid

import pytest

from orchestrator.classification.builder import build_classification_adapter
from orchestrator.data_structures.classification import ClassificationChunkBody, ClassificationChunkEnd, ClassificationChunkStart
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from orchestrator.profile.classification_stream_profile import ClassificationStreamProfile
from orchestrator.utils.log import logging
from orchestrator.utils.streamable import Streamable

motion_keywords = []
with open("configs/motion_kws.json", "r", encoding="utf-8") as f:
    motions = json.load(f)
    for motion in motions:
        if motion["motion_keywords_ch"]:
            motion_keywords.extend(motion["motion_keywords_ch"].split(","))
motion_keywords = list(set(motion_keywords))
print(f"Loaded {len(motion_keywords)} motion keywords")


class ClassificationResultCollector(Streamable):
    """Minimal collector for classification smoke tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {}

    async def _handle_start(self, chunk: ClassificationChunkStart, cur_time: float) -> None:
        self.input_buffer[chunk.request_id] = {"last_update_time": cur_time}
        self.results[chunk.request_id] = {"result": None, "message": None, "completed": False}

    async def _handle_body(self, chunk: ClassificationChunkBody, cur_time: float) -> None:
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        self.results[chunk.request_id]["result"] = chunk.classification_result
        self.results[chunk.request_id]["message"] = chunk.message

    async def _handle_end(self, chunk: ClassificationChunkEnd, cur_time: float) -> None:
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        self.results[chunk.request_id]["completed"] = True


async def _run_classification_stream_test(
    *,
    classification_client_cfg: dict,
    user_settings: dict,
    text: str,
    language: str = "zh",
    timeout_seconds: float = 10.0,
):
    """Run one classification stream test and return result plus elapsed time."""
    logger_cfg = classification_client_cfg["logger_cfg"]
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())

    collector = ClassificationResultCollector(logger_cfg=logger_cfg)
    asyncio.create_task(collector.run())

    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(language=language, user_settings=user_settings),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(name="classification_node", payload=adapter)
    collector_node = DAGNode(name="collector_node", payload=collector)
    graph.add_node(classification_node)
    graph.add_node(collector_node)
    graph.add_edge(classification_node.name, collector_node.name)
    graph.set_status(DAGStatus.RUNNING)

    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=classification_node.name)
    await adapter.feed_stream(start_chunk)
    for char in text:
        await adapter.feed_stream(TextChunkBody(request_id=request_id, text_segment=char))
    await adapter.feed_stream(TextChunkEnd(request_id=request_id))

    start_time = time.time()
    while True:
        result = collector.results.get(request_id)
        if result and result.get("completed"):
            elapsed = time.time() - start_time
            await adapter.interrupt()
            await collector.interrupt()
            await asyncio.sleep(adapter.sleep_time * 5)
            return result["result"], elapsed
        if time.time() - start_time > timeout_seconds:
            await adapter.interrupt()
            await collector.interrupt()
            raise TimeoutError("Classification stream timeout")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_openai_classification_client_stream():
    """Test OpenAI classification client streaming functionality.

    This test verifies that the OpenAI classification adapter can process text
    chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if OPENAI_API_KEY environment variable is not set.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("openai_api_key is not set, skipping test test_openai_classification_client_stream")

    logger_cfg = dict(
        logger_name="test_openai_classification_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    classification_client_cfg = dict(
        type="OpenAIClassificationClient",
        name="openai_classification_client",
        motion_keywords=motion_keywords,
        openai_model_name="gpt-4o-mini-2024-07-18",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                openai_api_key=openai_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "请转个圈"
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_xai_classification_client_stream():
    """Test XAI classification client streaming functionality.

    This test verifies that the XAI classification adapter can process text
    chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if XAI_API_KEY environment variable is not set.
    """
    xai_api_key = os.environ.get("XAI_API_KEY")
    if not xai_api_key:
        pytest.skip("xai_api_key is not set, skipping test test_xai_classification_client_stream")

    logger_cfg = dict(
        logger_name="test_xai_classification_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    classification_client_cfg = dict(
        type="XAIClassificationClient",
        name="xai_classification_client",
        motion_keywords=motion_keywords,
        xai_model_name="grok-3",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                xai_api_key=xai_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "请跳支舞"
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_gemini_classification_client_stream():
    """Test Gemini classification client streaming functionality.

    This test verifies that the Gemini classification adapter can process text
    chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if GEMINI_API_KEY environment variable is not set.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        pytest.skip("gemini_api_key is not set, skipping test test_gemini_classification_client_stream")

    logger_cfg = dict(
        logger_name="test_gemini_classification_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    classification_client_cfg = dict(
        type="GeminiClassificationClient",
        name="gemini_classification_client",
        motion_keywords=motion_keywords,
        gemini_model_name="gemini-2.5-flash-lite",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="en",
            user_settings=dict(
                gemini_api_key=gemini_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "Sing a song, please."
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_qwen_classification_client_stream():
    """Smoke test Qwen classification with correctness and latency budget.

    This is meant to replace manual phone-tapping for the classification hot path.
    It uses the real adapter, real streaming flow, and a small correctness set.
    """
    qwen_api_key = os.environ.get("QWEN_API_KEY")
    if not qwen_api_key:
        pytest.skip("qwen_api_key is not set, skipping test_qwen_classification_client_stream")

    max_elapsed_seconds = float(os.environ.get("QWEN_CLASSIFICATION_MAX_SECONDS", "8.0"))
    logger_cfg = dict(
        logger_name="test_qwen_classification_client_stream",
        file_level=logging.DEBUG,
        logger_path="logs/pytest.log",
    )
    classification_client_cfg = dict(
        type="QwenClassificationClient",
        name="qwen_classification_client",
        motion_keywords=motion_keywords,
        qwen_model_name="qwen-turbo-latest",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )

    cases = [
        ("你好呀", "accept"),
        ("给我唱首歌", "reject"),
        ("拜拜，先不聊了", "leave"),
    ]
    elapsed_values = []
    for text, expected in cases:
        result, elapsed = await _run_classification_stream_test(
            classification_client_cfg=classification_client_cfg,
            user_settings=dict(qwen_api_key=qwen_api_key),
            text=text,
            language="zh",
            timeout_seconds=max(max_elapsed_seconds * 2, 10.0),
        )
        elapsed_values.append(elapsed)
        assert result is not None
        assert result.value == expected

    assert max(elapsed_values) <= max_elapsed_seconds, (
        f"Qwen classification too slow: max_elapsed={max(elapsed_values):.3f}s, "
        f"budget={max_elapsed_seconds:.3f}s, cases={cases}"
    )


@pytest.mark.asyncio
async def test_sensenova_omni_classification_client_stream():
    """Test SenseNova Omni classification client streaming functionality.

    This test verifies that the SenseNova Omni classification adapter can
    process text chunks in streaming mode and correctly classify motion
    keywords.

    The test will be skipped if SENSENOVAOMNI_AK or SENSENOVAOMNI_SK
    environment variables are not set.
    """
    sensenovaomni_ak = os.environ.get("SENSENOVAOMNI_AK")
    sensenovaomni_sk = os.environ.get("SENSENOVAOMNI_SK")
    if not sensenovaomni_ak or not sensenovaomni_sk:
        pytest.skip(
            "sensenovaomni_ak or sensenovaomni_sk is not set, skipping test test_sensenova_omni_classification_client_stream"
        )

    logger_cfg = dict(
        logger_name="test_sensenova_omni_classification_client_stream",
        file_level=logging.DEBUG,
        logger_path="logs/pytest.log",
    )
    classification_client_cfg = dict(
        type="SenseNovaOmniClassificationClient",
        name="sensenovaomni_classification_client",
        motion_keywords=motion_keywords,
        wss_url="wss://api.sensenova.cn/agent-5o/duplex/ws2",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                sensenovaomni_ak=sensenovaomni_ak,
                sensenovaomni_sk=sensenovaomni_sk,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "你会跳舞吗"
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_sensenova_classification_client_stream():
    """Test SenseNova classification client streaming functionality.

    This test verifies that the SenseNova classification adapter can process
    text chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if SENSENOVA_AK or SENSENOVA_SK environment
    variables are not set.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    if not sensenova_ak or not sensenova_sk:
        pytest.skip(
            "sensenova_ak or sensenova_sk is not set, skipping test test_sensenova_classification_client_stream"
        )

    logger_cfg = dict(
        logger_name="test_sensenova_classification_client_stream",
        file_level=logging.DEBUG,
        logger_path="logs/pytest.log",
    )
    classification_client_cfg = dict(
        type="SenseNovaClassificationClient",
        name="sensenova_classification_client",
        motion_keywords=motion_keywords,
        sensenova_model_name="SenseNova-V6-5-Pro",
        sensenova_url="https://api.sensenova.cn/v1/llm/chat-completions",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "请唱个歌"
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_sensechat_classification_client_stream():
    """Test SenseChat classification client streaming functionality.

    This test verifies that the SenseChat classification adapter can process
    text chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if SENSECHAT_AK or SENSECHAT_SK environment
    variables are not set.
    """
    sensechat_ak = os.environ.get("SENSECHAT_AK")
    sensechat_sk = os.environ.get("SENSECHAT_SK")
    if not sensechat_ak or not sensechat_sk:
        pytest.skip(
            "sensechat_ak or sensechat_sk is not set, skipping test test_sensechat_classification_client_stream"
        )

    logger_cfg = dict(
        logger_name="test_sensechat_classification_client_stream",
        file_level=logging.DEBUG,
        logger_path="logs/pytest.log",
    )
    classification_client_cfg = dict(
        type="SenseChatClassificationClient",
        name="sensechat_classification_client",
        motion_keywords=motion_keywords,
        sensechat_model_name="SenseChat-5-1202",
        sensechat_url="https://api.sensenova.cn/v1/llm/chat-completions",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                sensechat_ak=sensechat_ak,
                sensechat_sk=sensechat_sk,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "请跳支舞"
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_deepseek_classification_client_stream():
    """Test DeepSeek classification client streaming functionality.

    This test verifies that the DeepSeek classification adapter can process
    text chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if DEEPSEEK_API_KEY environment variable is not
    set.
    """
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        pytest.skip("deepseek_api_key is not set, skipping test test_deepseek_classification_client_stream")

    logger_cfg = dict(
        logger_name="test_deepseek_classification_client_stream",
        file_level=logging.DEBUG,
        logger_path="logs/pytest.log",
    )
    classification_client_cfg = dict(
        type="DeepSeekClassificationClient",
        name="deepseek_classification_client",
        motion_keywords=motion_keywords,
        deepseek_model_name="deepseek-chat",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                deepseek_api_key=deepseek_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "请跳个舞"
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
