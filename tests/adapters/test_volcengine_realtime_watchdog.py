import asyncio
import json
import wave
from pathlib import Path

import pytest

from orchestrator.conversation.volcengine_realtime_voice_conversation_client import (
    VolcengineRealtimeVoiceConversationClient,
)


def _make_adapter() -> VolcengineRealtimeVoiceConversationClient:
    return VolcengineRealtimeVoiceConversationClient(
        name="volcengine_realtime_voice_agent_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        wss_url="wss://example.invalid/realtime",
        logger_cfg={"logger_name": "test_volcengine_realtime_watchdog"},
    )


@pytest.mark.asyncio
async def test_short_fallback_watchdog_fires_despite_intermediate_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    request_id = "req-short"
    fired: list[str] = []

    async def fake_send_asr_fallback(target_request_id: str) -> None:
        fired.append(target_request_id)

    monkeypatch.setattr(adapter, "_send_asr_fallback", fake_send_asr_fallback)
    monkeypatch.setattr(adapter.__class__, "ASR_FALLBACK_TIMEOUT", 0.02)

    adapter.input_buffer[request_id] = {
        "closed": False,
        "response_done": False,
        "asr_final_received": False,
        "audio_end_sent_time": 1.0,
        "input_audio_bytes_received": 14592,
        "sample_width": 2,
        "n_channels": 1,
        "frame_rate": 16000,
    }

    task = asyncio.create_task(adapter._run_short_fallback_watchdog(request_id, 1.0))
    for _ in range(3):
        adapter.input_buffer[request_id]["last_update_time"] = (
            asyncio.get_running_loop().time()
        )
        await asyncio.sleep(0.005)
    await task

    assert fired == [request_id]


@pytest.mark.asyncio
async def test_short_fallback_watchdog_does_not_fire_after_final_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    request_id = "req-final"
    fired: list[str] = []

    async def fake_send_asr_fallback(target_request_id: str) -> None:
        fired.append(target_request_id)

    monkeypatch.setattr(adapter, "_send_asr_fallback", fake_send_asr_fallback)
    monkeypatch.setattr(adapter.__class__, "ASR_FALLBACK_TIMEOUT", 0.02)

    adapter.input_buffer[request_id] = {
        "closed": False,
        "response_done": False,
        "asr_final_received": False,
        "audio_end_sent_time": 2.0,
        "input_audio_bytes_received": 14592,
        "sample_width": 2,
        "n_channels": 1,
        "frame_rate": 16000,
    }

    task = asyncio.create_task(adapter._run_short_fallback_watchdog(request_id, 2.0))
    await asyncio.sleep(0.005)
    adapter.input_buffer[request_id]["asr_final_received"] = True
    await task

    assert fired == []


@pytest.mark.asyncio
async def test_asr_timeout_watchdog_fails_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter()
    request_id = "req-timeout"
    failures: list[tuple[str, str]] = []

    async def fake_fail_request(target_request_id: str, reason: str) -> None:
        failures.append((target_request_id, reason))

    monkeypatch.setattr(adapter, "_fail_request", fake_fail_request)
    monkeypatch.setattr(adapter.__class__, "ASR_RESULT_TIMEOUT", 0.02)

    adapter.input_buffer[request_id] = {
        "closed": False,
        "response_done": False,
        "asr_final_received": False,
        "audio_end_sent_time": 3.0,
        "completion_reason": None,
    }

    await adapter._run_asr_timeout_watchdog(request_id, 3.0)

    assert failures == [
        (request_id, "实时语音识别超时，服务端未能识别语音内容，请重试。")
    ]
    assert adapter.input_buffer[request_id]["completion_reason"] == "asr_result_timeout"


def test_persist_retained_upstream_audio(tmp_path: Path) -> None:
    adapter = _make_adapter()
    adapter.audio_retention_dir = str(tmp_path)
    request_id = "req-save"
    request_state = {
        "audio_retention_saved": False,
        "retained_upstream_audio": bytearray(b"\x00\x00" * 1600),
        "completion_reason": "asr_fallback",
        "user_id": "user-1",
        "character_id": "char-1",
        "input_audio_bytes_received": 3200,
        "audio_end_sent_time": 123.45,
    }

    adapter._persist_retained_upstream_audio(request_id, request_state)

    wav_path = tmp_path / f"{request_id}_asr_fallback.wav"
    metadata_path = tmp_path / f"{request_id}_asr_fallback.json"
    assert wav_path.exists()
    assert metadata_path.exists()

    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == adapter.INPUT_FRAME_RATE
        assert wav_file.getnchannels() == adapter.N_CHANNELS
        assert wav_file.getsampwidth() == adapter.SAMPLE_WIDTH
        assert wav_file.getnframes() == 1600

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["request_id"] == request_id
    assert metadata["completion_reason"] == "asr_fallback"
    assert metadata["bytes"] == 3200
