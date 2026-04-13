import pytest

from orchestrator.conversation.volcengine_realtime_voice_conversation_client import (
    VolcengineRealtimeVoiceConversationClient,
)


def _make_adapter() -> VolcengineRealtimeVoiceConversationClient:
    return VolcengineRealtimeVoiceConversationClient(
        name="volcengine_realtime_voice_agent_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        wss_url="wss://example.invalid/realtime",
        logger_cfg={"logger_name": "test_volcengine_context_injection"},
    )


def _base_request_state(**overrides):
    state = {
        "user_prompt": "You are a helpful assistant named Xiaoxiao.",
        "profile_memory": None,
        "cascade_memories": None,
        "character_name": "",
        "voice_name": "xiaoxiao",
        "conversation_model_override": "",
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_build_system_role_with_user_prompt_only():
    adapter = _make_adapter()
    state = _base_request_state()
    result = await adapter._build_system_role(state, None)
    assert result == "You are a helpful assistant named Xiaoxiao."


@pytest.mark.asyncio
async def test_build_system_role_includes_long_term_memory():
    adapter = _make_adapter()
    cascade = {
        "long_term_memory": {"content": "User loves cats and has two cats."},
        "medium_term_memory": [],
        "short_term_memories": [],
    }
    state = _base_request_state()
    result = await adapter._build_system_role(state, cascade)
    assert "<long_term_memory>: User loves cats and has two cats." in result
    assert "You are a helpful assistant named Xiaoxiao." in result


@pytest.mark.asyncio
async def test_build_system_role_includes_medium_term_memory():
    adapter = _make_adapter()
    cascade = {
        "long_term_memory": {},
        "medium_term_memory": [
            {"content": "User mentioned a trip to Japan."},
            {"content": "User enjoyed the food there."},
        ],
        "short_term_memories": [],
    }
    state = _base_request_state()
    result = await adapter._build_system_role(state, cascade)
    assert (
        "<medium_term_memory>: User mentioned a trip to Japan.\nUser enjoyed the food there."
        in result
    )


@pytest.mark.asyncio
async def test_build_system_role_includes_profile_memory():
    adapter = _make_adapter()
    state = _base_request_state(
        profile_memory={"content": "Age: 25, Occupation: Engineer"},
    )
    result = await adapter._build_system_role(state, None)
    assert "<user_profile>: Age: 25, Occupation: Engineer" in result


@pytest.mark.asyncio
async def test_build_system_role_empty_when_nothing_provided():
    adapter = _make_adapter()
    state = _base_request_state(user_prompt="", profile_memory=None)
    result = await adapter._build_system_role(state, None)
    assert result == ""


@pytest.mark.asyncio
async def test_build_system_role_combined():
    adapter = _make_adapter()
    cascade = {
        "long_term_memory": {"content": "Long term here"},
        "medium_term_memory": [{"content": "Medium term here"}],
        "short_term_memories": [],
    }
    state = _base_request_state(
        user_prompt="System prompt text",
        profile_memory={"content": "Profile data"},
    )
    result = await adapter._build_system_role(state, cascade)
    assert result.startswith("System prompt text")
    assert "<long_term_memory>: Long term here" in result
    assert "<medium_term_memory>: Medium term here" in result
    assert "<user_profile>: Profile data" in result


def test_build_dialog_context_returns_empty_when_no_memories():
    adapter = _make_adapter()
    state = _base_request_state()
    result = adapter._build_dialog_context(state, None)
    assert result == []


def test_build_dialog_context_returns_empty_when_no_short_term():
    adapter = _make_adapter()
    cascade = {
        "long_term_memory": {},
        "medium_term_memory": [],
        "short_term_memories": [],
    }
    state = _base_request_state()
    result = adapter._build_dialog_context(state, cascade)
    assert result == []


def test_build_dialog_context_from_short_term_memories():
    adapter = _make_adapter()
    cascade = {
        "short_term_memories": [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
        ],
    }
    state = _base_request_state()
    result = adapter._build_dialog_context(state, cascade)
    assert len(result) == 4
    assert result[0] == {"role": "user", "content": "Hello there"}
    assert result[3] == {
        "role": "assistant",
        "content": "Why did the chicken cross the road?",
    }


def test_build_dialog_context_truncates_to_max_turns():
    adapter = _make_adapter()
    memories = []
    for i in range(30):
        memories.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg_{i}"}
        )
    cascade = {"short_term_memories": memories}
    state = _base_request_state()
    result = adapter._build_dialog_context(state, cascade)
    assert len(result) == 20
    assert result[0]["content"] == "msg_10"
    assert result[-1]["content"] == "msg_29"


def test_build_dialog_context_skips_empty_entries():
    adapter = _make_adapter()
    cascade = {
        "short_term_memories": [
            {"role": "user", "content": "valid"},
            {"role": "", "content": "no role"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "also valid"},
        ],
    }
    state = _base_request_state()
    result = adapter._build_dialog_context(state, cascade)
    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "valid"}
    assert result[1] == {"role": "assistant", "content": "also valid"}


def test_dialog_context_max_turns_respects_override():
    class CustomClient(VolcengineRealtimeVoiceConversationClient):
        DIALOG_CONTEXT_MAX_TURNS = 5

    adapter = CustomClient(
        name="custom",
        agent_prompts_file="configs/agent_prompts.yaml",
        wss_url="wss://example.invalid/realtime",
        logger_cfg={"logger_name": "test_override"},
    )
    memories = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    cascade = {"short_term_memories": memories}
    state = _base_request_state()
    result = adapter._build_dialog_context(state, cascade)
    assert len(result) == 5
    assert result[0]["content"] == "m5"
