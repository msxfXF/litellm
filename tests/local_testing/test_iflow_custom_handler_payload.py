from __future__ import annotations

from typing import Any, Dict, List

from iflow_custom_handler import IFlowLLM


def test_iflow_payload_converts_user_content_to_parts():
    llm = IFlowLLM()
    payload = llm._build_iflow_payload(  # type: ignore[attr-defined]
        model="iflow/glm-4.7",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
    )
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


def test_iflow_payload_keeps_tool_content_as_string():
    llm = IFlowLLM()
    payload = llm._build_iflow_payload(  # type: ignore[attr-defined]
        model="iflow/glm-4.7",
        messages=[
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "{\"ok\": true}"},
        ],
        optional_params={},
    )
    assert payload["messages"][1]["role"] == "tool"
    assert payload["messages"][1]["content"] == "{\"ok\": true}"
    assert payload["messages"][1]["tool_call_id"] == "x"


def test_iflow_payload_maps_functions_to_tools_and_function_call_to_tool_choice():
    llm = IFlowLLM()
    functions: List[Dict[str, Any]] = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    payload = llm._build_iflow_payload(  # type: ignore[attr-defined]
        model="iflow/glm-4.7",
        messages=[{"role": "user", "content": "weather"}],
        optional_params={"functions": functions, "function_call": {"name": "get_weather"}},
    )
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "get_weather"
    assert payload["tool_choice"]["type"] == "function"
    assert payload["tool_choice"]["function"]["name"] == "get_weather"

