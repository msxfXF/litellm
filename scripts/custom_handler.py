"""
Custom provider handler calling https://drlj.cn/openai/responses.
Headers/user-agent/instructions are hard-coded to match test_openai_responses.py.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
import requests
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import Choices, Message, ModelResponse, Usage

DRLJ_URL = "https://drlj.cn/openai/responses"
DEFAULT_HEADERS: Dict[str, str] = {
    "conversation_id": "019b7d1e-7b1c-7352-83ef-38214a25e1f4",
    "session_id": "019b7d1e-7b1c-7352-83ef-38214a25e1f4",
    "accept": "text/event-stream",
    "authorization": "Bearer cr_1fb51b828e394f883abfb04738a45ccc315b876ba27e45128e3d77625fbf5647",
    "content-type": "application/json",
    "user-agent": "codex_cli_rs/0.77.0 (Windows 10.0.19045; x86_64) unknown",
    "originator": "codex_cli_rs",
}
INSTRUCTIONS = (
    "You are GPT-5.2 running in the Codex CLI, a terminal-based coding assistant. "
    "Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.\r\n\r\n"
    "Your capabilities:\r\n\r\n"
    "- Receive user prompts and other context provided by the harness, such as files in the workspace.\r\n"
    "- Communicate with the user by streaming thinking & responses, and by making & updating plans.\r\n"
    "- Emit function calls to run terminal commands and apply patches. Depending on how this specific run is configured, "
    "you can request that these function calls be escalated to the user for approval before running. More on this in the "
    '"Sandbox and approvals" section.\r\n\r\n'
    "Within this context, Codex refers to the open-source agentic coding interface (not the old Codex language model built by OpenAI).\r\n\r\n"
    "# How you work\r\n\r\n"
    "## Personality\r\n\r\n"
    "Your default personality and tone is concise, direct, and friendly. You communicate efficiently, always keeping the "
    "user clearly informed about ongoing actions without unnecessary detail. You always prioritize actionable guidance, "
    "clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose "
    "explanations about your work.\r\n\r\n"
    "## AGENTS.md spec\r\n"
    "- Repos often contain AGENTS.md files. These files can appear anywhere within the repository.\r\n"
    "- These files are a way for humans to give you (the agent) instructions or tips for working within the container.\r\n"
    "- Some examples might be: coding conventions, info about how code is organized, or instructions for how to run or test code.\r\n"
    "- Instructions in AGENTS.md files:\r\n"
    "    - The scope of an AGENTS.md file is the entire directory tree rooted at the folder that contains it.\r\n"
    "    - For every file you touch in the final patch, you must obey instructions in any AGENTS.md file whose scope includes that file.\r\n"
    "    - Instructions about code style, structure, naming, etc. apply only to code within the AGENTS.md file's scope, unless the file states otherwise.\r\n"
    "    - More-deeply-nested AGENTS.md files take precedence in the case of conflicting instructions.\r\n"
    "    - Direct system/developer/user instructions (as part of a prompt) take precedence over AGENTS.md instructions.\r\n"
    "- The contents of the AGENTS.md file at the root of the repo and any directories from the CWD up to the root are included with the developer message "
    "and don't need to be re-read. When working in a subdirectory of CWD, or a directory outside the CWD, check for any AGENTS.md files that may be applicable.\r\n\r\n"
    "## Autonomy and Persistence\r\n"
    "Persist until the task is fully handled end-to-end within the current turn whenever feasible: do not stop at analysis or partial fixes; "
    "carry changes through implementation, verification, and a clear explanation of outcomes unless the user explicitly pauses or redirects you.\r\n\r\n"
    "Unless the user explicitly asks for a plan, asks a question about the code, is brainstorming potential solutions, or some other intent that makes it clear that code should not be written, "
    "assume the user wants you to make code changes or run tools to solve the user's problem. In these cases, it's bad to output your proposed solution in a message, you should go ahead and actually implement the change. "
    "If you encounter challenges or blockers, you should attempt to resolve them yourself.\r\n\r\n"
    "## Responsiveness\r\n\r\n"
    "## Planning\r\n\r\n"
    "You have access to an `update_plan` tool which tracks steps and progress and renders them to the user. Using the tool helps demonstrate that you've understood the task and convey how you're approaching it. "
    "Plans can help to make complex, ambiguous, or multi-phase work clearer and more collaborative for the user. A good plan should break the task into meaningful, logically ordered steps that are easy to verify as you go.\r\n\r\n"
    "Note that plans are not for padding out simple or single-step queries that you can just do or answer immediately.\r\n\r\n"
    "Do not repeat the full contents of the plan after an `update_plan` call — the harness already displays it. Instead, summarize the change made and highlight any important context or next step.\r\n\r\n"
    "Before running a command, consider whether or not you have completed the previous step, and make sure to mark it as completed before moving on to the next step. "
    "It may be the case that you complete all steps in your plan after a single pass of implementation. If this is the case, you can simply mark all the planned steps as completed. "
    "Sometimes, you may need to change plans in the middle of a task: call `update_plan` with the updated plan and make sure to provide an `explanation` of the rationale when doing so.\r\n\r\n"
    "Maintain statuses in the tool: exactly one item in_progress at a time; mark items complete when done; post timely status transitions. "
    "Do not jump an item from pending to completed: always set it to in_progress first. Do not batch-complete multiple items after the fact. "
    "Finish with all items completed or explicitly canceled/deferred before ending the turn. Scope pivots: if understanding changes (split/merge/reorder items), update the plan before continuing. "
    "Do not let the plan go stale while coding.\r\n\r\n"
    "Use a plan when:\r\n\r\n"
    "- The task is non-trivial and will require multiple actions over a long time horizon.\r\n"
    "- There are logical phases or dependencies where sequencing matters.\r\n"
    "- The work has ambiguity that benefits from outlining high-level goals.\r\n"
    "- You want intermediate checkpoints for feedback and validation.\r\n"
    "- When the user asked you to use the plan tool (aka \"TODOs\")\r\n"
    "- You generate additional steps while working, and plan to do them before yielding to the user\r\n\r\n"
    "# Tool Guidelines\r\n\r\n"
    "## Shell commands\r\n\r\n"
    "- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. "
    "(If the `rg` command is not found, then use alternatives.)\r\n"
    "- Do not use python scripts to attempt to output larger chunks of a file.\r\n"
    "- Parallelize tool calls whenever possible - especially file reads, such as `cat`, `rg`, `sed`, `ls`, `git show`, `nl`, `wc`. "
    "Use `multi_tool_use.parallel` to parallelize tool calls and only this.\r\n"
    "\r\n"
    "## apply_patch\r\n\r\n"
    "Use the `apply_patch` tool to edit files. Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. "
    "You can think of it as a high-level envelope:\r\n\r\n"
    "*** Begin Patch\r\n"
    "[ one or more file sections ]\r\n"
    "*** End Patch\r\n\r\n"
    "Within that envelope, you get a sequence of file operations.\r\n"
    "You MUST include a header with your intended action (Add/Delete/Update)\r\n"
    "Each operation starts with one of three headers:\r\n\r\n"
    "*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).\r\n"
    "*** Delete File: <path> - remove an existing file. Nothing follows.\r\n"
    "*** Update File: <path> - patch an existing file in place (optionally with a rename).\r\n\r\n"
    "Example patch:\r\n\r\n"
    "*** Begin Patch\r\n"
    "*** Add File: hello.txt\r\n"
    "+Hello world\r\n"
    "*** Update File: src/app.py\r\n"
    "*** Move to: src/main.py\r\n"
    "@@ def greet():\r\n"
    '-print("Hi")\r\n'
    '+print("Hello, world!")\r\n'
    "*** Delete File: obsolete.txt\r\n"
    "*** End Patch\r\n"
)
DEFAULT_TIMEOUT = httpx.Timeout(30.0)


class DrljResponsesLLM(CustomLLM):
    def _normalize_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_parts: List[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                        text = item.get("text")
                        if text:
                            text_parts.append(str(text))
            elif isinstance(content, str):
                text_parts.append(content)
        if not text_parts:
            return []
        return [
            {"role": "user", "content": [{"type": "input_text", "text": " ".join(text_parts)}]}
        ]

    def _build_payload(self, model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "instructions": INSTRUCTIONS}
        normalized_input = self._normalize_input(messages)
        if normalized_input:
            payload["input"] = normalized_input
        return payload

    def _iter_sse_objects(self, response_iter: Iterator[bytes]) -> Iterator[Dict[str, Any]]:
        for raw_line in response_iter:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data_str = line.removeprefix("data:").strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue

    def _stream_chunks_from_payload(self, payload: Dict[str, Any]) -> Iterator[GenericStreamingChunk]:
        with requests.post(
            DRLJ_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for obj in self._iter_sse_objects(resp.iter_lines()):
                obj_type = obj.get("type")
                if obj_type == "response.error":
                    error_obj = obj.get("error") or {}
                    raise CustomLLMError(
                        status_code=error_obj.get("code", 500),
                        message=error_obj.get("message", "custom provider error"),
                    )
                if obj_type == "response.output_text.delta":
                    delta = obj.get("delta") or ""
                    yield self._build_streaming_chunk(text=delta, is_finished=False)
                if obj_type == "response.output_text.done":
                    final_text = obj.get("text") or ""
                    yield self._build_streaming_chunk(text=final_text, is_finished=True)

    async def _astream_chunks_from_payload(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[GenericStreamingChunk]:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            async with client.stream(
                "POST", DRLJ_URL, headers=DEFAULT_HEADERS, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line.removeprefix("data:").strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    obj_type = obj.get("type")
                    if obj_type == "response.error":
                        error_obj = obj.get("error") or {}
                        raise CustomLLMError(
                            status_code=error_obj.get("code", 500),
                            message=error_obj.get("message", "custom provider error"),
                        )
                    if obj_type == "response.output_text.delta":
                        delta = obj.get("delta") or ""
                        yield self._build_streaming_chunk(text=delta, is_finished=False)
                    if obj_type == "response.output_text.done":
                        final_text = obj.get("text") or ""
                        yield self._build_streaming_chunk(text=final_text, is_finished=True)

    def _collect_text(self, payload: Dict[str, Any]) -> ModelResponse:
        response_id: Optional[str] = None
        deltas: List[str] = []
        final_text: str = ""
        with requests.post(
            DRLJ_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for obj in self._iter_sse_objects(resp.iter_lines()):
                obj_type = obj.get("type")
                if obj_type == "response.created":
                    response_id = obj.get("response", {}).get("id") or response_id
                elif obj_type == "response.error":
                    error_obj = obj.get("error") or {}
                    raise CustomLLMError(
                        status_code=error_obj.get("code", 500),
                        message=error_obj.get("message", "custom provider error"),
                    )
                elif obj_type == "response.output_text.delta":
                    delta = obj.get("delta") or ""
                    deltas.append(delta)
                elif obj_type == "response.output_text.done":
                    final_text = obj.get("text") or ""
        text = final_text or "".join(deltas)
        return ModelResponse(
            id=response_id,
            model=payload.get("model"),
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        custom_prompt_dict: Dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: Dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[httpx.Timeout] = None,
        client=None,
    ) -> ModelResponse:
        payload = self._build_payload(model=model, messages=messages)
        return self._collect_text(payload=payload)

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        custom_prompt_dict: Dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: Dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[httpx.Timeout] = None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        payload = self._build_payload(model=model, messages=messages)
        yield from self._stream_chunks_from_payload(payload=payload)

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        custom_prompt_dict: Dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: Dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[httpx.Timeout] = None,
        client=None,
    ) -> ModelResponse:
        payload = self._build_payload(model=model, messages=messages)
        deltas: List[str] = []
        response_id: Optional[str] = None
        final_text: str = ""
        async with httpx.AsyncClient(timeout=timeout or DEFAULT_TIMEOUT) as httpx_client:
            async with httpx_client.stream(
                "POST", DRLJ_URL, headers=DEFAULT_HEADERS, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line.removeprefix("data:").strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    obj_type = obj.get("type")
                    if obj_type == "response.created":
                        response_id = obj.get("response", {}).get("id") or response_id
                    elif obj_type == "response.error":
                        error_obj = obj.get("error") or {}
                        raise CustomLLMError(
                            status_code=error_obj.get("code", 500),
                            message=error_obj.get("message", "custom provider error"),
                        )
                    elif obj_type == "response.output_text.delta":
                        delta = obj.get("delta") or ""
                        deltas.append(delta)
                    elif obj_type == "response.output_text.done":
                        final_text = obj.get("text") or ""
        text = final_text or "".join(deltas)
        return ModelResponse(
            id=response_id,
            model=model,
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        custom_prompt_dict: Dict[str, Any],
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: Dict[str, Any],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[httpx.Timeout] = None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        payload = self._build_payload(model=model, messages=messages)
        async for chunk in self._astream_chunks_from_payload(payload=payload):
            yield chunk

    def _build_streaming_chunk(self, text: str, is_finished: bool) -> GenericStreamingChunk:
        return {
            "finish_reason": "stop" if is_finished else None,
            "index": 0,
            "is_finished": is_finished,
            "text": text,
            "tool_use": None,
            "usage": None,
        }


drlj_responses_llm = DrljResponsesLLM()
