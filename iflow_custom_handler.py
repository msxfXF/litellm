import json
import os
import random
import secrets
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

import httpx

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse

#
# NOTE: Per user request, these are hard-coded.
# If you later want to move them to env vars, swap these constants out.
#
IFLOW_CLIENT_ID = "10009311001"
IFLOW_CLIENT_SECRET = "4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW"

OAUTH_TOKEN_URL = "https://iflow.cn/oauth/token"
USER_INFO_URL = "https://iflow.cn/api/oauth/getUserInfo"
CHAT_URL = "https://apis.iflow.cn/v1/chat/completions"

TOKEN_INDEX_FILENAME = "tokens.index.json"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        # Be tolerant of UTF-8 BOM (common if edited via PowerShell Set-Content).
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _atomic_write_json(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _http_post_form(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, str]:
    data_bytes = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, body


def _http_get(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, body


def _exchange_refresh_token(refresh_token: str) -> dict:
    import base64

    auth_basic = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}".encode("utf-8")
    auth_header = "Basic " + base64.b64encode(auth_basic).decode("utf-8")
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": IFLOW_CLIENT_ID,
        "client_secret": IFLOW_CLIENT_SECRET,
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": auth_header,
    }
    status, body = _http_post_form(OAUTH_TOKEN_URL, headers, payload)
    if status >= 400:
        raise RuntimeError(f"iFlow refresh failed: HTTP {status} {body}")
    return json.loads(body)


def _get_user_info(access_token: str) -> dict:
    url = USER_INFO_URL + "?" + urllib.parse.urlencode({"accessToken": access_token})
    status, body = _http_get(url)
    if status >= 400:
        raise RuntimeError(f"iFlow getUserInfo failed: HTTP {status} {body}")
    return json.loads(body)


@dataclass
class _Cooldown:
    until_ms: int
    reason: str


class IFlowTokenStore:
    """
    Token storage contract:
      - Reads token index from ./tokens.index.json (proxy working directory)
      - Reads/writes per-account tokens.*.json in the same directory
      - Excludes "default" by design (per user request)
    """

    def __init__(self, token_index_path: Optional[str] = None) -> None:
        # Default to proxy working directory (per user request).
        self.token_index_path = token_index_path or os.path.join(os.getcwd(), TOKEN_INDEX_FILENAME)

    def _load_index(self) -> dict:
        index = _safe_read_json(self.token_index_path) or {}
        if "accounts" not in index or not isinstance(index.get("accounts"), dict):
            index["accounts"] = {}
        return index

    def _save_index(self, index: dict) -> None:
        index["version"] = index.get("version") or 1
        index["updated_at_ms"] = _now_ms()
        _atomic_write_json(self.token_index_path, index)

    def list_accounts(self) -> List[str]:
        index = self._load_index()
        accounts = list((index.get("accounts") or {}).keys())
        # exclude default, empty strings
        accounts = [a for a in accounts if isinstance(a, str) and a.strip() and a != "default"]
        return accounts

    def _token_file_for_account(self, index: dict, account: str) -> str:
        entry = (index.get("accounts") or {}).get(account) or {}
        if isinstance(entry, dict):
            token_file = entry.get("token_file")
            if isinstance(token_file, str) and token_file.strip():
                return token_file
        return f"tokens.{account}.json"

    def load_account_tokens(self, account: str) -> Tuple[str, dict]:
        index = self._load_index()
        token_file = self._token_file_for_account(index, account)
        token_path = os.path.join(os.path.dirname(self.token_index_path), token_file)
        data = _safe_read_json(token_path) or {}
        if not isinstance(data, dict):
            data = {}
        return token_file, data

    def save_account_tokens(self, account: str, token_file: str, tokens: dict) -> None:
        token_path = os.path.join(os.path.dirname(self.token_index_path), token_file)
        _atomic_write_json(token_path, tokens)

        # update index entry (best-effort)
        index = self._load_index()
        accounts = index.setdefault("accounts", {})
        entry = accounts.get(account) if isinstance(accounts, dict) else None
        if not isinstance(entry, dict):
            entry = {}
        entry.update(
            {
                "token_file": token_file,
                "apiKey": tokens.get("apiKey"),
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "expiry_date": tokens.get("expiry_date"),
                "token_type": tokens.get("token_type"),
                "scope": tokens.get("scope"),
            }
        )
        accounts[account] = entry
        self._save_index(index)

    def ensure_fresh_tokens(self, account: str, token_file: str, tokens: dict) -> dict:
        """
        Ensure access_token/apiKey are usable. Refresh if expiry is within 24h.
        Also re-fetch apiKey after refresh in case it rotated.
        """
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        expiry_date = tokens.get("expiry_date")

        if not refresh_token:
            raise RuntimeError(f"Missing refresh_token for account '{account}'. Please login first.")

        if not isinstance(expiry_date, int):
            # force refresh if expiry is missing/invalid
            expiry_date = 0

        if expiry_date - _now_ms() < 24 * 60 * 60 * 1000:
            refreshed = _exchange_refresh_token(refresh_token)
            tokens.update(
                {
                    "access_token": refreshed.get("access_token"),
                    "refresh_token": refreshed.get("refresh_token") or refresh_token,
                    "expiry_date": _now_ms() + int(refreshed.get("expires_in", 0)) * 1000,
                    "token_type": refreshed.get("token_type"),
                    "scope": refreshed.get("scope"),
                }
            )
            access_token = tokens.get("access_token")

            if access_token:
                info = _get_user_info(access_token)
                api_key = info.get("data", {}).get("apiKey") if info.get("success") else None
                if api_key:
                    tokens["apiKey"] = api_key

            self.save_account_tokens(account, token_file, tokens)
        else:
            # ensure apiKey exists (fallback: fetch if missing)
            if not tokens.get("apiKey") and access_token:
                info = _get_user_info(access_token)
                api_key = info.get("data", {}).get("apiKey") if info.get("success") else None
                if api_key:
                    tokens["apiKey"] = api_key
                    self.save_account_tokens(account, token_file, tokens)

        return tokens


class IFlowLLM(CustomLLM):
    """
    LiteLLM Custom Provider: iflow/*

    Client-facing behavior:
      - Client does NOT pass account id
      - Provider randomly selects one account from tokens.index.json (excluding "default")
      - Auto refreshes access_token + updates apiKey (sk-...) as needed
      - Non-streaming only (for now)
    """

    def __init__(self, token_index_path: Optional[str] = None) -> None:
        super().__init__()
        self.token_store = IFlowTokenStore(token_index_path=token_index_path)
        self._cooldowns: Dict[str, _Cooldown] = {}

    def _eligible_accounts(self) -> List[str]:
        accounts = self.token_store.list_accounts()
        now = _now_ms()
        eligible = []
        for a in accounts:
            cd = self._cooldowns.get(a)
            if cd and cd.until_ms > now:
                continue
            eligible.append(a)
        return eligible

    def _mark_cooldown(self, account: str, reason: str, seconds: int = 60) -> None:
        self._cooldowns[account] = _Cooldown(until_ms=_now_ms() + seconds * 1000, reason=reason)

    def _build_iflow_payload(self, model: str, messages: list, optional_params: dict) -> dict:
        # iFlow tends to accept "content parts" style (as seen in their examples).
        # Normalize plain-string message content to [{"type":"text","text": "..."}].
        normalized_messages: List[dict] = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str) and role in ("user", "assistant"):
                nm = dict(m)
                nm["content"] = [{"type": "text", "text": content}]
                normalized_messages.append(nm)
            else:
                normalized_messages.append(m)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
        }

        # Map OpenAI-ish params to iFlow fields (best-effort).
        if "temperature" in optional_params:
            payload["temperature"] = optional_params.get("temperature")
        if "top_p" in optional_params:
            payload["top_p"] = optional_params.get("top_p")

        max_tokens = optional_params.get("max_tokens")
        if max_tokens is not None:
            payload["max_new_tokens"] = max_tokens
        elif optional_params.get("max_new_tokens") is not None:
            payload["max_new_tokens"] = optional_params.get("max_new_tokens")

        # carry through tools if present (OpenAI-compatible)
        tools = optional_params.get("tools")
        tool_choice = optional_params.get("tool_choice")

        # Back-compat: support legacy OpenAI `functions` / `function_call`
        # Convert them to `tools` / `tool_choice` if `tools` were not provided.
        if tools is None and optional_params.get("functions") is not None:
            functions = optional_params.get("functions")
            if isinstance(functions, list):
                tools = [{"type": "function", "function": fn} for fn in functions if isinstance(fn, dict)]

        if tool_choice is None and optional_params.get("function_call") is not None:
            function_call = optional_params.get("function_call")
            if function_call in ("auto", "none"):
                tool_choice = function_call
            elif isinstance(function_call, dict) and function_call.get("name"):
                tool_choice = {
                    "type": "function",
                    "function": {"name": function_call["name"]},
                }

        if tools is not None:
            payload["tools"] = tools
            if tool_choice is None:
                # Some OpenAI-compatible backends require this explicitly.
                tool_choice = "auto"

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # default: disable thinking
        chat_template_kwargs = optional_params.get("chat_template_kwargs")
        if chat_template_kwargs is None:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["chat_template_kwargs"] = chat_template_kwargs

        # pass through anything user put in extra_body
        extra_body = optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        return payload

    def _extract_iflow_error(self, data: Any) -> Optional[Tuple[str, str]]:
        if not isinstance(data, dict):
            return ("invalid_response", "Non-JSON object returned by iFlow")

        # Common iFlow-style error fields observed in the wild:
        # {"status":"435","msg":"Model not support", ...}
        status = data.get("status")
        msg = data.get("msg") or data.get("message")
        if status is not None:
            try:
                status_int = int(status)
            except Exception:
                status_int = None
            if status_int is not None and status_int not in (0, 200):
                return (str(status), str(msg or "iFlow error"))
            if isinstance(status, str) and status not in ("0", "200"):
                return (status, str(msg or "iFlow error"))

        if data.get("success") is False:
            return ("error", str(msg or "iFlow error"))

        if "choices" not in data and ("error" in data or "msg" in data or "message" in data):
            return ("error", str(msg or data.get("error") or "iFlow error"))

        return None

    def _call_iflow_once(self, api_key: str, payload: dict, timeout_s: Optional[float]) -> httpx.Response:
        trace_id = secrets.token_hex(16)
        span_id = secrets.token_hex(8)
        session_id = f"session-{uuid.uuid4()}"
        conversation_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "iFlow-Cli",
            "session-id": session_id,
            "conversation-id": conversation_id,
            "traceparent": f"00-{trace_id}-{span_id}-01",
            "accept-language": "*",
        }
        timeout = httpx.Timeout(timeout_s) if isinstance(timeout_s, (int, float)) else httpx.Timeout(60.0)
        with httpx.Client(timeout=timeout) as client:
            return client.post(CHAT_URL, headers=headers, json=payload)

    async def _acall_iflow_once(self, api_key: str, payload: dict, timeout_s: Optional[float]) -> httpx.Response:
        trace_id = secrets.token_hex(16)
        span_id = secrets.token_hex(8)
        session_id = f"session-{uuid.uuid4()}"
        conversation_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "iFlow-Cli",
            "session-id": session_id,
            "conversation-id": conversation_id,
            "traceparent": f"00-{trace_id}-{span_id}-01",
            "accept-language": "*",
        }
        timeout = httpx.Timeout(timeout_s) if isinstance(timeout_s, (int, float)) else httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(CHAT_URL, headers=headers, json=payload)

    def _parse_model_response(self, resp_json: dict) -> ModelResponse:
        err = self._extract_iflow_error(resp_json)
        if err:
            code, msg = err
            raise RuntimeError(f"iFlow error {code}: {msg}")
        try:
            return ModelResponse(**resp_json)
        except Exception:
            # best-effort fallback
            mr = ModelResponse()
            mr.id = resp_json.get("id") or mr.id
            mr.created = resp_json.get("created") or mr.created
            mr.model = resp_json.get("model") or mr.model
            if isinstance(resp_json.get("choices"), list):
                mr.choices = resp_json["choices"]  # type: ignore
            if resp_json.get("usage") is not None:
                mr.usage = resp_json["usage"]  # type: ignore
            return mr

    def _make_generic_streaming_chunk(
        self,
        text: str,
        *,
        is_finished: bool,
        finish_reason: str = "",
        usage: Optional[dict] = None,
        index: int = 0,
    ) -> GenericStreamingChunk:
        # NOTE: Keep keys limited to GenericStreamingChunk annotations
        return {
            "text": text,
            "tool_use": None,
            "is_finished": is_finished,
            "finish_reason": finish_reason,
            "usage": usage,
            "index": index,
        }

    def _iter_iflow_sse(self, resp: httpx.Response) -> Iterator[dict]:
        for line in resp.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                try:
                    line = line.decode("utf-8", errors="ignore")
                except Exception:
                    continue
            if not isinstance(line, str):
                continue
            if not line.startswith("data:"):
                continue
            data_str = line.removeprefix("data:").strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj

    async def _aiter_iflow_sse(self, resp: httpx.Response) -> AsyncIterator[dict]:
        async for line in resp.aiter_lines():  # type: ignore[attr-defined]
            if not line or not isinstance(line, str):
                continue
            if not line.startswith("data:"):
                continue
            data_str = line.removeprefix("data:").strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj

    def streaming(  # type: ignore[override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        optional_params.pop("stream", None)

        timeout_s: Optional[float] = None
        if isinstance(timeout, (int, float)):
            timeout_s = float(timeout)
        elif isinstance(timeout, httpx.Timeout):
            timeout_s = float(timeout.connect_timeout or 60.0)

        payload = self._build_iflow_payload(model=model, messages=messages, optional_params=optional_params)
        payload["stream"] = True

        accounts = self._eligible_accounts()
        if not accounts:
            raise RuntimeError(
                f"No eligible iFlow accounts found in {self.token_store.token_index_path} "
                "(expected org accounts, excluding 'default')."
            )

        random.shuffle(accounts)
        last_error: Optional[Exception] = None

        for account in accounts:
            try:
                token_file, tokens = self.token_store.load_account_tokens(account)
                tokens = self.token_store.ensure_fresh_tokens(account, token_file, tokens)
                sk = tokens.get("apiKey")
                if not sk:
                    raise RuntimeError(f"Missing apiKey(sk) for account '{account}'.")

                trace_id = secrets.token_hex(16)
                span_id = secrets.token_hex(8)
                session_id = f"session-{uuid.uuid4()}"
                conversation_id = str(uuid.uuid4())
                req_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {sk}",
                    "User-Agent": "iFlow-Cli",
                    "session-id": session_id,
                    "conversation-id": conversation_id,
                    "traceparent": f"00-{trace_id}-{span_id}-01",
                    "accept-language": "*",
                    "accept": "text/event-stream",
                }
                _timeout = httpx.Timeout(timeout_s) if isinstance(timeout_s, (int, float)) else httpx.Timeout(60.0)

                finished = False
                with httpx.Client(timeout=_timeout) as _client:
                    with _client.stream("POST", CHAT_URL, headers=req_headers, json=payload) as resp:
                        if resp.status_code in (401, 403):
                            self._mark_cooldown(account, reason=f"auth_{resp.status_code}", seconds=10)
                            raise RuntimeError(f"iFlow HTTP {resp.status_code}: {resp.text}")
                        if resp.status_code >= 400:
                            self._mark_cooldown(account, reason=f"http_{resp.status_code}", seconds=30)
                            raise RuntimeError(f"iFlow HTTP {resp.status_code}: {resp.text}")

                        for obj in self._iter_iflow_sse(resp):
                            err = self._extract_iflow_error(obj)
                            if err:
                                code, msg = err
                                raise RuntimeError(f"iFlow error {code}: {msg}")

                            choices = obj.get("choices") if isinstance(obj.get("choices"), list) else []
                            if not choices:
                                continue
                            choice0 = choices[0] if isinstance(choices[0], dict) else {}
                            delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
                            text_delta = delta.get("content") or ""
                            tool_calls_delta = delta.get("tool_calls")
                            finish_reason = choice0.get("finish_reason")

                            if text_delta:
                                yield self._make_generic_streaming_chunk(
                                    text_delta, is_finished=False, finish_reason=""
                                )

                            if isinstance(tool_calls_delta, list):
                                for tc in tool_calls_delta:
                                    if isinstance(tc, dict):
                                        yield {
                                            "text": "",
                                            "tool_use": tc,
                                            "is_finished": False,
                                            "finish_reason": "",
                                            "usage": None,
                                            "index": 0,
                                        }
                            if finish_reason is not None and finished is False:
                                finished = True
                                yield self._make_generic_streaming_chunk(
                                    "",
                                    is_finished=True,
                                    finish_reason=str(finish_reason or "stop"),
                                    usage=obj.get("usage") if isinstance(obj.get("usage"), dict) else None,
                                )
                        if finished is False:
                            yield self._make_generic_streaming_chunk(
                                "",
                                is_finished=True,
                                finish_reason="stop",
                                usage=None,
                            )
                return
            except Exception as e:
                last_error = e
                self._mark_cooldown(account, reason="exception", seconds=30)
                continue

        raise RuntimeError(f"All iFlow accounts failed. Last error: {last_error}")

    async def astreaming(  # type: ignore[override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        optional_params.pop("stream", None)

        timeout_s: Optional[float] = None
        if isinstance(timeout, (int, float)):
            timeout_s = float(timeout)
        elif isinstance(timeout, httpx.Timeout):
            timeout_s = float(timeout.connect_timeout or 60.0)

        payload = self._build_iflow_payload(model=model, messages=messages, optional_params=optional_params)
        payload["stream"] = True

        accounts = self._eligible_accounts()
        if not accounts:
            raise RuntimeError(
                f"No eligible iFlow accounts found in {self.token_store.token_index_path} "
                "(expected org accounts, excluding 'default')."
            )

        random.shuffle(accounts)
        last_error: Optional[Exception] = None

        for account in accounts:
            try:
                token_file, tokens = self.token_store.load_account_tokens(account)
                tokens = self.token_store.ensure_fresh_tokens(account, token_file, tokens)
                sk = tokens.get("apiKey")
                if not sk:
                    raise RuntimeError(f"Missing apiKey(sk) for account '{account}'.")

                trace_id = secrets.token_hex(16)
                span_id = secrets.token_hex(8)
                session_id = f"session-{uuid.uuid4()}"
                conversation_id = str(uuid.uuid4())
                req_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {sk}",
                    "User-Agent": "iFlow-Cli",
                    "session-id": session_id,
                    "conversation-id": conversation_id,
                    "traceparent": f"00-{trace_id}-{span_id}-01",
                    "accept-language": "*",
                    "accept": "text/event-stream",
                }
                _timeout = httpx.Timeout(timeout_s) if isinstance(timeout_s, (int, float)) else httpx.Timeout(60.0)

                finished = False
                async with httpx.AsyncClient(timeout=_timeout) as _client:
                    async with _client.stream("POST", CHAT_URL, headers=req_headers, json=payload) as resp:
                        if resp.status_code in (401, 403):
                            self._mark_cooldown(account, reason=f"auth_{resp.status_code}", seconds=10)
                            raise RuntimeError(f"iFlow HTTP {resp.status_code}: {await resp.aread()}")
                        if resp.status_code >= 400:
                            self._mark_cooldown(account, reason=f"http_{resp.status_code}", seconds=30)
                            raise RuntimeError(f"iFlow HTTP {resp.status_code}: {await resp.aread()}")

                        async for obj in self._aiter_iflow_sse(resp):  # type: ignore[arg-type]
                            err = self._extract_iflow_error(obj)
                            if err:
                                code, msg = err
                                raise RuntimeError(f"iFlow error {code}: {msg}")

                            choices = obj.get("choices") if isinstance(obj.get("choices"), list) else []
                            if not choices:
                                continue
                            choice0 = choices[0] if isinstance(choices[0], dict) else {}
                            delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
                            text_delta = delta.get("content") or ""
                            tool_calls_delta = delta.get("tool_calls")
                            finish_reason = choice0.get("finish_reason")

                            if text_delta:
                                yield self._make_generic_streaming_chunk(
                                    text_delta, is_finished=False, finish_reason=""
                                )

                            if isinstance(tool_calls_delta, list):
                                for tc in tool_calls_delta:
                                    if isinstance(tc, dict):
                                        yield {
                                            "text": "",
                                            "tool_use": tc,
                                            "is_finished": False,
                                            "finish_reason": "",
                                            "usage": None,
                                            "index": 0,
                                        }
                            if finish_reason is not None and finished is False:
                                finished = True
                                yield self._make_generic_streaming_chunk(
                                    "",
                                    is_finished=True,
                                    finish_reason=str(finish_reason or "stop"),
                                    usage=obj.get("usage") if isinstance(obj.get("usage"), dict) else None,
                                )
                        if finished is False:
                            yield self._make_generic_streaming_chunk(
                                "",
                                is_finished=True,
                                finish_reason="stop",
                                usage=None,
                            )
                return
            except Exception as e:
                last_error = e
                self._mark_cooldown(account, reason="exception", seconds=30)
                continue

        raise RuntimeError(f"All iFlow accounts failed. Last error: {last_error}")

    def completion(  # type: ignore[override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> ModelResponse:
        optional_params.pop("stream", None)

        timeout_s: Optional[float] = None
        if isinstance(timeout, (int, float)):
            timeout_s = float(timeout)
        elif isinstance(timeout, httpx.Timeout):
            timeout_s = float(timeout.connect_timeout or 60.0)

        payload = self._build_iflow_payload(model=model, messages=messages, optional_params=optional_params)

        accounts = self._eligible_accounts()
        if not accounts:
            raise RuntimeError(
                f"No eligible iFlow accounts found in {self.token_store.token_index_path} "
                "(expected org accounts, excluding 'default')."
            )

        random.shuffle(accounts)
        last_error: Optional[Exception] = None

        for account in accounts:
            try:
                token_file, tokens = self.token_store.load_account_tokens(account)
                tokens = self.token_store.ensure_fresh_tokens(account, token_file, tokens)
                sk = tokens.get("apiKey")
                if not sk:
                    raise RuntimeError(f"Missing apiKey(sk) for account '{account}'.")

                resp = self._call_iflow_once(api_key=sk, payload=payload, timeout_s=timeout_s)
                if resp.status_code in (401, 403):
                    self._mark_cooldown(account, reason=f"auth_{resp.status_code}", seconds=10)
                    token_file, tokens = self.token_store.load_account_tokens(account)
                    tokens["expiry_date"] = 0  # force refresh
                    tokens = self.token_store.ensure_fresh_tokens(account, token_file, tokens)
                    sk2 = tokens.get("apiKey")
                    if not sk2:
                        raise RuntimeError(f"Missing apiKey(sk) after refresh for account '{account}'.")
                    resp = self._call_iflow_once(api_key=sk2, payload=payload, timeout_s=timeout_s)

                if resp.status_code >= 400:
                    self._mark_cooldown(account, reason=f"http_{resp.status_code}", seconds=30)
                    raise RuntimeError(f"iFlow HTTP {resp.status_code}: {resp.text}")

                data = resp.json()
                return self._parse_model_response(data)
            except Exception as e:
                last_error = e
                self._mark_cooldown(account, reason="exception", seconds=30)
                continue

        raise RuntimeError(f"All iFlow accounts failed. Last error: {last_error}")

    async def acompletion(  # type: ignore[override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> ModelResponse:
        optional_params.pop("stream", None)

        timeout_s: Optional[float] = None
        if isinstance(timeout, (int, float)):
            timeout_s = float(timeout)
        elif isinstance(timeout, httpx.Timeout):
            timeout_s = float(timeout.connect_timeout or 60.0)

        payload = self._build_iflow_payload(model=model, messages=messages, optional_params=optional_params)

        accounts = self._eligible_accounts()
        if not accounts:
            raise RuntimeError(
                f"No eligible iFlow accounts found in {self.token_store.token_index_path} "
                "(expected 5 org accounts, excluding 'default')."
            )

        random.shuffle(accounts)
        last_error: Optional[Exception] = None

        for account in accounts:
            try:
                token_file, tokens = self.token_store.load_account_tokens(account)
                tokens = self.token_store.ensure_fresh_tokens(account, token_file, tokens)
                sk = tokens.get("apiKey")
                if not sk:
                    raise RuntimeError(f"Missing apiKey(sk) for account '{account}'.")

                # First attempt
                resp = await self._acall_iflow_once(api_key=sk, payload=payload, timeout_s=timeout_s)
                if resp.status_code in (401, 403):
                    # Force refresh + retry once on auth errors
                    self._mark_cooldown(account, reason=f"auth_{resp.status_code}", seconds=10)
                    token_file, tokens = self.token_store.load_account_tokens(account)
                    tokens["expiry_date"] = 0  # force refresh
                    tokens = self.token_store.ensure_fresh_tokens(account, token_file, tokens)
                    sk2 = tokens.get("apiKey")
                    if not sk2:
                        raise RuntimeError(f"Missing apiKey(sk) after refresh for account '{account}'.")
                    resp = await self._acall_iflow_once(api_key=sk2, payload=payload, timeout_s=timeout_s)

                if resp.status_code >= 400:
                    self._mark_cooldown(account, reason=f"http_{resp.status_code}", seconds=30)
                    raise RuntimeError(f"iFlow HTTP {resp.status_code}: {resp.text}")

                data = resp.json()
                return self._parse_model_response(data)
            except Exception as e:
                last_error = e
                # mark cooldown on hard failures
                self._mark_cooldown(account, reason="exception", seconds=30)
                continue

        raise RuntimeError(f"All iFlow accounts failed. Last error: {last_error}")


# Instance referenced from proxy config:
#   litellm_settings.custom_provider_map:
#     - provider: iflow
#       custom_handler: iflow_custom_handler.iflow_llm
iflow_llm = IFlowLLM()
