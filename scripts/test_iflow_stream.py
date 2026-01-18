from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterator, Optional

import httpx


def iter_sse_data_lines(resp: httpx.Response) -> Iterator[str]:
    for line in resp.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        if not isinstance(line, str):
            continue
        if not line.startswith("data:"):
            continue
        data_str = line.removeprefix("data:").strip()
        if not data_str:
            continue
        yield data_str


def main() -> int:
    parser = argparse.ArgumentParser(description="LiteLLM Proxy streaming smoke test (SSE).")
    parser.add_argument("--base-url", default="http://127.0.0.1:4000", help="Proxy base URL")
    parser.add_argument("--api-key", default="sk-anything", help="Authorization bearer token")
    parser.add_argument("--model", default="iflow-glm", help="Proxy model name")
    parser.add_argument("--message", default="你好，给我一句话介绍 LiteLLM。", help="User message")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout (seconds)")
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
        "Accept": "text/event-stream",
    }
    payload: Dict[str, Any] = {
        "model": args.model,
        "stream": True,
        "messages": [{"role": "user", "content": args.message}],
    }

    final_text: str = ""

    with httpx.Client(timeout=httpx.Timeout(args.timeout)) as client:
        with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                sys.stderr.write(f"HTTP {resp.status_code}: {resp.text}\n")
                return 2

            for data_str in iter_sse_data_lines(resp):
                if data_str == "[DONE]":
                    break
                try:
                    obj: Dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "error" in obj and obj["error"] is not None:
                    sys.stderr.write(json.dumps(obj["error"], ensure_ascii=False) + "\n")
                    return 3

                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue

                choice0 = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
                content = delta.get("content")

                if isinstance(content, str) and content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    final_text += content

                finish_reason: Optional[str] = choice0.get("finish_reason")
                if finish_reason is not None:
                    break

    sys.stdout.write("\n")
    sys.stdout.flush()
    if not final_text:
        sys.stderr.write("No streamed content received.\n")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

