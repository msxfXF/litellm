"""
运行自定义 provider drlj-gpt-5-2，演示 streaming + completion。
保持 authorization / user-agent / originator / instructions 不变。
"""

import uuid

from litellm.proxy.example_config_yaml.drlj_responses_handler import DrljResponsesLLM


def make_headers() -> dict:
    return {
        "conversation_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "accept": "text/event-stream",
        "authorization": "Bearer cr_1fb51b828e394f883abfb04738a45ccc315b876ba27e45128e3d77625fbf5647",
        "content-type": "application/json",
        "user-agent": "codex_cli_rs/0.77.0 (Windows 10.0.19045; x86_64) unknown",
        "originator": "codex_cli_rs",
    }


def main() -> None:
    llm = DrljResponsesLLM()
    headers = make_headers()
    messages = [{"role": "user", "content": "hello"}]

    print("--- streaming ---")
    for chunk in llm.streaming(
        model="drlj-gpt-5-2",
        messages=messages,
        api_base="",
        custom_prompt_dict={},
        model_response=None,
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params={},
        headers=headers,
    ):
        print(chunk)

    print("--- completion ---")
    resp = llm.completion(
        model="drlj-gpt-5-2",
        messages=messages,
        api_base="",
        custom_prompt_dict={},
        model_response=None,
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params={},
        headers=headers,
    )
    print(resp)


if __name__ == "__main__":
    main()
