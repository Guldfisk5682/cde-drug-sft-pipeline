from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "submit_scorecard",
        "description": "Submit recall/precision/hallucination judgment for one benchmark sample.",
        "parameters": {
            "type": "object",
            "properties": {
                "recall_score": {"type": "number"},
                "precision_score": {"type": "number"},
                "hallucination_score": {"type": "number"},
                "matched_points": {"type": "array", "items": {"type": "string"}},
                "missing_points": {"type": "array", "items": {"type": "string"}},
                "hallucinated_points": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": [
                "recall_score",
                "precision_score",
                "hallucination_score",
                "matched_points",
                "missing_points",
                "hallucinated_points",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


SYSTEM_PROMPT = """你是一个药品说明书问答评测器。

你的任务是根据用户提供的：
1. question
2. gold answer
3. model prediction

严格判断模型回答的 recall、precision、hallucination。

定义如下：
- recall: gold answer 中的关键信息点，被 model prediction 覆盖了多少。近义表达视为命中。
- precision: model prediction 中的关键信息点，有多少可以在 gold answer 中找到依据。近义表达视为有依据。
- hallucination: model prediction 中无法从 gold answer 支持的信息比例。若 model prediction 多说了无依据信息，应提高 hallucination。

评测原则：
- 只能根据 gold answer 和 model prediction 比较，不得使用外部医学常识补全。
- 同义词、近义表达、等价改写视为命中。
- 表达更简洁但不漏关键点，可以给高 recall。
- 若 prediction 没有覆盖 gold 的关键点，应列入 missing_points。
- 若 prediction 额外编造、扩展、猜测了 gold 中没有的内容，应列入 hallucinated_points。
- 分数范围都为 0 到 1。
- 你必须调用 submit_scorecard 工具提交结果，不要输出自然语言正文。
"""


def build_messages(*, question: str, gold_answer: str, prediction: str) -> list[dict[str, Any]]:
    user_content = {
        "question": question,
        "gold_answer": gold_answer,
        "model_prediction": prediction,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False, indent=2)},
    ]


def request_judgment(
    *,
    question: str,
    gold_answer: str,
    prediction: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_messages(question=question, gold_answer=gold_answer, prediction=prediction),
        "tools": [TOOL_SPEC],
        "tool_choice": {"type": "function", "function": {"name": "submit_scorecard"}},
        "temperature": 0.0,
        "stream": False,
    }
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    tool_calls = data["choices"][0]["message"].get("tool_calls", [])
    if not tool_calls:
        raise RuntimeError("Judge model did not return a tool call.")
    arguments = tool_calls[0]["function"]["arguments"]
    parsed = json.loads(arguments)
    return parsed
