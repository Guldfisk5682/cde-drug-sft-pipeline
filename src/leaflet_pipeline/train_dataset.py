from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - local validation may not have torch installed
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from leaflet_pipeline.pipeline_schema import QASample


class JsonlQADataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = QASample.model_validate_json(line)
                self.samples.append(sample.model_dump(mode="json"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class QwenSFTCollator:
    def __init__(
        self,
        processor: Any,
        *,
        max_length: int = 1024,
        enable_thinking: bool = False,
        label_pad_token_id: int = -100,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.label_pad_token_id = label_pad_token_id

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        signature = inspect.signature(self.processor.apply_chat_template)
        self._supports_enable_thinking = "enable_thinking" in signature.parameters
        if not self._supports_enable_thinking:
            raise RuntimeError(
                "Current processor.apply_chat_template does not support enable_thinking. "
                "Please use a transformers version that supports Qwen3.5 thinking-mode control."
            )

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": message["role"], "content": content})
        return normalized

    def _render_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if torch is None:
            raise RuntimeError("torch is required for QwenSFTCollator")
        full_texts: list[str] = []
        prompt_lengths: list[int] = []

        for feature in features:
            messages = self._normalize_messages(feature["messages"])
            prompt_messages = messages[:-1]
            full_text = self._render_chat(messages, add_generation_prompt=False)
            prompt_text = self._render_chat(prompt_messages, add_generation_prompt=True)

            full_texts.append(full_text)
            prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_lengths.append(len(prompt_token_ids))

        batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.label_pad_token_id
        for row_idx, prompt_length in enumerate(prompt_lengths):
            seq_len = int(batch["attention_mask"][row_idx].sum().item())
            prefix_len = min(prompt_length, seq_len)
            labels[row_idx, :prefix_len] = self.label_pad_token_id
        batch["labels"] = labels
        return batch
