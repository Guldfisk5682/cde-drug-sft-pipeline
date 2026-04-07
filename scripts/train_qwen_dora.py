#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.train_dataset import JsonlQADataset, QwenSFTCollator

app = typer.Typer(add_completion=False, help="Train Qwen3.5 with DoRA on leaflet QA JSONL.")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def choose_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def build_bnb_config(config: dict[str, Any], torch_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if not config.get("load_in_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch_dtype,
    )


@app.command()
def main(
    config_path: Path = typer.Option(
        PROJECT_ROOT / "configs" / "train.yaml",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Training YAML config path.",
    ),
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config = load_config(config_path)

    torch_dtype = choose_torch_dtype(config.get("torch_dtype", "bfloat16"))
    quantization_config = build_bnb_config(config, torch_dtype)
    device_map = config.get("device_map", "auto")

    processor = AutoProcessor.from_pretrained(config["base_model"], trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        config["base_model"],
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
        )

    peft_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type=TaskType.CAUSAL_LM,
        use_dora=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = JsonlQADataset(resolve_path(config["train_dataset_path"]))
    eval_dataset_path = config.get("eval_dataset_path")
    eval_dataset = JsonlQADataset(resolve_path(eval_dataset_path)) if eval_dataset_path else None

    collator = QwenSFTCollator(
        processor,
        max_length=config.get("max_seq_length", 1024),
        enable_thinking=bool(config.get("enable_thinking", False)),
    )

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        optim=config.get("optim", "paged_adamw_8bit"),
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 200),
        eval_steps=config.get("eval_steps", 200),
        save_total_limit=config.get("save_total_limit", 2),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        remove_unused_columns=False,
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        report_to=config.get("report_to", "none"),
        save_strategy="steps",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    app()
