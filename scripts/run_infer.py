#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
import typer
import yaml
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from _bootstrap import PROJECT_ROOT

app = typer.Typer(add_completion=False, help="Run one-off inference for manual inspection.")

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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


def build_bnb_config(load_in_8bit: bool, load_in_4bit: bool, torch_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if load_in_8bit and load_in_4bit:
        raise ValueError("Only one of load_in_8bit/load_in_4bit can be enabled.")
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    return None


def resolve_device_map(device_map_value: str) -> object:
    if device_map_value.startswith("cuda:"):
        return {"": int(device_map_value.split(":", 1)[1])}
    return device_map_value


def build_prompt(tokenizer, question: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@app.command()
def main(
    model_name_or_path: str = typer.Option(..., help="Base model hub id or local path."),
    prompt: str | None = typer.Option(None, help="Question text for manual inference."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, exists=True, dir_okay=False, readable=True),
    adapter_path: Path | None = typer.Option(None, exists=True, dir_okay=True, readable=True),
    max_new_tokens: int | None = typer.Option(None, min=8),
    temperature: float | None = typer.Option(None, min=0.0),
    top_p: float | None = typer.Option(None, min=0.0, max=1.0),
) -> None:
    config = load_config(config_path)
    benchmark_config = config.get("benchmark", {})

    torch_dtype = choose_torch_dtype(benchmark_config.get("torch_dtype", config.get("torch_dtype", "float16")))
    quantization_config = build_bnb_config(
        benchmark_config.get("load_in_8bit", True),
        benchmark_config.get("load_in_4bit", False),
        torch_dtype,
    )
    device_map = resolve_device_map(benchmark_config.get("device_map", config.get("device_map", "cuda:0")))
    enable_thinking = benchmark_config.get("enable_thinking", config.get("enable_thinking", False))
    max_new_tokens = max_new_tokens or benchmark_config.get("max_new_tokens", 128)
    temperature = temperature if temperature is not None else benchmark_config.get("temperature", 0.0)
    top_p = top_p if top_p is not None else benchmark_config.get("top_p", 1.0)

    if not prompt:
        prompt = typer.prompt("Input question")

    print(f"[infer] loading processor from {model_name_or_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[infer] loading model from {model_name_or_path}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    if adapter_path is not None:
        print(f"[infer] loading adapter from {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    prompt_text = build_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_inputs = {key: value.to(target_device) for key, value in model_inputs.items()}
    input_width = int(model_inputs["input_ids"].shape[1])

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0][input_width:]
    prediction_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    typer.echo("\n[question]")
    typer.echo(prompt)
    typer.echo("\n[answer]")
    typer.echo(prediction_text)


if __name__ == "__main__":
    app()
