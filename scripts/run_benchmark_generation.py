#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
import yaml
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.benchmark_schema import BenchmarkPrediction
from leaflet_pipeline.benchmarking import iter_benchmark_samples, write_jsonl

app = typer.Typer(add_completion=False, help="Run benchmark generation for a base model or SFT adapter.")

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "benchmark" / "predictions.jsonl"
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


def chunked(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@app.command()
def main(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, exists=True, dir_okay=False, readable=True
    ),
    benchmark_path: Path = typer.Option(
        DEFAULT_BENCHMARK_PATH, exists=True, dir_okay=False, readable=True
    ),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_PATH),
    model_name_or_path: str = typer.Option(..., help="Base model path or hub id."),
    adapter_path: Path | None = typer.Option(None, exists=True, dir_okay=True, readable=True),
    torch_dtype: str | None = typer.Option(None),
    device_map: str | None = typer.Option(None),
    load_in_8bit: bool | None = typer.Option(None),
    load_in_4bit: bool | None = typer.Option(None),
    enable_thinking: bool | None = typer.Option(None),
    max_new_tokens: int | None = typer.Option(None, min=8),
    temperature: float | None = typer.Option(None, min=0.0),
    top_p: float | None = typer.Option(None, min=0.0, max=1.0),
    batch_size: int | None = typer.Option(None, min=1),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    config = load_config(config_path)
    benchmark_config = config.get("benchmark", {})
    torch_dtype = torch_dtype or benchmark_config.get("torch_dtype", config.get("torch_dtype", "float16"))
    device_map = device_map or benchmark_config.get("device_map", config.get("device_map", "cuda:0"))
    load_in_8bit = (
        load_in_8bit if load_in_8bit is not None else benchmark_config.get("load_in_8bit", True)
    )
    load_in_4bit = (
        load_in_4bit if load_in_4bit is not None else benchmark_config.get("load_in_4bit", False)
    )
    enable_thinking = (
        enable_thinking
        if enable_thinking is not None
        else benchmark_config.get("enable_thinking", config.get("enable_thinking", False))
    )
    max_new_tokens = max_new_tokens or benchmark_config.get("max_new_tokens", 128)
    temperature = temperature if temperature is not None else benchmark_config.get("temperature", 0.0)
    top_p = top_p if top_p is not None else benchmark_config.get("top_p", 1.0)
    batch_size = batch_size or benchmark_config.get("batch_size", 4)

    dtype = choose_torch_dtype(torch_dtype)
    quantization_config = build_bnb_config(load_in_8bit, load_in_4bit, dtype)
    resolved_device_map = resolve_device_map(device_map)
    print(
        "[benchmark] config summary: "
        f"model={model_name_or_path}, "
        f"adapter={adapter_path}, "
        f"dtype={dtype}, "
        f"load_in_8bit={load_in_8bit}, "
        f"load_in_4bit={load_in_4bit}, "
        f"device_map={resolved_device_map}, "
        f"max_new_tokens={max_new_tokens}, "
        f"temperature={temperature}, "
        f"top_p={top_p}, "
        f"batch_size={batch_size}, "
        f"limit={limit}",
        flush=True,
    )

    print(f"[benchmark] loading processor from {model_name_or_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("[benchmark] processor loaded", flush=True)
    print(f"[benchmark] loading model from {model_name_or_path}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=resolved_device_map,
    )
    print("[benchmark] model loaded", flush=True)

    if adapter_path is not None:
        print(f"[benchmark] loading adapter from {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print("[benchmark] adapter loaded", flush=True)

    model.eval()
    print("[benchmark] model set to eval()", flush=True)

    benchmark_samples = iter_benchmark_samples(benchmark_path)
    if limit is not None:
        benchmark_samples = benchmark_samples[:limit]
    print(f"[benchmark] loaded {len(benchmark_samples)} benchmark samples", flush=True)

    predictions: list[BenchmarkPrediction] = []
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[benchmark] target input device: {target_device}", flush=True)
    print("[benchmark] start generation...", flush=True)

    generated_count = 0
    total_batches = (len(benchmark_samples) + batch_size - 1) // batch_size if benchmark_samples else 0
    for batch_index, sample_batch in enumerate(chunked(benchmark_samples, batch_size), start=1):
        print(
            f"[benchmark] batch {batch_index}/{total_batches} "
            f"size={len(sample_batch)} "
            f"first_task={sample_batch[0].task_type}",
            flush=True,
        )
        prompts = [
            build_prompt(tokenizer, sample.question, enable_thinking=enable_thinking)
            for sample in sample_batch
        ]
        model_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        input_width = int(model_inputs["input_ids"].shape[1])
        model_inputs = {key: value.to(target_device) for key, value in model_inputs.items()}
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
        for row_index, sample in enumerate(sample_batch):
            completion_ids = generated[row_index][input_width:]
            prediction_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            predictions.append(
                BenchmarkPrediction(
                    benchmark_id=sample.benchmark_id,
                    model_name=str(adapter_path or model_name_or_path),
                    question=sample.question,
                    prediction=prediction_text,
                    reference_answer=sample.reference_answer,
                    task_type=sample.task_type,
                )
            )
        generated_count += len(sample_batch)
        print(
            f"[benchmark] finished batch {batch_index}/{total_batches} "
            f"generated={generated_count}/{len(benchmark_samples)}",
            flush=True,
        )

    write_jsonl(output_path, predictions)
    typer.echo(f"Wrote {len(predictions)} benchmark predictions to {output_path}")


if __name__ == "__main__":
    app()
