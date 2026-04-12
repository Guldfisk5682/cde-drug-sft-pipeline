#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.benchmark_schema import BenchmarkPrediction
from leaflet_pipeline.benchmarking import iter_benchmark_samples, write_jsonl

app = typer.Typer(add_completion=False, help="Run benchmark generation for a base model or SFT adapter.")

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "benchmark" / "predictions.jsonl"


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
    benchmark_path: Path = typer.Option(
        DEFAULT_BENCHMARK_PATH, exists=True, dir_okay=False, readable=True
    ),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_PATH),
    model_name_or_path: str = typer.Option(..., help="Base model path or hub id."),
    adapter_path: Path | None = typer.Option(None, exists=True, dir_okay=True, readable=True),
    torch_dtype: str = typer.Option("float16"),
    device_map: str = typer.Option("cuda:0"),
    load_in_8bit: bool = typer.Option(True),
    load_in_4bit: bool = typer.Option(False),
    enable_thinking: bool = typer.Option(False),
    max_new_tokens: int = typer.Option(192, min=8),
    temperature: float = typer.Option(0.0, min=0.0),
    top_p: float = typer.Option(1.0, min=0.0, max=1.0),
    limit: int | None = typer.Option(None, min=1),
) -> None:
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
        f"limit={limit}",
        flush=True,
    )

    print(f"[benchmark] loading processor from {model_name_or_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
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
    for index, sample in enumerate(benchmark_samples, start=1):
        print(
            f"[benchmark] sample {index}/{len(benchmark_samples)} "
            f"task={sample.task_type} question={sample.question}",
            flush=True,
        )
        prompt = build_prompt(tokenizer, sample.question, enable_thinking=enable_thinking)
        model_inputs = tokenizer(prompt, return_tensors="pt")
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
        completion_ids = generated[0][model_inputs["input_ids"].shape[1] :]
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
        print(
            f"[benchmark] finished sample {index}/{len(benchmark_samples)} "
            f"prediction_chars={len(prediction_text)}",
            flush=True,
        )

    write_jsonl(output_path, predictions)
    typer.echo(f"Wrote {len(predictions)} benchmark predictions to {output_path}")


if __name__ == "__main__":
    app()
