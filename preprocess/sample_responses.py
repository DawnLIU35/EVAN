import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from common import build_messages, is_correct, load_images, load_json, save_json


def generate(llm, processor, rows, indices, root, sampling_params, batch_size):
    for start in tqdm(range(0, len(indices), batch_size), desc="Sampling"):
        batch = indices[start : start + batch_size]
        requests = [
            {
                "prompt": processor.apply_chat_template(
                    build_messages(rows[i]), add_generation_prompt=True, tokenize=False
                ),
                "multi_modal_data": {"image": load_images(rows[i], root)},
            }
            for i in batch
        ]
        for i, output in zip(batch, llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)):
            for completion in output.outputs:
                rows[i]["responses"].append(completion.text)
                rows[i]["correct"].append(is_correct(completion.text, rows[i]["reward_model"]["ground_truth"]))


def main():
    parser = argparse.ArgumentParser(description="Sample responses with vLLM and verify them against the answer.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--until-correct", action="store_true", help="only resample rows without a correct response")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    rows = load_json(args.input)
    for row in rows:
        row.setdefault("responses", [])
        row.setdefault("correct", [])

    processor = AutoProcessor.from_pretrained(args.model)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(n=args.n, max_tokens=args.max_tokens, temperature=args.temperature)
    root = Path(args.input).parent
    for _ in range(args.max_rounds if args.until_correct else 1):
        pending = [i for i, row in enumerate(rows) if not (args.until_correct and any(row["correct"]))]
        if not pending:
            break
        generate(llm, processor, rows, pending, root, sampling_params, args.batch_size)

    save_json(args.output, rows)
    print(f"{sum(any(row['correct']) for row in rows)} / {len(rows)} rows have a correct response")


if __name__ == "__main__":
    main()
