import argparse
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from common import Scorer, blank_like, load_images, load_json, save_json

scorer = None


def init_worker(model_name, devices):
    global scorer
    device = devices.get()
    torch.cuda.set_device(device)
    scorer = Scorer(model_name, f"cuda:{device}")


@torch.inference_mode()
def score(job):
    row, root = job
    images = load_images(row, root)
    inputs, blank_inputs = scorer.prompt_inputs(row, images), scorer.prompt_inputs(row, blank_like(images))
    logp, blank_logp = [], []
    for response in row["responses"]:
        response_ids = scorer.response_ids(response)
        logp.append(-scorer.nll(inputs, response_ids).item())
        blank_logp.append(-scorer.nll(blank_inputs, response_ids).item())
    return logp, blank_logp


def main():
    parser = argparse.ArgumentParser(description="Compute the scaled image information gain (sIIG) of each question.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    rows = load_json(args.input)
    root = Path(args.input).parent
    ctx = mp.get_context("spawn")
    devices = ctx.Queue()
    for device in range(torch.cuda.device_count()):
        devices.put(device)
    with ctx.Pool(torch.cuda.device_count(), initializer=init_worker, initargs=(args.model, devices)) as pool:
        results = list(tqdm(pool.imap(score, [(row, root) for row in rows]), total=len(rows), desc="Scoring"))

    for row, (logp, blank_logp) in zip(rows, results):
        row["logp"], row["blank_logp"] = logp, blank_logp
        row["siig"] = sum((lp - blp) / (-blp + args.eps) for lp, blp in zip(logp, blank_logp)) / len(logp)
    save_json(args.output, rows)


if __name__ == "__main__":
    main()
