import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from common import Scorer, blank_like, first_correct, load_images, load_json, save_json

scorer = None


def init_worker(model_name, devices):
    global scorer
    device = devices.get()
    torch.cuda.set_device(device)
    scorer = Scorer(model_name, f"cuda:{device}")


def image_stats(image_processor):
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    return mean, std, image_processor.temporal_patch_size, image_processor.patch_size, image_processor.merge_size


def unpack_images(pixel_values, grid_thw, image_processor) -> list[torch.Tensor]:
    mean, std, temporal, patch, merge = image_stats(image_processor)
    images, offset = [], 0
    for t, h, w in grid_thw.tolist():
        tokens = pixel_values[offset : offset + t * h * w].float()
        offset += t * h * w
        frames = tokens.reshape(t, h // merge, w // merge, merge, merge, 3, temporal, patch, patch)
        frames = frames.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * temporal, 3, h * patch, w * patch)
        images.append((frames[0] * std + mean).clamp(0, 1))
    return images


def pack_images(images, grid_thw, image_processor) -> torch.Tensor:
    mean, std, temporal, patch, merge = image_stats(image_processor)
    tokens = []
    for image, (t, h, w) in zip(images, grid_thw.tolist()):
        frames = ((image - mean.to(image.device)) / std.to(image.device)).unsqueeze(0).repeat(t * temporal, 1, 1, 1)
        frames = frames.reshape(t, temporal, 3, h // merge, merge, patch, w // merge, merge, patch)
        tokens.append(frames.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(t * h * w, 3 * temporal * patch * patch))
    return torch.cat(tokens)


def mask_top_patches(image, grad, patch_size, mask_ratio) -> torch.Tensor:
    h, w = image.shape[1] // patch_size, image.shape[2] // patch_size
    scores = grad.abs().sum(dim=0).reshape(h, patch_size, w, patch_size).sum(dim=(1, 3)).flatten()
    num_masked = min(max(1, math.floor(scores.numel() * mask_ratio)), scores.numel())
    keep = torch.ones_like(scores)
    keep[torch.topk(scores, num_masked).indices] = 0
    keep = keep.reshape(h, w).repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    return image * keep


def mask_row(job):
    row, root, image_dir, mask_ratio, eps = job
    target = (first_correct(row) or "").strip()
    if not target:
        return None
    image_processor = scorer.processor.image_processor
    images = load_images(row, root)
    inputs = scorer.prompt_inputs(row, images)
    response_ids = scorer.response_ids(target)
    try:
        with torch.no_grad():
            blank_loss = scorer.nll(scorer.prompt_inputs(row, blank_like(images)), response_ids)
        unpacked = unpack_images(inputs["pixel_values"], inputs["image_grid_thw"], image_processor)
        pixels = [image.to(scorer.device).requires_grad_(True) for image in unpacked]
        packed = pack_images(pixels, inputs["image_grid_thw"], image_processor).to(inputs["pixel_values"].dtype)
        loss = scorer.nll({**inputs, "pixel_values": packed}, response_ids)
        siig = (blank_loss - loss) / (blank_loss + eps)
        grads = torch.autograd.grad(-siig, pixels)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    paths = []
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", row["extra_info"]["qid"])
    for index, (image, grad) in enumerate(zip(pixels, grads)):
        masked = mask_top_patches(image.detach(), grad, image_processor.patch_size, mask_ratio)
        array = (masked.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        path = Path(image_dir) / f"{name}-{index}.png"
        Image.fromarray(array, mode="RGB").save(path)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Build evidence-ablated images with sIIG-gradient patch masking.")
    parser.add_argument("--input", required=True, help="RankMix-selected questions with sampled responses")
    parser.add_argument("--base", required=True, help="full training set to attach the evidence-ablated images to")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    rows = load_json(args.input)
    root = Path(args.input).parent
    Path(args.image_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(row, root, args.image_dir, args.mask_ratio, args.eps) for row in rows]
    ctx = mp.get_context("spawn")
    devices = ctx.Queue()
    for device in range(torch.cuda.device_count()):
        devices.put(device)
    with ctx.Pool(torch.cuda.device_count(), initializer=init_worker, initargs=(args.model, devices)) as pool:
        results = list(tqdm(pool.imap(mask_row, jobs), total=len(jobs), desc="Masking"))

    output_dir = Path(args.output).parent.resolve()
    ablated = {
        row["extra_info"]["qid"]: [os.path.relpath(Path(path).resolve(), output_dir) for path in paths]
        for row, paths in zip(rows, results)
        if paths
    }
    train_rows = load_json(args.base)
    for row in train_rows:
        if row["extra_info"]["qid"] in ablated:
            row["adv_image"] = ablated[row["extra_info"]["qid"]]
    save_json(args.output, train_rows)
    print(f"Built evidence-ablated images for {len(ablated)} / {len(rows)} selected questions")


if __name__ == "__main__":
    main()
