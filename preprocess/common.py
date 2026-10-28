import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evan.prompt import load_format_prompt
from evan.reward import accuracy_reward, normalize_response

FORMAT_PROMPT = load_format_prompt()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_messages(row: dict) -> list[dict]:
    content = []
    for segment in re.split("(<image>)", FORMAT_PROMPT.render(content=row["prompt"][0]["content"])):
        if segment == "<image>":
            content.append({"type": "image"})
        elif segment:
            content.append({"type": "text", "text": segment})
    return [{"role": "user", "content": content}]


def load_images(row: dict, root) -> list[Image.Image]:
    images = []
    for path in row["images"]:
        with Image.open(Path(root) / path) as image:
            images.append(image.convert("RGB"))
    return images


def blank_like(images: list[Image.Image]) -> list[Image.Image]:
    return [Image.new("RGB", image.size, (0, 0, 0)) for image in images]


def is_correct(response: str, ground_truth: str) -> bool:
    try:
        return accuracy_reward(normalize_response(response), ground_truth) > 0
    except Exception:
        return False


def first_correct(row: dict) -> str | None:
    return next((response for response, correct in zip(row["responses"], row["correct"]) if correct), None)


class Scorer:
    """Teacher-forced response likelihood of a VLM (e.g. Qwen3-VL) under a given image."""

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device)
        self.model.eval().requires_grad_(False)

    def prompt_inputs(self, row: dict, images: list[Image.Image]) -> dict:
        prompt = self.processor.apply_chat_template(build_messages(row), add_generation_prompt=True, tokenize=False)
        return dict(self.processor(text=[prompt], images=images, return_tensors="pt", add_special_tokens=False))

    def response_ids(self, response: str) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        return torch.tensor([ids])

    def nll(self, prompt_inputs: dict, response_ids: torch.Tensor) -> torch.Tensor:
        input_ids = torch.cat([prompt_inputs["input_ids"], response_ids], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[:, -response_ids.shape[1] :] = response_ids
        inputs = {**prompt_inputs, "input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}
        return self.model(**{key: value.to(self.device) for key, value in inputs.items()}).loss
