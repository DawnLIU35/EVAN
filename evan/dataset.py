import copy
import os
import re

from verl.utils.dataset.rl_dataset import RLHFDataset

from .prompt import load_format_prompt


class EVANDataset(RLHFDataset):
    """RLHF dataset that renders the reasoning prompt and attaches evidence-ablated images (``adv_image``)."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        self.template = load_format_prompt(config.get("format_prompt"))
        first_file = data_files if isinstance(data_files, str) else data_files[0]
        self.image_dir = os.path.dirname(os.path.abspath(os.path.expanduser(first_file)))
        super().__init__(data_files, tokenizer, config, processor=processor, max_samples=max_samples)

    def maybe_filter_out_long_prompts(self, dataframe=None):
        dataframe = dataframe.map(self._format_row, desc="Formatting prompts")
        return super().maybe_filter_out_long_prompts(dataframe)

    def _format_row(self, row: dict) -> dict:
        prompt = copy.deepcopy(row[self.prompt_key])
        prompt[0]["content"] = self.template.render(content=prompt[0]["content"])
        row[self.prompt_key] = prompt
        for key in (self.image_key, "adv_image"):
            if row.get(key):
                row[key] = [os.path.join(self.image_dir, path) for path in row[key]]
        return row

    def _build_messages(self, example: dict) -> list:
        messages = copy.deepcopy(example[self.prompt_key])
        images = list(example.get(self.image_key) or [])
        num_used = 0
        for message in messages:
            if not images or not isinstance(message["content"], str):
                continue
            content = []
            for segment in re.split("(<image>)", message["content"]):
                if segment == "<image>":
                    content.append({"type": "image", "image": images[num_used]})
                    num_used += 1
                elif segment:
                    content.append({"type": "text", "text": segment})
            message["content"] = content
        assert num_used == len(images), f"{len(images)} images but {num_used} <image> placeholders"
        return messages

    def __getitem__(self, item):
        row = super().__getitem__(item)
        adv_images = row.get("adv_image")
        row["adv_raw_prompt"] = None
        if adv_images and all(os.path.exists(path) for path in adv_images):
            row["adv_raw_prompt"] = self._build_messages({self.prompt_key: row[self.prompt_key], self.image_key: adv_images})
        return row
