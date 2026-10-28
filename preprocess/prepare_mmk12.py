import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from common import save_json


def main():
    parser = argparse.ArgumentParser(description="Download the MMK12 test split used for validation during training.")
    parser.add_argument("--data-dir", default="data/mmk12")
    parser.add_argument("--repo-id", default="PAPOGalaxy/PAPO_MMK12_test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    parquet = hf_hub_download(args.repo_id, "data/train-00000-of-00001.parquet", repo_type="dataset", local_dir=data_dir)

    rows = []
    for index, row in enumerate(pq.read_table(parquet).to_pylist()):
        images = []
        for image_index, image in enumerate(row["images"]):
            image_format = Image.open(io.BytesIO(image["bytes"])).format
            suffix = ".jpg" if image_format == "JPEG" else f".{image_format.lower()}"
            path = f"images/sample-{index:05d}-{image_index}{suffix}"
            (data_dir / path).write_bytes(image["bytes"])
            images.append(path)
        problem = row["problem"].strip()
        missing = len(images) - problem.count("<image>")
        if missing > 0:
            problem = " ".join(["<image>"] * missing) + "\n" + problem
        rows.append(
            {
                "prompt": [{"role": "user", "content": problem}],
                "images": images,
                "reward_model": {"ground_truth": row["answer"].strip()},
                "data_source": "mmk12_test",
                "extra_info": {"row_idx": index},
            }
        )
    save_json(data_dir / "mmk12_test.json", rows)
    print(f"Saved {len(rows)} rows to {data_dir / 'mmk12_test.json'}")


if __name__ == "__main__":
    main()
