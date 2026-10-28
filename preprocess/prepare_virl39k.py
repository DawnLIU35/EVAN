import argparse
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from common import save_json


def unbox(answer: str) -> str:
    text = answer.strip()
    if len(text) >= 2 and text[0] == text[-1] == "$":
        text = text[1:-1].strip()
    prefix = "\\boxed{"
    if not text.startswith(prefix):
        return text
    depth = 1
    for index in range(len(prefix), len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[len(prefix) : index].strip()
    return text


def convert(row: dict) -> dict:
    images = list(row["image"])
    question = row["question"].strip()
    missing = len(images) - question.count("<image>")
    if missing > 0:
        question = " ".join(["<image>"] * missing) + "\n" + question
    return {
        "prompt": [{"role": "user", "content": question}],
        "images": images,
        "reward_model": {"ground_truth": unbox(row["answer"])},
        "data_source": "virl39k",
        "extra_info": {"qid": row["qid"], "category": row["category"], "source": row["source"]},
    }


def main():
    parser = argparse.ArgumentParser(description="Download ViRL39K and convert it to the training format.")
    parser.add_argument("--data-dir", default="data/virl39k")
    parser.add_argument("--repo-id", default="TIGER-Lab/ViRL39K")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet = hf_hub_download(args.repo_id, "39Krelease.parquet", repo_type="dataset", local_dir=data_dir)
    images_zip = hf_hub_download(args.repo_id, "images.zip", repo_type="dataset", local_dir=data_dir)
    with zipfile.ZipFile(images_zip) as archive:
        archive.extractall(data_dir)

    rows = [convert(row) for row in pq.read_table(parquet).to_pylist()]
    save_json(data_dir / "virl39k.json", rows)
    print(f"Saved {len(rows)} rows to {data_dir / 'virl39k.json'}")


if __name__ == "__main__":
    main()
