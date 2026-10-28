#!/usr/bin/env bash
set -euxo pipefail

python3 -m evan.main \
    data.train_files=${TRAIN_FILE:-data/virl39k/virl39k_evan.json} \
    data.val_files=${VAL_FILE:-data/mmk12/mmk12_test.json} \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct} \
    evan.rollout_n=4 \
    trainer.experiment_name=evan-qwen3-vl-8b \
    "$@"
