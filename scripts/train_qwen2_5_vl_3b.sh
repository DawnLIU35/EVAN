#!/usr/bin/env bash
set -euxo pipefail

python3 -m evan.main \
    data.train_files=${TRAIN_FILE:-data/virl39k/virl39k_evan.json} \
    data.val_files=${VAL_FILE:-data/mmk12/mmk12_test.json} \
    actor_rollout_ref.model.path=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct} \
    evan.rollout_n=2 \
    trainer.experiment_name=evan-qwen2.5-vl-3b \
    "$@"
