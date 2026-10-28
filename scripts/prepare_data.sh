#!/usr/bin/env bash
set -euxo pipefail

DATA_DIR=${DATA_DIR:-data}
SCORER=${SCORER:-Qwen/Qwen3-VL-8B-Instruct}
VIRL=${DATA_DIR}/virl39k

python3 preprocess/prepare_virl39k.py --data-dir ${VIRL}
python3 preprocess/prepare_mmk12.py --data-dir ${DATA_DIR}/mmk12

python3 preprocess/sample_responses.py --model ${SCORER} \
    --input ${VIRL}/virl39k.json --output ${VIRL}/responses.json --n 2 --max-tokens 12288
python3 preprocess/score_siig.py --model ${SCORER} \
    --input ${VIRL}/responses.json --output ${VIRL}/siig.json
python3 preprocess/select_rankmix.py \
    --input ${VIRL}/siig.json --output ${VIRL}/rankmix20.json --percent 20
python3 preprocess/sample_responses.py --model ${SCORER} \
    --input ${VIRL}/rankmix20.json --output ${VIRL}/rankmix20_targets.json \
    --until-correct --n 5 --max-rounds 10 --max-tokens 8192
python3 preprocess/mask_evidence.py --model ${SCORER} \
    --input ${VIRL}/rankmix20_targets.json --base ${VIRL}/virl39k.json \
    --output ${VIRL}/virl39k_ean.json --image-dir ${VIRL}/ablated --mask-ratio 0.5
