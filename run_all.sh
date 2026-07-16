#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
RUN_ROOT="${RUN_ROOT:-runs}"
ARMS=(baseline_matched multitask dbzd_full dbzd_stopgrad)
SEEDS=(42 43 44)

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ -f "$RUN_ROOT/${arm}_s${seed}/model_final.pt" ]]; then
      echo "Skipping completed ${arm} seed ${seed}"
      continue
    fi
    args=(python train.py --config "$CONFIG" --run-root "$RUN_ROOT" --arm "$arm" --seed "$seed")
    if [[ -f "$RUN_ROOT/${arm}_s${seed}/checkpoint_latest.pt" ]]; then
      args+=(--resume)
    fi
    "${args[@]}"
  done
done
