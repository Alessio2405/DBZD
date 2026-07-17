#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
RUN_ROOT="${RUN_ROOT:-runs}"
ARMS=(baseline_matched multitask dbzd_full dbzd_stopgrad)
SEEDS=(42 43 44)
EXPECTED_REVISION="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["experiment_revision"])' "$CONFIG")"

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$RUN_ROOT/${arm}_s${seed}"
    resolved_config="$run_dir/resolved_config.yaml"
    current_revision=""
    if [[ -f "$resolved_config" ]]; then
      current_revision="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("experiment_revision", ""))' "$resolved_config")"
    fi
    if [[ -f "$run_dir/model_final.pt" && "$current_revision" == "$EXPECTED_REVISION" ]]; then
      echo "Skipping completed ${arm} seed ${seed}"
      continue
    fi
    args=(python train.py --config "$CONFIG" --run-root "$RUN_ROOT" --arm "$arm" --seed "$seed")
    if [[ -f "$run_dir/checkpoint_latest.pt" && "$current_revision" == "$EXPECTED_REVISION" ]]; then
      args+=(--resume)
    elif [[ -n "$current_revision" && "$current_revision" != "$EXPECTED_REVISION" ]]; then
      echo "Re-running stale revision ${current_revision} for ${arm} seed ${seed}"
    fi
    "${args[@]}"
  done
done
