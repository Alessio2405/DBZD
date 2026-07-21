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
    summary="$run_dir/summary.json"
    current_revision=""
    completed_summary="false"
    if [[ -f "$resolved_config" ]]; then
      current_revision="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("experiment_revision", ""))' "$resolved_config")"
    elif [[ -f "$summary" ]]; then
      current_revision="$(python -c 'import json,sys; print((json.load(open(sys.argv[1], encoding="utf-8")).get("config") or {}).get("experiment_revision", ""))' "$summary")"
    fi
    if [[ -f "$summary" ]]; then
      completed_summary="$(python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1], encoding="utf-8")).get("completed"))).lower())' "$summary")"
    fi
    if [[ -n "$current_revision" && "$current_revision" != "$EXPECTED_REVISION" ]]; then
      echo "Refusing to overwrite stale revision ${current_revision} in ${run_dir}" >&2
      exit 2
    fi
    if [[ ( -f "$run_dir/model_final.pt" || "$completed_summary" == "true" ) && "$current_revision" == "$EXPECTED_REVISION" ]]; then
      echo "Skipping completed ${arm} seed ${seed}"
      if [[ "$arm" == "dbzd_full" ]]; then
        python scripts/validate_alpha.py --run-dir "$run_dir"
      fi
      continue
    fi
    args=(python train.py --config "$CONFIG" --run-root "$RUN_ROOT" --arm "$arm" --seed "$seed")
    if [[ -f "$run_dir/checkpoint_latest.pt" && "$current_revision" == "$EXPECTED_REVISION" ]]; then
      args+=(--resume)
    fi
    "${args[@]}"
    if [[ "$arm" == "dbzd_full" ]]; then
      python scripts/validate_alpha.py --run-dir "$run_dir"
    fi
  done
done
