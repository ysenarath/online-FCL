#!/usr/bin/env bash
# Reproduce Table 1 of "Federated Continual Learning Goes Online" (ICLR 2025)
# CIFAR10, 5 tasks, M in {200, 500, 1000}
#
# Usage:
#   chmod +x reproduce_table1.sh
#   ./reproduce_table1.sh
#
# Each run's console output ends with:
#   Final average accuracy: <A> (+-) <std>
#   Final average forgetting: <F> (+-) <std>
# matching the A(up-arrow) / F(down-arrow) entries in Table 1.
# Results are also written under ./output/FCL/cifar10/... (see utils/train_utils.py:save_results).

set -euo pipefail

COMMON="--dataset_name cifar10 --n_tasks 5 --n_clients 5 --n_runs 3 \
        --batch_size 10 --burnin 30 --jump 5 --fl_update favg"
        # NOTE: paper text explicitly says FedAvg is used for parameter averaging (Sec 4.1),
        # so we override the repo's CLI default (w_favg) with --fl_update favg.

LOG_DIR="./logs_table1"
mkdir -p "$LOG_DIR"

for M in 200 500 1000; do

  # --- ER: plain reservoir sampling, no class balancing ---
  uv run python main_OFCL.py $COMMON --memory_size "$M" \
    --update_strategy reservoir \
    2>&1 | tee "$LOG_DIR/ER_M${M}.log"

  # --- CBR: class-balanced update, random selection within class ---
  uv run python main_OFCL.py $COMMON --memory_size "$M" \
    --update_strategy balanced --balanced_update random \
    2>&1 | tee "$LOG_DIR/CBR_M${M}.log"

  # --- Uncertainty-based, class-balanced update: LC / MS / RC / EN / BI ---
  #     each with bottom-k (most representative) and top-k (most uncertain) sampling
  for SCORE in confidence margin ratio entropy bregman; do
    for STEP in topk bottomk; do
      uv run python main_OFCL.py $COMMON --memory_size "$M" \
        --update_strategy balanced --balanced_update uncertainty \
        --uncertainty_score "$SCORE" --balanced_step "$STEP" \
        2>&1 | tee "$LOG_DIR/${SCORE}_${STEP}_M${M}.log"
    done
  done

done

echo "All Table 1 runs finished. Logs in $LOG_DIR/, raw results in ./output/" 