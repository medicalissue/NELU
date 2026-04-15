#!/bin/bash
#
# Launch the γ-mode ablation as 8 parallel CIFAR-100 MobileNetV2 runs,
# one run per GPU. Each process pins to a single GPU via CUDA_VISIBLE_DEVICES.
#
# 5 variants × seeds = 8 total runs:
#    nelu_pl             seeds 42, 123   (GPU 0, 1)
#    nelu_pc             seeds 42, 123   (GPU 2, 3)
#    nelu_sched          seeds 42, 123   (GPU 4, 5)
#    nelu_schedlearn_pl  seed  42       (GPU 6)
#    nelu_schedlearn_pc  seed  42       (GPU 7)
#
# Usage:
#    bash scripts/run_cifar_gamma_ablation.sh
#    EPOCHS=100 bash scripts/run_cifar_gamma_ablation.sh   # quick run
#    NO_WANDB=1 bash scripts/run_cifar_gamma_ablation.sh   # wandb off
#
# Each run writes results/ablation_gamma_mode/<mode>_s<seed>_e<epochs>.json
# and logs/cifar_gamma_ablation/<tag>.log. After all runs complete, a
# summary table is printed and saved to results/ablation_gamma_mode/summary.json.

set -u  # error on unset vars; tolerate individual run failures

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ─── Isolate from any ambient WANDB env vars ─────────────────────
# If the parent shell had WANDB_RUN_ID / WANDB_RESUME exported (e.g.
# from an earlier resume of a ConvNeXt run), every child process here
# would try to resume that run and pollute its history with CIFAR
# metrics. Explicitly unset these so each run creates its own fresh
# wandb run keyed by the --name we pass via the Python script.
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP 2>/dev/null || true
echo "[isolate] unset WANDB_RUN_ID / WANDB_RESUME / WANDB_NAME / WANDB_RUN_GROUP"
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    echo "[isolate] note: WANDB_PROJECT=$WANDB_PROJECT inherited (ok if intended)"
fi

EPOCHS=${EPOCHS:-200}
LOG_DIR=${LOG_DIR:-logs/cifar_gamma_ablation}
RESULTS_DIR="results/ablation_gamma_mode"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# wandb on/off
WANDB_FLAG="--wandb"
if [[ "${NO_WANDB:-0}" == "1" ]]; then
    WANDB_FLAG=""
fi

# (mode, seed, gpu) triples
runs=(
    "nelu_pl            42   0"
    "nelu_pl            123  1"
    "nelu_pc            42   2"
    "nelu_pc            123  3"
    "nelu_sched         42   4"
    "nelu_sched         123  5"
    "nelu_schedlearn_pl 42   6"
    "nelu_schedlearn_pc 42   7"
)

echo "============================================================"
echo " CIFAR-100 MobileNetV2  γ-mode ablation  (8 parallel runs)"
echo " epochs=$EPOCHS   wandb=${WANDB_FLAG:-off}"
echo " log dir:     $LOG_DIR"
echo " results dir: $RESULTS_DIR"
echo "============================================================"

pids=()
tags=()
gpus=()

for entry in "${runs[@]}"; do
    # shellcheck disable=SC2206
    fields=($entry)
    mode="${fields[0]}"
    seed="${fields[1]}"
    gpu="${fields[2]}"

    tag="${mode}_s${seed}"
    logfile="$LOG_DIR/${tag}.log"
    echo "[launch] GPU $gpu  $mode  seed=$seed  →  $logfile"

    CUDA_VISIBLE_DEVICES=$gpu \
        python experiments/ablation_gamma_mode.py \
            --modes "$mode" --seeds "$seed" \
            --epochs "$EPOCHS" --amp $WANDB_FLAG \
        > "$logfile" 2>&1 &

    pids+=($!)
    tags+=("$tag")
    gpus+=("$gpu")

    # Stagger launches so wandb.init() / data download don't collide.
    sleep 3
done

echo ""
echo "All 8 runs launched.  PIDs: ${pids[*]}"
echo "Tail any log with:  tail -f $LOG_DIR/<tag>.log"
echo ""
echo "Waiting for completion... (this will take ~2-3 hours)"

# Wait for all, track individual success
failed=()
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    tag="${tags[$i]}"
    gpu="${gpus[$i]}"
    if wait "$pid"; then
        echo "[  OK  ] GPU $gpu  $tag"
    else
        echo "[FAIL  ] GPU $gpu  $tag  (see $LOG_DIR/${tag}.log)"
        failed+=("$tag")
    fi
done

echo ""
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All 8 runs completed successfully."
else
    echo "WARNING: ${#failed[@]} run(s) failed: ${failed[*]}"
fi

# Aggregate summary from the individual JSON result files
python - <<PYEOF
import json, statistics as st
from pathlib import Path

out_dir = Path("$RESULTS_DIR")
all_runs = []
for f in sorted(out_dir.glob("*.json")):
    if f.name == "summary.json":
        continue
    try:
        with open(f) as fp:
            all_runs.append(json.load(fp))
    except Exception as e:
        print(f"  [skip] failed to read {f}: {e}")

if not all_runs:
    print("No result JSONs found in $RESULTS_DIR — all runs may have failed.")
    raise SystemExit(1)

by_mode = {}
for r in all_runs:
    by_mode.setdefault(r["mode"], []).append(r)

print()
print("=" * 78)
print(" γ-mode ablation — CIFAR-100 MobileNetV2")
print("=" * 78)
print(f"{'mode':>22s}  {'n':>3s}  {'mean best':>11s}  {'std':>6s}  {'min..max':>14s}  {'avg γ_eff':>10s}")
print("-" * 78)

order = ["nelu_pl", "nelu_pc", "nelu_sched", "nelu_schedlearn_pl", "nelu_schedlearn_pc"]
summary_data = {}
for mode in order:
    if mode not in by_mode:
        continue
    rs = by_mode[mode]
    accs = [r["best_test_acc"] for r in rs]
    m = sum(accs) / len(accs)
    s = st.stdev(accs) if len(accs) > 1 else 0.0
    mn = min(accs)
    mx = max(accs)
    # final γ_eff across runs
    gammas = []
    for r in rs:
        gs = r.get("final_gamma_stats", {})
        gammas.append(gs.get("gamma_eff_mean", gs.get("gamma_mean", float("nan"))))
    g_avg = sum(gammas) / len(gammas) if gammas else float("nan")
    print(f"{mode:>22s}  {len(accs):>3d}  {m:>10.2f}%  {s:>5.2f}  {mn:5.2f}..{mx:5.2f}  {g_avg:>10.4f}")
    summary_data[mode] = {
        "n": len(accs),
        "mean": m,
        "std": s,
        "min": mn,
        "max": mx,
        "seeds": [r["seed"] for r in rs],
        "accs": accs,
        "final_gamma_eff_mean": g_avg,
    }

print("-" * 78)

with open(out_dir / "summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)
print(f"\nSaved: {out_dir}/summary.json")
PYEOF

if [[ ${#failed[@]} -gt 0 ]]; then
    exit 1
fi
