#!/bin/bash
#SBATCH -J canto-train
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH -o /home/%u/musetalk15/exp_out/logs/%x-%j.out
# Resumable MuseTalk training on one 80 GB MIG slice.
#   sbatch --export=ALL,STAGE=stage1 train_canto.sbatch      (or STAGE=stage2, stage3a: uses configs/training/canto_<STAGE>.yaml)
# Saves a full resume state every checkpointing_steps and again ~30 min before the time limit,
# then exits and queues its own successor, which resumes from the newest complete state.
# Stop gracefully: touch exp_out/<stage>/canto_<stage>/STOP   (no successor is queued)
# Disable the chain only: touch exp_out/<stage>/canto_<stage>/NO_CHAIN
set -uo pipefail
STAGE=${STAGE:?set STAGE=stage1, stage2 or stage3a}
cd "$HOME/musetalk15"
OUT="exp_out/$STAGE/canto_$STAGE"; mkdir -p "$OUT" exp_out/logs
[ -f "$OUT/DONE" ] && { echo "$STAGE already finished"; exit 0; }
[ -f "$OUT/STOP" ] && { echo "STOP file present in $OUT; remove it to continue"; exit 0; }
module load slurm cuda11.8/toolkit/11.8.0
if [ "$STAGE" = stage1 ] || [ "$STAGE" = stage3a ]; then   # streaming window must be in place (written by the PC sender)
  W=dataset/HDTF_stream/train_window.txt; N=$(( $(wc -l < "$W" 2>/dev/null || echo 1) - 1 ))
  [ "$N" -ge 1000 ] || { echo "training window has only $N clips (need >= 1000); is the PC sender running?"; exit 1; }
  echo "training window: $N clips"
fi
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRAIN_SAVE_MARGIN_SEC=1800
latest() { ls -d "$OUT"/checkpoint-*/COMPLETE 2>/dev/null | sed 's#.*/checkpoint-\([0-9]*\)/COMPLETE#\1#' | sort -n | tail -1; }
BEFORE=$(latest); BEFORE=${BEFORE:-0}
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
DL=$(date -d "$END" +%s 2>/dev/null || echo 0)
if [ "$DL" -gt $(( $(date +%s) + 3600 )) ]; then export TRAIN_DEADLINE_EPOCH=$DL
else echo "WARNING: could not read a valid job end time ('$END'); no deadline stop"; unset TRAIN_DEADLINE_EPOCH; fi
echo "job $SLURM_JOB_ID stage=$STAGE host=$(hostname) start=$(date -Is) end=$END resume_from=$BEFORE"
nvidia-smi -L
accelerate launch --config_file configs/training/gpu_single.yaml --main_process_port 29502 \
    train.py --config "configs/training/canto_$STAGE.yaml"
rc=$?
AFTER=$(latest); AFTER=${AFTER:-0}
echo "train.py exit=$rc at $(date -Is); newest complete checkpoint $BEFORE -> $AFTER"
if [ $rc -eq 0 ] && [ "$AFTER" -gt "$BEFORE" ] && [ ! -f "$OUT/DONE" ] && [ ! -f "$OUT/STOP" ] && [ ! -f "$OUT/NO_CHAIN" ]; then
  sbatch --export=ALL,STAGE="$STAGE" "$HOME/musetalk15/train_canto.sbatch"
elif [ $rc -ne 0 ] || [ "$AFTER" -le "$BEFORE" ]; then
  echo "Not queuing a successor (failure or no progress). Fix, then resubmit; training resumes from checkpoint-$AFTER."
fi
exit $rc
