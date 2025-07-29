#!/bin/bash

VAL_MIN=${1:-3}
VAL_MAX=${2:-40}
TEACHER_FORCING=${3:-0.5}

start_time=$(date +%s)
echo "Starting evaluation script at: $(date)"
echo "Using val_window_size_min=$VAL_MIN, val_window_size_max=$VAL_MAX, teacher_forcing_ratio=$TEACHER_FORCING"

python train_exp_dynamic.py --mode mlp --evaluate_only --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX
python train_exp_dynamic.py --mode standard --evaluate_only --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX
python train_exp_dynamic.py --mode autoregressive --evaluate_only --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX --teacher_forcing_ratio $TEACHER_FORCING
python train_pure_decoder_dynamic.py --mode TDec --evaluate_only --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX --teacher_forcing_ratio $TEACHER_FORCING

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Evaluation script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"