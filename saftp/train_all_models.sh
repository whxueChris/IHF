#!/bin/bash

start_time=$(date +%s)
echo "Starting script at: $(date)"

WINDOW_MIN=${1:-3}
WINDOW_MAX=${2:-50}
VAL_MIN=${3:-15}
VAL_MAX=${4:-15}
TEACHER_FORCING=${5:-0.5}

echo "Using window_size_min=$WINDOW_MIN, window_size_max=$WINDOW_MAX"
echo "Using val_window_size_min=$VAL_MIN, val_window_size_max=$VAL_MAX, teacher_forcing_ratio=$TEACHER_FORCING"

python train_exp_dynamic.py --mode mlp --window_size_min $WINDOW_MIN --window_size_max $WINDOW_MAX --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX
python train_exp_dynamic.py --mode standard --window_size_min $WINDOW_MIN --window_size_max $WINDOW_MAX --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX
python train_exp_dynamic.py --mode autoregressive --teacher_forcing_ratio $TEACHER_FORCING --window_size_min $WINDOW_MIN --window_size_max $WINDOW_MAX --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX
python train_pure_decoder_dynamic.py --mode TDec --teacher_forcing_ratio $TEACHER_FORCING --window_size_min $WINDOW_MIN --window_size_max $WINDOW_MAX --val_window_size_min $VAL_MIN --val_window_size_max $VAL_MAX


end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"