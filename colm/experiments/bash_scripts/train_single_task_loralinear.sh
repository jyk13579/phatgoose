#!/bin/bash

# Default values
MODEL_TYPE=t5xl
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -exp_name)
      EXP_NAME="$2"
      shift
      ;;
    -dataset)
      DATASET="$2"
      shift
      ;;
    -model_type)
      MODEL_TYPE="$2"
      shift
      ;;
    -extra_bindings)
      EXTRA_BINDINGS="$2"
      shift
      ;;
    *)
      # Unknown option, ignore
      ;;
  esac

  shift
done


if [ -z "$EXP_NAME" ]; then
  echo "Error: exp_name is not set."
  exit 1
fi

if [ -z "$DATASET" ]; then
  echo "Error: exp_name is not set."
  exit 1
fi

echo -e "\nTrain ${DATASET}\n"

echo -e "Using LoRA adapter\n"

# Use the variables directly in the command
EXP_NAME=${EXP_NAME} CUDA_VISIBLE_DEVICES=2 python src/launch_single_process.py --gin_files colm/datasets/p3_${MODEL_TYPE}.gin colm/datasets/flanv2_${MODEL_TYPE}.gin colm/models/${MODEL_TYPE}/t5.gin colm/models/moe_lora_rank16.gin colm/experiments/train_single_task_loralinear.gin  --gin_bindings P/TRAIN/Trainer.datasets=\"D/${DATASET}/TRAIN\" P/EVALUATE/Evaluator.datasets=\"D/${DATASET}/EVAL\" ${EXTRA_BINDINGS}

# colm/experiments/wandb.gin
#  "D/P3HSWAG/EVAL", "D/P3COPA/EVAL", "D/P3WIC/EVAL", "D/P3WINOGRANDE/EVAL", "D/P3CB/EVAL", "D/P3STORYCLOZE/EVAL", "D/P3ANLI/R1/EVAL", "D/P3ANLI/R2/EVAL", "D/P3ANLI/R3/EVAL", "D/P3WSCFIXED/EVAL"