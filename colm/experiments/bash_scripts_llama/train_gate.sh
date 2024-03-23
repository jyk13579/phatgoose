#!/bin/bash

# Default values
SCORE_TYPE=original
SCALING_SCORES=True
ELEMENTWISE_AFFINE=False
MODEL_TYPE=t5xl
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -score_type)
      SCORE_TYPE="$2"
      shift
      ;;
    -scaling_scores)
      SCALING_SCORES="$2"
      shift
      ;;
    -elementwise_affine)
      ELEMENTWISE_AFFINE="$2"
      shift
      ;;
    -exp_name)
      EXP_NAME="$2"
      shift
      ;;
    -old_exp_name)
      OLD_EXP_NAME="$2"
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

if [ -z "$OLD_EXP_NAME" ]; then
  echo "Error: old_exp_name is not set."
  exit 1
fi

echo -e "\nTrain ${DATASET}\n"
echo -e "\nExp name ${EXP_NAME}\n"
echo -e "score_type: ${SCORE_TYPE}, scaling_scores: ${SCALING_SCORES} elementwise_affine: ${ELEMENTWISE_AFFINE}\n"

echo -e "Using LoRA adapter\n"
# Use the variables directly in the command
EXP_NAME=${EXP_NAME} CUDA_VISIBLE_DEVICES=2 python src/launch_single_process.py --gin_files colm/datasets/p3_${MODEL_TYPE}.gin colm/datasets/flanv2_${MODEL_TYPE}.gin colm/models/${MODEL_TYPE}/t5.gin colm/models/${MODEL_TYPE}/moe_lora_rank16.gin colm/experiments/train_single_task.gin colm/experiments/wandb.gin --gin_bindings P/TRAIN/Trainer.datasets=\"D/${DATASET}/TRAIN\" P/EVALUATE/Evaluator.datasets=\"D/${DATASET}/EVAL\" M/MODEL/Router.score_type=\"${SCORE_TYPE}\" M/MODEL/Router.scaling_scores=${SCALING_SCORES} M/MODEL/Router.elementwise_affine=${ELEMENTWISE_AFFINE} 'M/MODEL/Model.trainable_params=".*gate.*"' P/TRAIN/Trainer.num_steps=100 'M/MODEL/FFNExperts.learn_input_gate="only_sigmoid"' 'main.procedure_exec_order=["P/TRAIN"]' 'M/MODEL/Model.init_moma_calls=[@M/MODEL/ENCODER/watch_hiddens, @M/MODEL/DECODER/watch_hiddens, @M/MODEL/ENCODER/make_moe, @M/MODEL/DECODER/make_moe, @M/MODEL/load_weights]' M/MODEL/load_weights.weight_path=\"exp_out/${OLD_EXP_NAME}/best.pt\" ${EXTRA_BINDINGS}
