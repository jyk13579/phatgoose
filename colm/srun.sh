#!/bin/bash

#SBATCH --job-name=orpo        # Job name
#SBATCH -o ./slurm/out_%j.txt              # Path to output log file (%j expands to job name)
#SBATCH -e ./slurm/err_%j.err              # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ         # Partition name
#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Request one task (default)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --time=24:00:00            # Time limit
#SBATCH --gres=gpu:1               # Number of GPUs to be allocated


srun EXP_NAME=${EXP_NAME} python3 src/launch_single_process.py --gin_files colm/datasets/p3_${MODEL_TYPE}.gin colm/datasets/flanv2_${MODEL_TYPE}.gin colm/datasets/bigbench.gin colm/models/${MODEL_TYPE}/t5.gin ${ARCH_GIN} colm/experiments/eval.gin --gin_bindings M/MODEL/Router.score_type=\"${SCORE_TYPE}\" M/MODEL/Router.scaling_scores=${SCALING_SCORES} M/MODEL/Router.elementwise_affine=${ELEMENTWISE_AFFINE} ${EXTRA_BINDINGS}