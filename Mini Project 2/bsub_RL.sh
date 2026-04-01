#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ensemble[1-3]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o logs/Output_%J_%I.out
#BSUB -e logs/Output_%J_%I.err

module load python3/3.11.9
module load cuda/12.8

source /zhome/06/9/168972/AdvML/02460-Advanced-Machine-Learning---Group-20/Mini\ Project\ 2/.venv/bin/activate
cd /zhome/06/9/168972/AdvML/02460-Advanced-Machine-Learning---Group-20/Mini\ Project\ 2

SCRIPT="ensemble_vaeB.py"
COMMON="--num-reruns 10 --seed 42 --num-curves 25 --num-iter 2000 --device cuda"

D=$LSB_JOBINDEX

python -u "$SCRIPT" train     --experiment expB_d${D} --num-decoders $D $COMMON
python -u "$SCRIPT" geodesics --experiment expB_d${D} --num-decoders $D $COMMON