#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ensemble[1-3]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o logs/Output_%J_%I.out
#BSUB -e logs/Output_%J_%I.err

source /zhome/69/0/168594/Documents/advML/Mini_Project_1/.venv/bin/activate
cd /zhome/69/0/168594/Documents/advML/Mini_Project_1 
SCRIPT="02460-Advanced-Machine-Learning---Group-20/Mini Project 2/ensemble_vaeB.py"
COMMON="--num-reruns 10 --seed 42 --num-curves 30 --num-iter 1500 --device cuda"

D=$LSB_JOBINDEX

python -u "$SCRIPT" train     --experiment expB_d${D} --num-decoders $D $COMMON
python -u "$SCRIPT" geodesics --experiment expB_d${D} --num-decoders $D $COMMON