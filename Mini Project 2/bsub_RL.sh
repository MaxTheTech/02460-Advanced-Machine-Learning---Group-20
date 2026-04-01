#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ensemble_cov
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Output_%J.err

module load python3/3.11.9
module load cuda/12.8

source /zhome/06/9/168972/AdvML/02460-Advanced-Machine-Learning---Group-20/Mini\ Project\ 2/.venv/bin/activate
cd /zhome/06/9/168972/AdvML/02460-Advanced-Machine-Learning---Group-20/Mini\ Project\ 2

SCRIPT="ensemble_vaeB.py"
EXP="expB_cov"
MAX_D=3
COMMON="--experiment $EXP --num-reruns 10 --seed 42 --num-curves 25 --num-iter 2000 --device cuda"

# Step 1: train encoder + all decoders once (frozen encoder for decoders 1..MAX_D)
python -u "$SCRIPT" train --num-decoders $MAX_D $COMMON

# Step 2: compute geodesics for each K reusing the same trained models
for D in $(seq 1 $MAX_D); do
    python -u "$SCRIPT" geodesics --num-decoders $D $COMMON
done

# Step 3: aggregate into CoV plot (reads geo_dists_K{1,2,3}.npy from $EXP/)
python -u "$SCRIPT" cov --num-decoders $MAX_D $COMMON