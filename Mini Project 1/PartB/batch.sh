#!/bin/sh 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J advML
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 03:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s215225@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
cd /zhome/69/0/168594/Documents/advML/
source Mini_Project_1/.venv/bin/activate

# python Mini_Project_1/mains.py train --network unet --epochs 50 --batch-size 32 --lr 1e-4 --device cuda
# python Mini_Project_1/mains.py sample --network unet --epochs 50 --batch-size 32 --lr 1e-4 --device cuda
# python Mini_Project_1/mains.py train --network VAE --epochs 50 --batch-size 32 --lr 1e-4 --device cuda
# python Mini_Project_1/mains.py sample --network VAE --epochs 50 --batch-size 32 --lr 1e-4 --device cuda
# python Mini_Project_1/mains.py train --network DDPM_VAE --epochs 50 --batch-size 32 --lr 1e-4 --device cuda
# python Mini_Project_1/mains.py sample --network DDPM_VAE --epochs 50 --batch-size 32 --lr 1e-4 --device cuda
python Mini_Project_1/betaAndOther.py
