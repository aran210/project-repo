#!/bin/bash
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=au123


cd /vol/bitbucket/au123/projectenv && source bin/activate && cd llama3

cd ./PoE-ER

sleep 345500
