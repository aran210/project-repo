#!/bin/bash
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=4:mem=48gb:ngpus=1:gpu_type=RTX6000

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a

cd $HOME/projectenv/ && source bin/activate && cd Llama-Uni-Setup

pip install -r requirements.txt
pip install datasets
pip install peft
pip install transformers
pip install trl
pip install torch
pip install huggingface_hub

torchrun --nproc_per_node 1 mixed_paft_inference_zs.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 16
