#!/bin/bash

# Redirect all output and errors to a log file
exec > >(tee -a 55_to_58_logfile.log) 2>&1

torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2 --num_training_samples 10000 --num_epochs 2 --is_snli True --is_paft False --inc_premise True

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise True --start_index 0 --num_run 55

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 55

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 55 --upsample_factor 4 --lr 2e-05 --start_index 0 --num_run 3 --is_halved_bias_set False

torchrun --nproc_per_node 1 final_model_validation.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 55

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 55


#  final_model_custom_range for fixed resample
#  final_model_custom_range for gentle resample
#  final_model_custom for balancing
# final_model split files for split epoch training
