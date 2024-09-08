#!/bin/bash

# Redirect all output and errors to a log file
exec > >(tee -a simple_1_to_12.log) 2>&1

torchrun --nproc_per_node 1 simple_classifier_train.py --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias False --num_epochs 3 --upsample_factor 4 --lr 2e-05 --start_index 0 --num_run 1 --is_halved_bias_set False --lower_samples False --biased_range 1000 --unbiased_range 1000

torchrun --nproc_per_node 1 simple_classifier_validation.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 1

torchrun --nproc_per_node 1 simple_classifier_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 1


#  for resampling use train_custom
#  for upsampling challening examples use train_up_chall