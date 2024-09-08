#!/bin/bash

# testing custom ranges of biased vs unbiased (i.e. like downsampling biased data) - 10k samples, 8k biased (upsampled from 2k) with 2k biased
torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2 --num_training_samples 10000 --num_epochs 3 --is_snli True --is_paft False --inc_premise True

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise True --start_index 0 --num_run 28

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 28

torchrun --nproc_per_node 1 final_model_train_custom_range.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 4 --lr 2e-05 --start_index 0 --num_run 28 --is_halved_bias_set False --lower_samples True --biased_range 2000 --unbiased_range 2000

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 28

git add .
git commit -m "Run 28, SNLI, Upsample = 4x, LR = 2e-05, Bias Model = 3 epochs training with 10k samples"
git push


# testing custom ranges of biased vs unbiased (i.e. like downsampling biased data) - 10k samples, 8k biased (upsampled from 2k) with 2k biased
torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2 --num_training_samples 10000 --num_epochs 3 --is_snli True --is_paft False --inc_premise True

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise True --start_index 0 --num_run 29

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 29

torchrun --nproc_per_node 1 final_model_train_custom_range.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 8 --lr 2e-05 --start_index 0 --num_run 29 --is_halved_bias_set False --lower_samples True --biased_range 2000 --unbiased_range 2000

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 29

git add .
git commit -m "Run 29, SNLI, Upsample = 8x, LR = 2e-05, Bias Model = 3 epochs training with 10k samples"
git push