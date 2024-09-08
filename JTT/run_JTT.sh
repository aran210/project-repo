#!/bin/bash


# echo "Training ID Model"
# torchrun --nproc_per_node 1 train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2

# echo "Identifying Error Set"
# torchrun --nproc_per_node 1 identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2

# echo "Re-Training"
# torchrun --nproc_per_node 1 retrain.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2

# echo "Validation"
# torchrun --nproc_per_node 1 validation.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2 | tee validation_deets.txt

# echo "Testing"
# torchrun --nproc_per_node 1 test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2


# echo "All scripts have been executed. Ready for processing."

# echo "Pushing to GitHub."

# git add .
# git commit -m "JTT Run X Done (SNLI,, 10k Examples, Upsample 8x, ID Epochs = Re-Train Epochs = 3, LR=2e-05, NEW ERROR METHOD)"
# git push
