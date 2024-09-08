#!/bin/bash

#  example run for ensemble training
# echo "Training ID Model"
# torchrun --nproc_per_node 1 train.py  --is_snli True --is_mixed_paft False --num_training_samples 10000 --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2

# echo "Identifying Error Set"
# torchrun --nproc_per_node 1 ensemble_identification.py --is_snli True --is_mixed_paft False --num_training_samples 10000 --ensemble_run 1 --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2

# echo "Re-Training"
# torchrun --nproc_per_node 1 ensemble_retrain.py --is_snli True --is_mixed_paft False --num_training_samples 10000 --selection_method error --upsample_factor 8 --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2

# echo "Validation"
# torchrun --nproc_per_node 1 validation.py --is_snli True --is_mixed_paft False --num_examples 10000 --upsample_factor 8 --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2 | tee test_results/ensemble/run1/validation_deets.txt

# echo "Testing"
# torchrun --nproc_per_node 1 test.py --is_snli True --is_mixed_paft False --is_ensemble True  --ensemble_run 1 --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2 | tee test_results/ensemble/run1/retrain_deets.txt

# echo "All scripts have been executed. Ready for processing."

# echo "Pushing to GitHub."

# git add .
# git commit -m "Ensemble JTT Run 1 Done (SNLI, 10k Examples, Upsample 8x, ID Epochs = Re-Train Epochs = 3, LR=2e-05, Selection = Error)"
# git push

