#!/bin/bash
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=au123


cd /vol/bitbucket/au123/projectenv && source bin/activate && cd llama3

cd ./PoE-ER


# SNLI, inc_premise = True/True
torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 1 --is_snli True --is_paft False --inc_premise False

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise False --start_index 0 --num_run 1

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 1

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 2 --lr 2e-05 --start_index 0 --num_run 1 --is_halved_bias_set False

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 1

git add .
git commit -m "Run 1, SNLI"
git push



# SNLI, Upsample Factor = 4, inc_premise = True/True
torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 1 --is_snli True --is_paft False --inc_premise False

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise False --start_index 0 --num_run 2

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 2

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 3 --lr 2e-05 --start_index 0 --num_run 2 --is_halved_bias_set False

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 2

git add .
git commit -m "Run 2, SNLI, Upsample Factor = 3, inc_premise = True/True"
git push



# SNLI, is_paft=False everywhere, Upsample Factor = 2
torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 1 --is_snli True --is_paft False --inc_premise False

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise False --start_index 0 --num_run 3

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 3

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 4 --lr 2e-05 --start_index 0 --num_run 3 --is_halved_bias_set False

torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 3


git add .
git commit -m "Run 3, SNLI, is_paft=False everywhere, Upsample Factor = 4"
git push



# SNLI, is_paft=False everywhere, Upsample Factor = 4, inc_premise = True/True
torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 1 --is_snli True --is_paft False --inc_premise False

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise False --start_index 0 --num_run 4

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 4

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 4 --upsample_factor 4 --lr 2e-05 --start_index 0 --num_run 4 --is_halved_bias_set False

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 4

git add .
git commit -m "Run 4, SNLI, is_paft=False everywhere, Upsample Factor = 4, inc_premise = True/True"
git push





# SNLI, inc_premise = True/True, split training
torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 1 --is_snli True --is_paft False --inc_premise False

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise False --start_index 0 --num_run 5

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 5

torchrun --nproc_per_node 1 final_model_train_split_1.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --num_epochs_biased 1 --num_epochs_unbiased 2 --upsample_factor 2 --lr 2e-05 --start_index 0 --num_run 5

torchrun --nproc_per_node 1 final_model_train_split_2.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --num_epochs_biased 1 --num_epochs_unbiased 2 --upsample_factor 2 --lr 2e-05 --start_index 0 --num_run 5

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 5

git add .
git commit -m "Run 5, SNLI, inc_premise = True/True, split training, num_epochs_biased = 1, num_epochs_unbiased = 2"
git push



# SNLI, is_paft = False everywhere inc_premise = True/True, split training
# torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 1 --is_snli True --is_paft False --inc_premise False

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise False --start_index 0 --num_run 6

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 6

torchrun --nproc_per_node 1 final_model_train_split_1.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --num_epochs_biased 2 --num_epochs_unbiased 1 --upsample_factor 2 --lr 2e-05 --start_index 0 --num_run 6

torchrun --nproc_per_node 1 final_model_train_split_2.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --num_epochs_biased 2 --num_epochs_unbiased 1 --upsample_factor 2 --lr 2e-05 --start_index 0 --num_run 6

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 6

git add .
git commit -m "Run 6, SNLI, is_paft = False everywhere inc_premise = True/True, split training, num_epochs_biased = 2, num_epochs_unbiased = 1"
git push
