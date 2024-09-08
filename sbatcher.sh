#!/bin/bash
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=au123


cd /vol/bitbucket/au123/projectenv && source bin/activate && cd llama3

cd ./PoE-ER


#torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 3 --is_snli True --is_paft False --inc_premise False

#torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise False --start_index 0 --num_run 24

#torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 24

#torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 1 --lr 2e-04 --start_index 0 --num_run 24 --is_halved_bias_set False

#torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --is_snli True --is_paft False --num_run 24


#git add .
#git commit -m "Run 24 - SNLI, BASELINE SNLI-FT REPLICATION"
#git push



torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --num_epochs 3 --is_snli True --is_paft False --inc_premise True

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise True --start_index 0 --num_run 16

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 16

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 8 --lr 2e-04 --start_index 0 --num_run 16 --is_halved_bias_set False

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 16

git add .
git commit -m "Run 16, SNLI, Upsample = 8x, LR = 2e-04, Bias Model = 3 epochs training"
git push



#torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --num_epochs 3 --is_snli True --is_paft False --inc_premise True

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise True --start_index 0 --num_run 17

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 17

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 8 --lr 2e-05 --start_index 0 --num_run 17 --is_halved_bias_set False

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 17

git add .
git commit -m "Run 17, SNLI, Upsample = 8x, LR = 2e-05, Bias Model = 3 epochs training"
git push




# torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --num_epochs 3 --is_snli True --is_paft False --inc_premise True

torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_premise True --start_index 0 --num_run 18

torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 18

torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 10000 --is_snli True --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 8 --lr 2e-06 --start_index 0 --num_run 18 --is_halved_bias_set False

torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli True --is_paft False --num_run 18

git add .
git commit -m "Run 18, SNLI, Upsample = 8x, LR = 2e-06, Bias Model = 3 epochs training"
git push




#torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --num_epochs 3 --is_snli False --is_paft False --inc_premise True

#torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --is_snli False --is_paft False --inc_premise True --start_index 0 --num_run 13

#torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli False --is_paft False --num_run 13

#torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --is_snli False --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 8 --lr 2e-04 --start_index 0 --num_run 13 --is_halved_bias_set False

#torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli False --is_paft False --num_run 13

#git add .
#git commit -m "Run 13, MNLI, Upsample = 8x, LR = 2e-04, Bias Model = 3 epochs training, 20k samples"
#git push



# torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --num_epochs 3 --is_snli False --is_paft False --inc_premise True

#torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --is_snli False --is_paft False --inc_premise True --start_index 0 --num_run 14

#torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli False --is_paft False --num_run 14

#torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --is_snli False --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 8 --lr 2e-05 --start_index 0 --num_run 14 --is_halved_bias_set False

#torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli False --is_paft False --num_run 14

#git add .
#git commit -m "Run 14, MNLI, Upsample = 8x, LR = 2e-05, Bias Model = 3 epochs training, 20k samples"
#git push




# torchrun --nproc_per_node 1 bias_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --num_epochs 3 --is_snli False --is_paft False --inc_premise True

#torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --is_snli False --is_paft False --inc_premise True --start_index 0 --num_run 15

#torchrun --nproc_per_node 1 bias_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli False --is_paft False --num_run 15

#torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --num_training_samples 20000 --is_snli False --is_paft False --inc_bias True --num_epochs 3 --upsample_factor 8 --lr 2e-06 --start_index 0 --num_run 15 --is_halved_bias_set False

#torchrun --nproc_per_node 1 final_model_test.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 3 --is_snli False --is_paft False --num_run 15

#git add .
#git commit -m "Run 15, MNLI, Upsample = 8x, LR = 2e-06, Bias Model = 3 epochs training, 20k samples"
#git push
