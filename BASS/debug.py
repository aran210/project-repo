from typing import List, Optional


import os

from datasets import load_dataset, concatenate_datasets


import time

import json

from peft import LoraConfig

from transformers import TrainingArguments

from trl import SFTTrainer, setup_chat_format

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

from huggingface_hub import login

def extract_file(filename):
    with open(filename, 'r') as file:
        file_contents = file.readlines()
    file_contents = [el.strip() for el in file_contents]
    
    return file_contents

def extract_data(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]

    return premise, hypothesis, label


def prepare_sample(sample,idx,is_paft):

    label_mapping = {0:"Entailment",1:"Neutral",2:"Contradiction"}

    premise, hypothesis, label = extract_data(sample)

    system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."
    
    user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "

    model_answer = f"{label_mapping[label]}"
    
    if is_paft:
        dialog = { "messages":
                    [{"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": model_answer}]
                }
    else:
        dialog = { "messages":
                    [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": model_answer}]
                }
            
    return dialog

def gen_train(chat_formatted_data):
    for formatted_prompt in chat_formatted_data:
        yield {'text': formatted_prompt}




# torchrun --nproc_per_node 1 final_model_train.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft True --inc_bias True --num_epochs 3 --upsample_factor 2 --lr 2e-04 --start_index 0 --num_run 1 --is_halved_bias_set False
def main(
    num_training_samples: int = 10000,
    is_snli: bool = True,
    is_paft: bool = True,
    inc_bias: bool = True,
    upsample_factor: int = 8,
    start_index: int = 0,
    num_run: int = 1,
    is_halved_bias_set: bool = False
):
    """
    
    fine-tuning script
    
    """

    run = num_run
    seed = -1 # need to set seed

    print("> Loading and Preparing Train Set...")



    if is_paft:
        prefix = 'paft-snli' if is_snli else 'paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'

    ############ SECTION 1: Load Train Set, Upsample Unbiased Set and Create Final Train Set

    if start_index > 0:
        end_index = start_index + num_training_samples
        if is_snli:
            train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(start_index,end_index))
        else:
            train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(start_index,end_index))
    else:
        if is_snli:
            train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(num_training_samples))
        else:
            train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(num_training_samples))
        
    

    print(f"Orig train set size: {len(train_set)}")
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    # print(f"Cleaned train set size: {len(train_set)}")

    unbiased_set_ids = [int(idx) for idx in extract_file(f"test_results/run{run}/unbiased_set_ids")]
    biased_set_ids = [int(idx) for idx in extract_file(f"test_results/run{run}/biased_set_ids")]

    print(f"Unbiased set size: {len(unbiased_set_ids)}")
    print(f"Biased set size: {len(biased_set_ids)}")

    unbiased_set = train_set.select(unbiased_set_ids)
    biased_set = train_set.select(biased_set_ids)

    print(biased_set[1111])
    print(biased_set[-1])
    
    upsampled_set = [unbiased_set]*(upsample_factor-1)
    upsampled_set = concatenate_datasets(upsampled_set)
    
    print(f"Upsampled unbiased set size: {len(upsampled_set)}")

    if inc_bias:
        final_train_set = concatenate_datasets([upsampled_set,train_set]).shuffle(seed=seed)
    else:
        final_train_set = concatenate_datasets(upsampled_set, [unbiased_set]).shuffle(seed=seed)

    
    if is_halved_bias_set:
        halved_biased_set = biased_set.shuffle(seed=seed).select(range(len(biased_set)//2))
        final_train_set = concatenate_datasets([halved_biased_set,upsampled_set,unbiased_set]).shuffle(seed=seed)

    
    final_train_set = final_train_set.filter(lambda example: example["label"] in [0,1,2])

    print(f"Cleaned final train set size: {len(final_train_set)}")

    # print(f"Final train set size: {len(final_train_set)}")

    X = final_train_set.map(lambda sample,idx: prepare_sample(sample,idx,is_paft),with_indices=True, remove_columns=train_set.features,batched=False)
    y = [example["label"] for example in final_train_set]

    print(f"Prepared final train set size: {len(X)}")
    print(X[0])

    print(f"Prepared final train labels set size: {len(y)}")
    print(y[0])


    


main(num_training_samples = 10000,
     is_snli = True,
    is_paft = True,
    inc_bias = True,
    upsample_factor = 2,
    start_index = 0,
    num_run = 21,
    is_halved_bias_set = False)