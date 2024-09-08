from typing import List, Optional

import fire

import os


from datasets import load_dataset, concatenate_datasets

from datasets.arrow_dataset import Dataset

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




# torchrun --nproc_per_node 1 baseline_train.py --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft True --num_epochs 3 --lr 2e-05 --start_index 0 --num_run 1 --is_shuffle False --seed 42
def main(
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    num_training_samples: int = 10000,
    is_snli: bool = True,
    is_paft: bool = False,
    num_epochs: int = 3,
    lr: float = 2e-4,
    start_index: int = 0,
    num_run: int = 1,
    is_shuffle: bool = False,
    seed: int = 42
):
    """
    
    fine-tuning script
    
    """

    run = num_run

    print("> Loading and Preparing Train Set...")



    if is_paft:
        prefix = 'paft-snli' if is_snli else 'paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'

    ############ SECTION 1: Load Train Set, Upsample Unbiased Set and Create Final Train Set

    if is_shuffle:
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
    else:
        if start_index > 0:
            end_index = start_index + num_training_samples
            if is_snli:
                train_set = load_dataset("stanfordnlp/snli", split="train").select(range(start_index,end_index))
            else:
                train_set = load_dataset("nyu-mll/multi_nli", split="train").select(range(start_index,end_index))
        else:
            if is_snli:
                train_set = load_dataset("stanfordnlp/snli", split="train").select(range(num_training_samples))
            else:
                train_set = load_dataset("nyu-mll/multi_nli", split="train").select(range(num_training_samples))
        
    

    print(f"Orig train set size: {len(train_set)}")
    
    final_train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    X = final_train_set.map(lambda sample,idx: prepare_sample(sample,idx,is_paft),with_indices=True, remove_columns=train_set.features,batched=False)
    y = [example["label"] for example in final_train_set]

    print(f"Prepared final train set size: {len(X)}")
    print(X[0])

    print(f"Prepared final train labels set size: {len(y)}")
    print(y[0])
    



    ############ SECTION 2: Set Fine-Tuning Parameters





    
    
    print("> Setting Up Fine-Tuning...")
 
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=128,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

 
    args = TrainingArguments(
        output_dir = f"LLaMA3-BASELINE-{prefix.upper()}",
        num_train_epochs=num_epochs,                     # number of training epochs
        per_device_train_batch_size=max_batch_size,          # batch size per device during training
        gradient_accumulation_steps=10,          # number of steps before performing a backward/update pass
        gradient_checkpointing=False,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=lr,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )





    ############ SECTION 3: Load Base Model and Format Data for Fine-Tuning




    
    print("> Loading LLaMA-3-8B-Instruct Base Model...")    

    cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_token = ""

    login(
      token=hf_token, # ADD YOUR TOKEN HERE
      add_to_git_credential=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )

    LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token # <|eot_id|>
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    # template dataset
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = X.map(template_dataset, remove_columns=["messages"])


    print(train_dataset[0])
    print(len(train_dataset))





    ############ SECTION 4: FINE-TUNING and Save Model

    

    
 
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    print("> Training...")

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()
    
    # save model
    trainer.save_model()

    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")
