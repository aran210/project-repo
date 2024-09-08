from typing import List, Optional

import fire

import os

import numpy as np

from scipy.stats import entropy

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


def prepare_sample(sample,idx,is_mixed):

    label_mapping = {0:"Entailment",1:"Neutral",2:"Contradiction"}

    premise, hypothesis, label = extract_data(sample)

    system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."
    
    user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "

    model_answer = f"{label_mapping[label]}"
    
    if is_mixed:
        if idx % 2 == 0:
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


def extract_preds(preds):
    results = []
    for i,pred in enumerate(preds):
        pred = str(pred)
        answer_idx = pred.find("'answer': '")
        answer = pred[answer_idx+len("'answer': '"):-2]
        results.append(answer)

    return results

def get_int_preds(predictions):

    label_mappings = {"Ent":0, "Neutral":1, "Contr":2}

    preds = extract_preds(predictions)

    int_preds = [label_mappings[pred] for pred in preds]

    return int_preds



def main(
    is_snli: bool,  # Add this parameter
    is_mixed_paft: bool,  # Add this parameter
    num_training_samples: int,  # Add this parameter
    selection_method: str,  # Add this parameter
    upsample_factor: int,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None
):
    """
    
    fine-tuning script
    
    """

    print("> Loading and Preparing Train Set...")

    num_epochs = 3
    # num_training_samples = 25000
    # upsample_factor = 8
    lr = 2e-5
    ensemble_runs = 8
    # selection_method = "standard" # standard, error, ambiguous, severity
    entropy_threshold = 0.8
    seed = -1 # set seed

    # is_snli = True
    # is_mixed_paft = True

    if is_mixed_paft:
        prefix = 'mixed-paft-snli' if is_snli else 'mixed-paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'

    ############ SECTION 1: Load Train Set, Upsample Error Set and Create Final Train Set



    if is_snli:
        train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(num_training_samples))
    else:
        train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(num_training_samples))
    

    print(f"Orig train set size: {len(train_set)}")
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    print(f"Cleaned train set size: {len(train_set)}")

    error_set_ids = {}
    for ensemble_run in range(ensemble_runs):
        error_set_ids[f"{ensemble_run}"] = [int(idx) for idx in extract_file(f"error_set/ensemble_{prefix}_error_set_ids_{ensemble_run}")]

    if selection_method == "error":
        print("Error selection method")
        error_sets = []
        for ensemble_run in range(ensemble_runs):
            this_error_set = train_set.select(error_set_ids[f"{ensemble_run}"])
            error_sets.append(this_error_set)
        upsampled_error_set = concatenate_datasets(error_sets)
    elif selection_method == "ambiguous":
        print("Ambiguous selection method")
        int_preds = []
        for ensemble_run in range(ensemble_runs):
            ensemble_preds = extract_file(f"error_set/ensemble_{prefix}_error_set_preds_{ensemble_run}")
            int_preds.append(get_int_preds(ensemble_preds))
        
        # Transpose the list of lists to get columns
        int_preds = np.array(int_preds).T

        # Calculate the frequency distribution and entropy for each sample
        entropies = []
        for row in int_preds:
            counts = np.bincount(row, minlength=3)  # Count the occurrences of each class
            probabilities = counts / len(row)       # Convert counts to probabilities
            entropies.append(entropy(probabilities, base=2))  # Calculate entropy (base 2)

        entropies = np.array(entropies)

        # Get the indexes where entropy is above the threshold
        error_set_ids = np.where(entropies > entropy_threshold)[0]

        # Print the indexes and corresponding entropies
        for ix in error_set_ids:
            print(f"Index: {ix}, Entropy: {entropies[ix]}")

        error_set = train_set.select(error_set_ids)
        upsampled_error_set = [error_set]*(upsample_factor-1)
        upsampled_error_set = concatenate_datasets(upsampled_error_set)
    else: # standard
        print("Standard selection method")
        error_set_ids = list(set().union(*error_set_ids.values())) 
        error_set = train_set.select(error_set_ids)
        upsampled_error_set = [error_set]*(upsample_factor-1)
        upsampled_error_set = concatenate_datasets(upsampled_error_set)


    print(f"Error set size: {len(error_set_ids)}")
    
    print(f"Upsampled error set size: {len(upsampled_error_set)}")

    final_train_set = concatenate_datasets([upsampled_error_set,train_set]).shuffle(seed=seed)

    print(f"Final train set size: {len(final_train_set)}")

    X = final_train_set.map(lambda sample,idx: prepare_sample(sample,idx,is_mixed_paft),with_indices=True, remove_columns=train_set.features,batched=False)
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
        output_dir = f"LLaMA3-JTT-RETRAINED-MODEL-{prefix.upper()}",
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
