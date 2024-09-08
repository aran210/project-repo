from typing import List, Optional

import fire

import os


from datasets import load_dataset, concatenate_datasets, load_from_disk

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




# run1 e.g. torchrun --nproc_per_node 1 simple_classifier_train_up_chall.py --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli False --is_paft False --inc_bias False --num_epochs 3 --upsample_factor 4 --lr 2e-05 --start_index 0 --num_run 1 --is_halved_bias_set False --lower_samples False --biased_range 1000 --unbiased_range 1000 --should_balance False --balance_challenging_set False --include_challenging_set False --upsample_challenging_set False --ch_upsample_factor 1
def main(
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    num_training_samples: int = 10000,
    is_snli: bool = False,
    is_paft: bool = False,
    inc_bias: bool = True,
    num_epochs: int = 3,
    upsample_factor: int = 8,
    lr: float = 2e-4,
    start_index: int = 0,
    num_run: int = 1,
    is_halved_bias_set: bool = False,
    lower_samples: bool = False,
    biased_range: int = 1000,
    unbiased_range:int = 9000,
    should_balance:bool = False,
    balance_challenging_set: bool = False,
    include_challenging_set: bool = False,
    upsample_challenging_set: bool = False,
    ch_upsample_factor: int = 1
):
    """
    
    fine-tuning script
    
    """

    seed = -1 # set custom seed when you run
    run = num_run
    balance_classes = should_balance

    print("> Loading and Preparing Train Set...")



    if is_paft:
        prefix = 'paft-snli' if is_snli else 'paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'

    ############ SECTION 1: Load Train Set, Upsample Unbiased Set and Create Final Train Set

    if start_index > 0:
        end_index = start_index + num_training_samples
        if is_snli:
            # train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(start_index,end_index))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/snli_train').shuffle(seed=seed).select(range(start_index,end_index))
        else:
            # train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(start_index,end_index))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/mnli_train').shuffle(seed=seed).select(range(start_index,end_index))
    else:
        if is_snli:
            # train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(num_training_samples))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/snli_train').shuffle(seed=seed).select(range(num_training_samples))
        else:
            # train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(num_training_samples))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/mnli_train').shuffle(seed=seed).select(range(num_training_samples))
        
    

    print(f"Orig train set size: {len(train_set)}")
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    # print(f"Cleaned train set size: {len(train_set)}")

    challenge_set_ids = [int(idx) for idx in extract_file(f"test_results/simple_classifier/challenging_ids_{'snli' if is_snli else 'mnli'}")]

    if num_training_samples == 10000:
        if is_snli:
            unbiased_set_ids = [int(idx) for idx in extract_file(f"test_results/simple_classifier/unbiased_set_ids_10k_snli.txt")]
            biased_set_ids = [int(idx) for idx in extract_file(f"test_results/simple_classifier/biased_set_ids_10k_snli.txt")]            
        else:
            unbiased_set_ids = [int(idx) for idx in extract_file(f"test_results/simple_classifier/unbiased_set_ids_10k.txt")]
            biased_set_ids = [int(idx) for idx in extract_file(f"test_results/simple_classifier/biased_set_ids_10k.txt")]
    else:
        unbiased_set_ids = [int(idx) for idx in extract_file(f"test_results/simple_classifier/unbiased_set_ids_20k.txt")]
        biased_set_ids = [int(idx) for idx in extract_file(f"test_results/simple_classifier/biased_set_ids_20k.txt")]    

    print(f"Unbiased set size: {len(unbiased_set_ids)}")
    print(f"Biased set size: {len(biased_set_ids)}")

    unbiased_set = train_set.select(unbiased_set_ids)
    biased_set = train_set.select(biased_set_ids)

    challenge_set_ids = [idx for idx in challenge_set_ids if idx in unbiased_set_ids]
    challenge_set = train_set.select(challenge_set_ids)

    ch_entail_set = challenge_set.filter(lambda example: example["label"] == 0)
    ch_neutral_set = challenge_set.filter(lambda example: example["label"] == 1)
    ch_contra_set = challenge_set.filter(lambda example: example["label"] == 2)

    print(f"ch_entail set size: {len(ch_entail_set)}")
    print(f"ch_neutral set size: {len(ch_neutral_set)}")
    print(f"ch_contra set size: {len(ch_contra_set)}")
    
    if balance_challenging_set:
        
        max_size = max(len(ch_entail_set),len(ch_contra_set),len(ch_neutral_set))
        
        if len(ch_entail_set) < max_size:
            difference = max_size - len(ch_entail_set)
            if difference <= len(ch_entail_set):
                ch_entail_set = concatenate_datasets([ch_entail_set,ch_entail_set.select(range(difference))])
            else:
                floor_div = max_size // len(ch_entail_set)
                leftover = max_size - (len(ch_entail_set) * floor_div)
                ch_entail_set = concatenate_datasets([ch_entail_set]*floor_div)
                ch_entail_set = concatenate_datasets([ch_entail_set,ch_entail_set.select(range(leftover))])

        if len(ch_contra_set) < max_size:
            difference = max_size - len(ch_contra_set)
            if difference <= len(ch_contra_set):
                ch_contra_set = concatenate_datasets([ch_contra_set,ch_contra_set.select(range(difference))])
            else:
                floor_div = max_size // len(ch_contra_set)
                leftover = max_size - (len(ch_contra_set) * floor_div)
                ch_contra_set = concatenate_datasets([ch_contra_set]*floor_div)
                ch_contra_set = concatenate_datasets([ch_contra_set,ch_contra_set.select(range(leftover))])

        if len(ch_neutral_set) < max_size:
            difference = max_size - len(ch_neutral_set)
            if difference <= len(ch_neutral_set):
                ch_neutral_set = concatenate_datasets([ch_neutral_set,ch_neutral_set.select(range(difference))])
            else:
                floor_div = max_size // len(ch_neutral_set)
                leftover = max_size - (len(ch_neutral_set) * floor_div)
                ch_neutral_set = concatenate_datasets([ch_neutral_set]*floor_div)
                ch_neutral_set = concatenate_datasets([ch_neutral_set,ch_neutral_set.select(range(leftover))])

        challenge_set = concatenate_datasets([ch_entail_set,ch_neutral_set,ch_contra_set]).shuffle(seed=seed)

        ch_entail_set = challenge_set.filter(lambda example: example["label"] == 0)
        ch_neutral_set = challenge_set.filter(lambda example: example["label"] == 1)
        ch_contra_set = challenge_set.filter(lambda example: example["label"] == 2)

        print(f"ch_entail set size: {len(ch_entail_set)}")
        print(f"ch_neutral set size: {len(ch_neutral_set)}")
        print(f"ch_contra set size: {len(ch_contra_set)}")

    if upsample_challenging_set:
        challenge_set = concatenate_datasets([challenge_set]*ch_upsample_factor)

    

    
    print("for debugging")
    print(unbiased_set[100]['premise'], unbiased_set[100]['hypothesis'], unbiased_set[100]['label'])
    print(biased_set[100]['premise'], biased_set[100]['hypothesis'], biased_set[100]['label'])


    print(len(challenge_set))
    if include_challenging_set:
        unbiased_set = concatenate_datasets([unbiased_set,challenge_set])

    
    entail_set = unbiased_set.filter(lambda example: example["label"] == 0)
    neutral_set = unbiased_set.filter(lambda example: example["label"] == 1)
    contra_set = unbiased_set.filter(lambda example: example["label"] == 2)
    

    print(f"entail set size: {len(entail_set)}")
    print(f"neutral set size: {len(neutral_set)}")
    print(f"contra set size: {len(contra_set)}")

    # balance classes in unbiased set
    if balance_classes:
        max_size = max(len(entail_set),len(contra_set),len(neutral_set))
        
        if len(entail_set) < max_size:
            difference = max_size - len(entail_set)
            if difference <= len(entail_set):
                entail_set = concatenate_datasets([entail_set,entail_set.select(range(difference))])
            else:
                floor_div = max_size // len(entail_set)
                leftover = max_size - (len(entail_set) * floor_div)
                entail_set = concatenate_datasets([entail_set]*floor_div)
                entail_set = concatenate_datasets([entail_set,entail_set.select(range(leftover))])

        if len(contra_set) < max_size:
            difference = max_size - len(contra_set)
            if difference <= len(contra_set):
                contra_set = concatenate_datasets([contra_set,contra_set.select(range(difference))])
            else:
                floor_div = max_size // len(contra_set)
                leftover = max_size - (len(contra_set) * floor_div)
                contra_set = concatenate_datasets([contra_set]*floor_div)
                contra_set = concatenate_datasets([contra_set,contra_set.select(range(leftover))])

        if len(neutral_set) < max_size:
            difference = max_size - len(neutral_set)
            if difference <= len(neutral_set):
                neutral_set = concatenate_datasets([neutral_set,neutral_set.select(range(difference))])
            else:
                floor_div = max_size // len(neutral_set)
                leftover = max_size - (len(neutral_set) * floor_div)
                neutral_set = concatenate_datasets([neutral_set]*floor_div)
                neutral_set = concatenate_datasets([neutral_set,neutral_set.select(range(leftover))])

        unbiased_set = concatenate_datasets([entail_set,neutral_set,contra_set]).shuffle(seed=seed)

        entail_set = unbiased_set.filter(lambda example: example["label"] == 0)
        neutral_set = unbiased_set.filter(lambda example: example["label"] == 1)
        contra_set = unbiased_set.filter(lambda example: example["label"] == 2)

    
        print(f"entail set size: {len(entail_set)}")
        print(f"neutral set size: {len(neutral_set)}")
        print(f"contra set size: {len(contra_set)}")

        print(f"NEW UNBIASED SET SIZE: {len(unbiased_set)}")
    

    if lower_samples:
        if len(unbiased_set) < unbiased_range:
            unbiased_range = len(unbiased_set) - 1
        if len(biased_set_ids) < biased_range:
            biased_range = len(biased_set_ids) - 1
            
        unbiased_set = unbiased_set.select(range(unbiased_range))

        print(f"lower samples unbiased length: {len(unbiased_set)}")
        
        biased_set = biased_set.select(range(biased_range))

        print(f"lower samples biased length: {len(biased_set)}")
        
        upsampled_set = [unbiased_set]*upsample_factor
        upsampled_set = concatenate_datasets(upsampled_set)
        final_train_set = concatenate_datasets([upsampled_set, biased_set]).shuffle(seed=seed)
    
    else:
        if upsample_factor == 1:
            final_train_set = concatenate_datasets([unbiased_set, biased_set]).shuffle(seed=seed)
        else:
            upsampled_set = [unbiased_set]*upsample_factor
            upsampled_set = concatenate_datasets(upsampled_set)
            
            print(f"Upsampled unbiased set size: {len(upsampled_set)}")
    
            if inc_bias:
                final_train_set = concatenate_datasets([upsampled_set,biased_set]).shuffle(seed=seed)
    
            
            if is_halved_bias_set:
                halved_biased_set = biased_set.shuffle(seed=seed).select(range(len(biased_set)//2))
                final_train_set = concatenate_datasets([halved_biased_set,upsampled_set,unbiased_set]).shuffle(seed=seed)

    

    # final_train_set = final_train_set.filter(lambda example: example["label"] in [0,1,2])

    print(f"Cleaned final train set size: {len(final_train_set)}")

    # print(f"Final train set size: {len(final_train_set)}")

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
        output_dir = f"SIMPLE-FINAL-MODEL-{prefix.upper()}",
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
    hf_token = "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"

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

