import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

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




# torchrun --nproc_per_node 1 bias_model_train_TinyBERT.py --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --num_epochs 3 --is_snli True --is_paft True --inc_premise False --lr 2e-05
def main(
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    num_training_samples: int = 10000,
    num_epochs: int = 3,  
    is_snli: bool = True,
    is_paft: bool = True,
    inc_premise: bool = False,
    lr: float = 2e-04
):
    """
    
    fine-tuning script
    
    """

    print("> Loading and Preparing Train Set...")
    

    if is_paft:
        prefix = 'paft-snli' if is_snli else 'paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'



    ############ SECTION 1: Load Train Set, Get Subset


    if is_snli:
        train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=42).select(range(num_training_samples))
    else:
        train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=42).select(range(num_training_samples))

    print(f"Orig train set size: {len(train_set)}")
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    print(f"Cleaned train set size: {len(train_set)}")




    ############ SECTION 2: Set Fine-Tuning Parameters


    
    
    print("> Setting Up Fine-Tuning...")

 
    args = TrainingArguments(
        output_dir = f"TinyBERT-BIAS-MODEL-{prefix.upper()}",
        num_train_epochs=num_epochs,                     # number of training epochs
        per_device_train_batch_size=max_batch_size,          # batch size per device during training
        gradient_accumulation_steps=5,          # number of steps before performing a backward/update pass
        gradient_checkpointing=False,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=lr,                     
        # bf16=True,                              
        # tf32=True,                             
        max_grad_norm=1,                      
        warmup_ratio=0.06,                     
        lr_scheduler_type="linear",           
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )





    # ############ SECTION 3: Load Base Model and Format Data for Fine-Tuning

    print("> Loading TinyBERT Model...") 
    

    cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
    model_id = "huawei-noah/TinyBERT_General_4L_312D"
    hf_token = "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"

    login(
      token=hf_token, # ADD YOUR TOKEN HERE
      add_to_git_credential=False
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               cache_dir=cache_dir,
                                                               device_map="auto",
                                                               token=hf_token,
                                                               num_labels=3)
    
    def tokenize_function(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=512)
    
    # Apply the tokenizer to the training and validation dataset
    train_set = train_set.map(tokenize_function, remove_columns=train_set.features,batched=False)



    # Define a function to tokenize and return length
    def tokenize_and_filter(examples, max_length=512):
        # Tokenize the examples
        tokenized_inputs = tokenizer(examples['premise'], examples['hypothesis'], truncation=False, padding=False)
        
        # Calculate the length of each tokenized input
        lengths = [len(tokens['input_ids']) for tokens in tokenized_inputs]
        
        # Create a mask for inputs that are under or equal to the max length
        mask = [length <= max_length for length in lengths]
        return mask
    
    
    # Apply the tokenization and filtering
    mask = train_set.map(tokenize_and_filter, batched=False)
    
    # Filter the dataset based on the mask
    train_dataset = train_set.select([i for i, keep in enumerate(mask['mask']) if keep])


    print(train_dataset[0])
    print(len(train_dataset))





    # ############ SECTION 4: FINE-TUNING and Save Model

    

    
 
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset
        # dataset_text_field="text",
        # peft_config=peft_config,
        # max_seq_length=max_seq_len,
        # tokenizer=tokenizer,
        # packing=False,
        # dataset_kwargs={
        #     "add_special_tokens": False,  # We template with special tokens
        #     "append_concat_token": False, # No need to add additional separator token
        # }
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





