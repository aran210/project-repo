from typing import List, Optional

import fire

import os

from llama import Dialog, Llama, generation
from llama.tokenizer import ChatFormat, Message, Tokenizer

from datasets import load_dataset

from datasets.arrow_dataset import Dataset

import time

import json

from peft import LoraConfig

from transformers import TrainingArguments

from trl import SFTTrainer, setup_chat_format

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

from huggingface_hub import login

def extract_data(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]

    return premise, hypothesis, label


def prepare_sample(sample,idx):

    label_mapping = {0:"Entailment",1:"Neutral",2:"Contradiction"}

    premise, hypothesis, label = extract_data(sample)

    # system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."
    
    user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "

    model_answer = f"{label_mapping[label]}"
    

    # 3 prompts (9, 8, 3)
    if idx % 9 in [1,4]:
        dialog = { "messages":
                  [{"role": "system", "content": "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."},
                  {"role": "user", "content": user_prompt},
                  {"role": "assistant", "content": model_answer}]
             }
    elif idx % 9 == 2:
        dialog = { "messages":
                  [{"role": "system", "content":"Based on the 'Premise' passage, is it entailment or neutral or contradiction that the 'Hypothesis' passage holds? Please respond with exactly one of 'Entailment', 'Neutral' or 'Contradiction' only."},
                  {"role": "user", "content": user_prompt},
                  {"role": "assistant", "content": model_answer}]
             }
    elif idx % 9 == 3:
        dialog = { "messages":
                  [{"role": "system", "content":"You are performing a natural language inference task with perfect accuracy. You will be given a premise and hypothesis, and must classify whether the hypothesis is entailed or contradicted by the premise. If you decide neither, then it is neutral. Respond with 'Entailment', 'Neutral', or 'Contradiction' accordingly."},
                  {"role": "user", "content": user_prompt},
                  {"role": "assistant", "content": model_answer}]
             }
    else:
        dialog = { "messages":
                  [{"role": "user", "content": user_prompt},
                  {"role": "assistant", "content": model_answer}]
             }

    return dialog

def gen_train(chat_formatted_data):
    for formatted_prompt in chat_formatted_data:
        yield {'text': formatted_prompt}


def load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(ckpt_dir=ckpt_dir,
                            tokenizer_path=tokenizer_path,
                            max_seq_len=max_seq_len,
                            max_batch_size=max_batch_size)
    return generator



# torchrun --nproc_per_node 1 mixed_PAFT_finetune.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 2 --is_snli True --num_examples 10000
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    is_snli: bool = True,
    num_examples: int = 10000
):
    """
    
    fine-tuning script
    
    """


    print("> Loading and Preparing Train Set...")

    if is_snli:
        train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=42).select(range(num_examples))
    else:
        train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=42).select(range(num_examples))
    
    valid_samples = train_set.filter(lambda example: example["label"] in [0,1,2])

    X = valid_samples.map(lambda sample,idx: prepare_sample(sample,idx), with_indices=True, remove_columns=valid_samples.features,batched=False)

    print(X[0])
    print(X[1])
    print(X[2])
    
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
        output_dir = "LLaMA3-MIXED-PAFT-SNLI-FT" if is_snli else "LLaMA3-MIXED-PAFT-MNLI-FT",
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=max_batch_size,          # batch size per device during training
        gradient_accumulation_steps=10,          # number of steps before performing a backward/update pass
        gradient_checkpointing=False,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )


    print("> Loading Model...")    

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

    # model, tokenizer = setup_chat_format(model, tokenizer)

    # print(tokenizer.apply_chat_template([dialog["messages"] for dialog in X], tokenize=False, add_generation_prompt=True))
    # print(tokenizer.apply_chat_template(train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False))
    # print(tokenizer.eos_token)

    print(train_dataset[0])
    print(train_dataset[1])
    print(train_dataset[2])
 
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
