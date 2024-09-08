from typing import List, Optional

import fire

import os

from llama import Dialog, Llama, generation
from llama.tokenizer import ChatFormat, Message, Tokenizer

from datasets import load_dataset

from datasets.arrow_dataset import Dataset

import time

import json

from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM

from transformers import TrainingArguments

from trl import SFTTrainer, setup_chat_format

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch

from huggingface_hub import login


cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
peft_model_id = "au123/finetuning_llama3_test"
hf_token = "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"

login(
  token=hf_token, # ADD YOUR TOKEN HERE
  add_to_git_credential=False
)

# model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         cache_dir=cache_dir,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         token=hf_token
# )

tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)

model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        token=hf_token,
        device_map="auto",
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )


# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

messages = [
    [{"role": "system", "content": "Finish the given sentence."},
    {"role": "user", "content": "I love "}]
]


prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(prompt)

outputs = pipe(prompt, max_new_tokens=256, temperature=0.6, top_p=0.9, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

response = outputs[0][0]['generated_text']
answer = response[response.find("assistant<|end_header_id|>\n\n")+len("assistant<|end_header_id|>\n\n"):]

# response = outputs['generated_text']
# answer = tokenizer.decode(response, skip_special_tokens=True)
# print(f"Generated Answer:\n{outputs}")


print("\n\nBEFORE FINE-TUNING (BASE MODEL)")
print(f"> SYSTEM PROMPT: Finish the given sentence.\n> USER PROMPT: I love \n> GENERATED OUTPUT: playing with my cat, Luna!\n\n")

print("\n\nAFTER FINE-TUNING (RE-TRAINED MODEL)")
print(f"> SYSTEM PROMPT: Finish the given sentence.\n> USER PROMPT: I love \n> GENERATED OUTPUT: {answer}\n\n")

