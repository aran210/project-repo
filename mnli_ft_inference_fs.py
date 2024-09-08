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


def extract_data(example,dataset_name):
    if dataset_name == "snli-hard":
        premise = example["sentence1"]
        hypothesis = example["sentence2"]
    else:
        premise = example["premise"]
        hypothesis = example["hypothesis"]

    return premise, hypothesis


def prepare_demonstrations(dataset_name, dataset):
    normal_mappings = {0:"Entailment",1:"Neutral",2:"Contradiction"}
    hans_mappings = {0:"Entailment",1:"Non-Entailment",2:"Non-Entailment"}
    
    demos = []
    
    # enc
    demos.append(dataset[2])
    demos.append(dataset[0])
    demos.append(dataset[11])


    for demo in demos:
        demo['label'] = normal_mappings[demo['label']]

    return demos



def prepare_sample(sample, dataset_name):

    premise, hypothesis = extract_data(sample, dataset_name)

    system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."
    
    user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
    
    dialog = { "messages":
                  [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}]
             }
    
    return dialog




def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.75,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    
    Inference of MNLI-Finetuned Model on SNLI Test with MNLI Few-Shot Examples
    
    """

    is_prompt_agnostic = False

    cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    if is_prompt_agnostic:
        peft_model_id = "au123/LLaMA3-PROMPT-AGNOSTIC-MNLI-FT"
    else:
        peft_model_id = "au123/LLaMA3-MNLI-FT"

    hf_token = "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"
    
    login(
      token=hf_token, # ADD YOUR TOKEN HERE
      add_to_git_credential=False
    )
    
    dataset_test_flags = {
        "snli": True,
        "snli-hard": True,
        "mnli-mm": True,
        "mnli-m": True,
        "hans": True
    }

    test_set_paths = {
        "snli": ["stanfordnlp/snli", "test"],
        "snli-hard": ["au123/snli-hard", "test"],
        "mnli-mm": ["nyu-mll/multi_nli", "validation_mismatched"],
        "mnli-m": ["nyu-mll/multi_nli", "validation_matched"],
        "hans": ["hans","validation"]
    }

    train_set_paths = {
        "snli": ["nyu-mll/multi_nli", "train"],
        "snli-hard": ["nyu-mll/multi_nli", "train"],
        "mnli-mm": ["nyu-mll/multi_nli", "train"],
        "mnli-m": ["nyu-mll/multi_nli", "train"],
        "hans": ["nyu-mll/multi_nli", "train"]
    }

    to_test = []
    test_sets = {}
    train_sets = {}

    for dataset_name in dataset_test_flags.keys():
        if dataset_test_flags[dataset_name]:
            to_test.append(dataset_name)

    print("> Loading Test Sets...")
    
    for dataset_name in to_test:
        test_sets[dataset_name] = load_dataset(test_set_paths[dataset_name][0],split=test_set_paths[dataset_name][1])

    print("> Loading Train Sets...")
    
    for dataset_name in to_test:
        train_sets[dataset_name] = load_dataset(train_set_paths[dataset_name][0],split=train_set_paths[dataset_name][1])


    print("> Building LLaMA-3-8b-Instruct Model...")

    
    model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id,
            token=hf_token,
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id,cache_dir=cache_dir)


    
    print("> Preparing Data and Generating Responses...")

    max_gen_len = 1
    num_test_examples = -1

    LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    # print(formatted_examples)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token # <|eot_id|>
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE


    for dataset_name in to_test:
        if num_test_examples == -1:
            test_set = test_sets[dataset_name]
        else:
            test_set = test_sets[dataset_name].select(range(num_test_examples))
        train_set = train_sets[dataset_name]


        demonstrations = prepare_demonstrations(dataset_name, train_set)

        if dataset_name == "snli":
            test_set = test_set.filter(lambda example: example["label"] in [0,1,2])


        test_set = test_set.map(lambda sample: prepare_sample(sample,dataset_name), remove_columns=test_set.features,batched=False)


        few_shot_prompt = ""
        for demo in demonstrations:
            ex = f"Premise: {demo['premise']}\nHypothesis: {demo['hypothesis']}\nRelationship: {demo['label']}\n \n"
            few_shot_prompt = few_shot_prompt + ex
       
        def add_few_shot_prompt(sample):
            sample["messages"][1]['content'] = few_shot_prompt + sample["messages"][1]['content']
            return sample

        # Apply the function using the map method
        test_set = test_set.map(add_few_shot_prompt)


        print(test_set[0])

        predictions = []

        for index, ex in enumerate(test_set):
            ex = tokenizer.apply_chat_template(ex["messages"][:2], tokenize=True, add_generation_prompt=True,return_tensors="pt").to(model.device)
            
            output = model.generate(ex,
                          max_new_tokens=max_gen_len,
                          temperature=temperature,
                          top_p=top_p,
                          eos_token_id=tokenizer.eos_token_id,
                          pad_token_id=tokenizer.pad_token_id)
        
            response = tokenizer.decode(output[0])
            # response = output[0]['generated_text'][2]['content']
            answer = response[response.find("assistant<|end_header_id|>\n\n")+len("assistant<|end_header_id|>\n\n"):]
            
            predictions.append({"answer":answer})

            if index % 100 == 0:
                print(index, {"answer": answer})

        # train_set_name = {"snli":"mnli","mnli-mm":"snli","mnli-m":"snli"}
        
        # if is_prompt_agnostic:
        #     preds_file = f"fs-prompt-agnostic-{train_set_name[dataset_name]}-on-{dataset_name}-robustness-9" if dataset_name == "snli-hard" else f"NEW-fs-prompt-agnostic-{train_set_name[dataset_name]}-on-{dataset_name}-robustness-9"
        # else:
        #     preds_file = f"fs-{train_set_name[dataset_name]}-on-{dataset_name}-robustness-9" if dataset_name == "snli-hard" else f"NEW-fs-{train_set_name[dataset_name]}-on-{dataset_name}-robustness-9"


        if is_prompt_agnostic:
            preds_file = f"fs-PAFT-matched-mnli-on-{dataset_name}"
        else:
            preds_file = f"fs-matched-mnli-on-{dataset_name}"
            
        dir_name = "finetuned_mnli_baselines"
        output_path = os.path.join(dir_name,preds_file)
        with open(output_path, 'w') as file:
            for pred in predictions:
                file.write(f"{pred}\n")

    
    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")


