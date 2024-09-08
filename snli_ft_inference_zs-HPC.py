from typing import List, Optional

import fire

import os

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


def extract_data(example, dataset_name):
    if dataset_name == "snli-hard":
        premise = example["sentence1"]
        hypothesis = example["sentence2"]
    else:
        premise = example["premise"]
        hypothesis = example["hypothesis"]

    return premise, hypothesis


def prepare_sample(sample,dataset_name):

    premise, hypothesis = extract_data(sample, dataset_name)

    if dataset_name == "hans":
        # system_prompt = "Given the premise is factually true. Therefore, it must be entailment or non-entailment that the hypothesis is inferred from it. Please choose exactly one of 'Entailment' or 'Non-Entailment' in a single word response."
        system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."
    else:
        system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."
    
    user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
    
    dialog = { "messages":
                 # [{"role": "system", "content": system_prompt},
                 [{"role": "user", "content": user_prompt}]
             }
    
    return dialog

def process_batches(generator, batch_size, dialogs, max_gen_len, temperature, top_p):
    results = []
    for i in range(0, len(dialogs), batch_size):
        batch_dialogs = dialogs[i : i + batch_size]
        batch_results = generator.chat_completion(batch_dialogs,
                                        max_gen_len=max_gen_len,
                                        temperature=temperature,
                                        top_p=top_p)
        results.extend(batch_results)
    return results
        



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
    
    run different prompts with zs and save to file
    
    """

    # cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
    # peft_model_id = "au123/LLaMA3-SNLI-FT" # WITH SYSTEM PROMPT
    peft_model_id = "au123/LLaMA3-PROMPT-AGNOSTIC-SNLI-FT" # WITHOUT SYSTEM PROMPT
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

    to_test = []
    test_sets = {}

    for dataset_name in dataset_test_flags.keys():
        if dataset_test_flags[dataset_name]:
            to_test.append(dataset_name)

    print("> Loading Test Sets...")
    
    for dataset_name in to_test:
        test_sets[dataset_name] = load_dataset(test_set_paths[dataset_name][0],split=test_set_paths[dataset_name][1])



    print("> Building LLaMA-3-8b-Instruct Model...")

    
    model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id,
            token=hf_token,
            device_map="auto",
            # cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    
    # tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
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

        if dataset_name == "snli":
            test_set = test_set.filter(lambda example: example["label"] in [0,1,2])

        test_set = test_set.map(lambda sample: prepare_sample(sample, dataset_name), remove_columns=test_set.features,batched=False)

        
        # template dataset
        # def template_dataset(examples):
        #     return{"text":  tokenizer.apply_chat_template(examples["messages"][:2], tokenize=False, add_generation_prompt=True,return_tensors="pt")}
        
        # test_set = test_set.map(template_dataset, remove_columns=["messages"])

        # print(len(test_set))
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
        
        # preds_file = f"zs-{dataset_name}-9" if dataset_name == "snli-hard" else f"NEW-zs-{dataset_name}-9"
        preds_file = f"zs-PAFT-INFERENCE-{dataset_name}"

        dir_name = "finetuned_snli_baselines"
        output_path = os.path.join(dir_name,preds_file)
        with open(output_path, 'w') as file:
            for pred in predictions:
                file.write(f"{pred}\n")

    
    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")


