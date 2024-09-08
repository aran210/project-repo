from typing import List, Optional

import fire

import os

from llama import Dialog, Llama

from datasets import load_dataset

import time

import json

def get_prompt_store():
    dir_name = "fs_prompts_outputs"
    file_name = "fs-prompt-store.json"
    store_path = os.path.join(dir_name,file_name)
    
    with open(store_path, 'r') as file:
        prompt_store = json.load(file)

    return prompt_store


def get_best_prompt(prompt_store):
    best_prompt_index = 7 # prompt +1
    return prompt_store[best_prompt_index]



def extract_data(dataset_name,example):
    if dataset_name=="snli-hard":
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

    snli_demos = ["snli","snli-hard"]
    
    if dataset_name in snli_demos:
        # enc
        demos.append(dataset[2])
        demos.append(dataset[0])
        demos.append(dataset[1])

        # nce
        # demos.append(dataset[0])
        # demos.append(dataset[1])
        # demos.append(dataset[2])
    else:
        demos.append(dataset[2])
        demos.append(dataset[0])
        demos.append(dataset[11])

    for demo in demos:
        if dataset_name == "hans":
            demo['label'] = hans_mappings[demo['label']]
        else:
            demo['label'] = normal_mappings[demo['label']]

    return demos


def prepare_data(dataset, num_examples, system_prompt, dataset_name, demonstrations):
    dialogs = []
    
    few_shot_prompt = ""
    for demo in demonstrations:
        ex = f"Premise: {demo['premise']}\nHypothesis: {demo['hypothesis']}\nRelationship: {demo['label']}\n \n"
        few_shot_prompt = few_shot_prompt + ex

    for i, example in enumerate(dataset):

        if dataset_name == "snli-hard":
            if example['gold_label'] == -1:
                continue
        else:
            if example['label'] == -1:
                continue

        premise, hypothesis = extract_data(dataset_name,example)

        user_prompt = few_shot_prompt + f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
        
        dialog = [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}]
        
        dialogs.append(dialog)

        # print("############################")
        # print(system_prompt)
        # print(user_prompt)

        if i != -1:  
            if i == num_examples-1:
                break

    return dialogs 

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
        

def load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(ckpt_dir=ckpt_dir,
                            tokenizer_path=tokenizer_path,
                            max_seq_len=max_seq_len,
                            max_batch_size=max_batch_size)
    return generator



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    
    run different prompts with zs and save to file
    
    """
    
    dataset_test_flags = {
        "snli": True,
        # "snli-hard": True,
        # "mnli-mm": True,
        # "mnli-m": True,
        # "hans": True
    }

    test_set_paths = {
        "snli": ["stanfordnlp/snli", "test"],
        "snli-hard": ["au123/snli-hard", "test"],
        "mnli-mm": ["nyu-mll/multi_nli", "validation_mismatched"],
        "mnli-m": ["nyu-mll/multi_nli", "validation_matched"],
        "hans": ["hans","validation"]
    }

    train_set_paths = {
        "snli": ["stanfordnlp/snli", "train"],
        "snli-hard": ["stanfordnlp/snli", "train"],
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
        
    print("> Loading Prompts...")

    # get prompts from file
    prompt_store = get_prompt_store()
    best_prompt = get_best_prompt(prompt_store)


    print("> Building LLaMA-3-8b-Instruct Model...")
# 
    generator = load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    
    print("> Preparing Data and Generating Responses...")

    max_gen_len = 3
    num_test_examples = -1

    for dataset_name in to_test:

        test_set = test_sets[dataset_name]
        train_set = train_sets[dataset_name]
        
        demonstrations = prepare_demonstrations(dataset_name, train_set)
        
        preds_file = f"fs-preds-{dataset_name}-enc-8.txt"

        system_prompt = best_prompt["text_hans"] if dataset_name == "hans" else best_prompt["text"]

        dialogs = prepare_data(test_set, num_test_examples, system_prompt, dataset_name, demonstrations)

        # print(dialogs[0])
        
        dir_name = "fs_baselines"
        output_path = os.path.join(dir_name,preds_file)

        results = process_batches(generator, max_batch_size, dialogs, max_gen_len, temperature, top_p)

        # results = [ {'generation':{'content': 'hello'}},{'generation':{'content': 'world'}} ]

        with open(output_path, 'w') as file:
            for result in results:
                file.write(f"{result['generation']['content']}\n")
    

    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")
