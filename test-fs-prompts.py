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


def get_untested_prompts(prompt_store):
    return [prompt for prompt in prompt_store if not prompt['tested']]


def mark_tested(prompts):
    for prompt in prompts:
        prompt['tested'] = True


def update_store(data):
    dir_name = "fs_prompts_outputs"
    file_name = "fs-prompt-store.json"
    store_path = os.path.join(dir_name,file_name)
    
    with open(store_path, 'w') as file:
        json.dump(data, file, indent=4)


def extract_data(dataset_name,example):
    if dataset_name=="snli-hard":
        premise = example["sentence1"]
        hypothesis = example["sentence2"]
    else:
        premise = example["premise"]
        hypothesis = example["hypothesis"]

    return premise, hypothesis


def prepare_demonstrations(dataset_name, dataset, num_demonstrations):

    normal_mappings = {0:"Entailment",1:"Neutral",2:"Contradiction"}
    hans_mappings = {0:"Entailment",1:"Non-Entailment",2:"Non-Entailment"}
    
    demos = []
    
    # for i in range(num_demonstrations):
    #     demos.append(dataset[i])

    # enc
    demos.append(dataset[2])
    demos.append(dataset[0])
    demos.append(dataset[1])

    changes_needed = ["mnli-mm","mnli-m","hans"]
    if dataset_name in changes_needed:
        demos[0] = dataset[2]
        demos[1] = dataset[0]
        demos[2] = dataset[11]

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

        # print("##############")
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
    untested_prompts = get_untested_prompts(prompt_store)


    print("> Building LLaMA-3-8b-Instruct Model...")

    generator = load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    
    print("> Preparing Data and Generating Responses...")

    max_gen_len = 3
    num_test_examples = -1
    num_demonstrations = 3

    for dataset_name in to_test:
        test_set = test_sets[dataset_name]
        train_set = train_sets[dataset_name]
        demonstrations = prepare_demonstrations(dataset_name, train_set, num_demonstrations)
        
        for prompt in untested_prompts:
            preds_file = prompt["preds_file"]
            preds_file = preds_file.replace('NAME',dataset_name)

            if dataset_name == "hans":
                system_prompt = prompt["text_hans"]
            else:
                system_prompt = prompt["text"]

            dialogs = prepare_data(test_set, num_test_examples, system_prompt, dataset_name, demonstrations)

            # print(dialogs[0])
            
            dir_name = "fs_prompts_outputs"
            output_path = os.path.join(dir_name,preds_file)
    
            results = process_batches(generator, max_batch_size, dialogs, max_gen_len, temperature, top_p)

            # results = [ {'generation':{'content': 'hello'}},{'generation':{'content': 'world'}} ]
    
            with open(output_path, 'w') as file:
                for result in results:
                    file.write(f"{result['generation']['content']}\n")
    

    # mark_tested(untested_prompts)
    # update_store(prompt_store)
    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")
