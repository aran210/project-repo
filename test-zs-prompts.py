from typing import List, Optional

import fire

import os

from llama import Dialog, Llama

from datasets import load_dataset

import time

import json

def get_prompt_store():
    dir_name = "zs_prompts_outputs"
    file_name = "zs-prompt-store.json"
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
    dir_name = "zs_prompts_outputs"
    file_name = "zs-prompt-store.json"
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

def prepare_data(dataset, num_examples, system_prompt, dataset_name):
    dialogs = []

    for i, example in enumerate(dataset):

        if dataset_name == "snli-hard":
            if example['gold_label'] == -1:
                continue
        else:
            if example['label'] == -1:
                continue

        premise, hypothesis = extract_data(dataset_name,example)
    
        user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
        
        dialog = [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}]
        dialogs.append(dialog)


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
    temperature: float = 0.7,
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

    dataset_paths = {
        "snli": ["stanfordnlp/snli", "test"],
        "snli-hard": ["au123/snli-hard", "test"],
        "mnli-mm": ["nyu-mll/multi_nli", "validation_mismatched"],
        "mnli-m": ["nyu-mll/multi_nli", "validation_matched"],
        "hans": ["hans","validation"]
    }

    to_test = []
    dataset = {}

    for dataset_name in dataset_test_flags.keys():
        if dataset_test_flags[dataset_name]:
            to_test.append(dataset_name)

    print("> Loading Test Sets...")
    
    for dataset_name in to_test:
        dataset[dataset_name] = load_dataset(dataset_paths[dataset_name][0],split=dataset_paths[dataset_name][1])


    print("> Preparing Prompts and Dialogs...")


    # get prompts from file
    prompt_store = get_prompt_store()
    untested_prompts = get_untested_prompts(prompt_store)


    print("> Building LLaMA-3-8b-Instruct Model...")

    if ft_model:
        ckpt_dir = 
    generator = load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    
    print("> Generating Responses...")

    max_gen_len = 3
    num_test_examples = -1

    for dataset_name in to_test:
        test_set = dataset[dataset_name]
        
        for prompt in untested_prompts:
            preds_file = prompt["preds_file"]
            preds_file = preds_file.replace('NAME',dataset_name)

            if dataset_name == "hans":
                system_prompt = prompt["text_hans"]
            else:
                system_prompt = prompt["text"]

            print(dataset_name, system_prompt)
                
            dialogs = prepare_data(test_set, num_test_examples, system_prompt, dataset_name)
            
            dir_name = "zs_prompts_rerun"
            output_path = os.path.join(dir_name,preds_file)
    
            results = process_batches(generator, max_batch_size, dialogs, max_gen_len, temperature, top_p)
    
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
