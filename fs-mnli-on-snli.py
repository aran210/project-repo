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


def get_prompts(prompt_store):
    prompt_nums = [8,9]
    return [prompt for prompt in prompt_store if prompt['prompt_num'] in prompt_nums]


def extract_data(dataset_name,example):
    if dataset_name=="snli-hard":
        premise = example["sentence1"]
        hypothesis = example["sentence2"]
    else:
        premise = example["premise"]
        hypothesis = example["hypothesis"]

    return premise, hypothesis


def prepare_demonstrations(dataset):

    normal_mappings = {0:"Entailment",1:"Neutral",2:"Contradiction"}
    
    shuffle_seed = 24
    entailments = []
    neutrals = []
    contradictions = []

    shuffled_dataset = dataset.shuffle(shuffle_seed)

    for index,example in enumerate(shuffled_dataset):
        if example['label'] == 0:
            if len(entailments) == 5:
                continue
            else:
                entailments.append(example)
        elif example['label'] == 1:
            if len(neutrals) == 5:
                continue
            else:
                neutrals.append(example)
        else:
            if len(contradictions) == 5:
                continue
            else:
                contradictions.append(example)
        
        if len(entailments) == 5 and len(neutrals) == 5 and len(contradictions) == 5:
            break

    demos = []

    for i in range(5):
        demos.append([entailments[i],neutrals[i],contradictions[i]])
    
    # demos.append(dataset[2])
    # demos.append(dataset[0])
    # demos.append(dataset[11])

    for demo_set in demos:
        # print(demo_set)
        for demo in demo_set:
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
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    
    
    """
    
    print("> Loading SNLI Test Set...")

    
    snli_test = load_dataset("stanfordnlp/snli", split="test")

    print("> Loading MNLI Train Set...")
    
    mnli_train = load_dataset("nyu-mll/multi_nli", split="train")

    print("> Loading Prompts...")

    # get prompts from file
    prompt_store = get_prompt_store()
    untested_prompts = get_prompts(prompt_store)

    print("> Building LLaMA-3-8b-Instruct Model...")

    generator = load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    
    print("> Preparing Data and Generating Responses...")

    max_gen_len = 3
    num_test_examples = -1
    num_demonstrations = 3


    demonstrations = prepare_demonstrations(mnli_train)
    
    for index, demos in enumerate(demonstrations):
        preds_file = f"testing-fs-mnli-on-snli-{index}"

        system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."

        dialogs = prepare_data(snli_test, num_test_examples, system_prompt, "snli", demos)

        print(dialogs[0])
        
        dir_name = "robustness_baselines"
        output_path = os.path.join(dir_name,preds_file)

        results = process_batches(generator, max_batch_size, dialogs, max_gen_len, temperature, top_p)

        #results = [ {'generation':{'content': 'hello'}},{'generation':{'content': f"{dialogs[0]}"}} ]
        
        with open(output_path, 'w') as file:
            for result in results:
                file.write(f"{result['generation']['content']}\n")
    

    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")
