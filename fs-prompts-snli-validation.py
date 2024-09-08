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


def extract_data(dataset_name,example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]

    return premise, hypothesis, label


def prepare_demonstrations(dataset_name, dataset):

    normal_mappings = {0:"Entailment",1:"Neutral",2:"Contradiction"}
    
    demos = []
    
    # ENC
    # demos.append(dataset[2])
    # demos.append(dataset[0])
    # demos.append(dataset[1])

    # NCE
    demos.append(dataset[0])
    demos.append(dataset[1])
    demos.append(dataset[2])

    for demo in demos:
        demo['label'] = normal_mappings[demo['label']]

    return demos


def prepare_data(dataset, num_examples, system_prompt, dataset_name, demonstrations):
    dialogs = []
    labels = []
    
    few_shot_prompt = ""
    for demo in demonstrations:
        ex = f"Premise: {demo['premise']}\nHypothesis: {demo['hypothesis']}\nRelationship: {demo['label']}\n \n"
        few_shot_prompt = few_shot_prompt + ex

    for i, example in enumerate(dataset):

        if example['label'] == -1:
            continue

        premise, hypothesis, label = extract_data(dataset_name,example)

        user_prompt = few_shot_prompt + f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
        
        dialog = [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}]
        
        dialogs.append(dialog)
        labels.append(label)

        print("##############################")
        print(system_prompt)
        print(user_prompt)

        if i != -1:  
            if i == num_examples-1:
                break

    return dialogs, labels 

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
    
    run different prompts with fs and save to file
    baseline is only ENC examples
    
    """
    
    dataset_validation_flags = {
        "snli": True
    }

    validation_set_paths = {
        "snli": ["stanfordnlp/snli", "validation"]
    }

    train_set_paths = {
        "snli": ["stanfordnlp/snli", "train"],
    }

    to_test = []
    validation_sets = {}
    train_sets = {}

    for dataset_name in dataset_validation_flags.keys():
        if dataset_validation_flags[dataset_name]:
            to_test.append(dataset_name)

    print("> Loading Validation Sets...")
    
    for dataset_name in to_test:
        validation_sets[dataset_name] = load_dataset(validation_set_paths[dataset_name][0],split=validation_set_paths[dataset_name][1])

    print("> Loading Train Sets...")
    
    for dataset_name in to_test:
        train_sets[dataset_name] = load_dataset(train_set_paths[dataset_name][0],split=train_set_paths[dataset_name][1])

    print("> Loading Prompts...")

    # get prompts from file
    prompt_store = get_prompt_store()
    untested_prompts = prompt_store


    print("> Building LLaMA-3-8b-Instruct Model...")

    # generator = load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    
    print("> Preparing Data and Generating Responses...")

    max_gen_len = 3
    num_test_examples = -1

    for dataset_name in to_test:
        validation_set = validation_sets[dataset_name]
        train_set = train_sets[dataset_name]
        demonstrations = prepare_demonstrations(dataset_name, train_set)
        
        for prompt in untested_prompts:
            preds_file = f"fs-snli-validation-nce-prompt-{prompt['prompt_num']}.txt"

            system_prompt = prompt["text_hans"] if dataset_name == "hans" else prompt["text"]

            dialogs, labels = prepare_data(validation_set, num_test_examples, system_prompt, dataset_name, demonstrations)
            
            dir_name = "fs_validation"
            output_path = os.path.join(dir_name,preds_file)
    
            # results = process_batches(generator, max_batch_size, dialogs, max_gen_len, temperature, top_p)

            # results = [ {'generation':{'content': 'hello'}},{'generation':{'content': 'world'}} ]
    
            # with open(output_path, 'w') as file:
            #     for result in results:
            #         file.write(f"{result['generation']['content']}\n")

            # with open(f"{dir_name}/snli-validation-labels.txt", 'w') as file:
            #     for label in labels:
            #         file.write(f"{label}\n")

    
    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")
