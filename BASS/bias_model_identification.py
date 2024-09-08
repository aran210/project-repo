from typing import List, Optional

import fire

import os

from datasets import load_dataset, load_from_disk

from datasets.arrow_dataset import Dataset

import time

import random

import json

from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM

from transformers import TrainingArguments

from trl import SFTTrainer, setup_chat_format

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch

from huggingface_hub import login



def extract_file(filename):
    with open(filename, 'r') as file:
        file_contents = file.readlines()
    file_contents = [el.strip() for el in file_contents]
    
    return file_contents



def extract_data(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]

    return premise, hypothesis



def prepare_sample(sample,is_paft,inc_premise):

    premise, hypothesis = extract_data(sample)

    system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."

    if inc_premise:
        user_prompt = f"Premise: \nHypothesis: {hypothesis}\nRelationship: "
    else:
        user_prompt = f"Hypothesis: {hypothesis}\nRelationship: "

    if is_paft:
        dialog = { "messages":
              [{"role": "user", "content": user_prompt}]
         }
    else:
        dialog = { "messages":
              [{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}]
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
        

def extract_preds(preds):
    results = []
    for i,pred in enumerate(preds):
        pred = str(pred)
        answer_idx = pred.find("'answer': '")
        answer = pred[answer_idx+len("'answer': '"):-2]
        results.append(answer)

    return results


def process_results(predictions,labels):
    
    label_mappings = {"Ent":0, "Neutral":1, "Contr":2}
    
    preds = extract_preds(predictions)

    int_preds = [label_mappings[pred] for pred in preds]
    labels = [int(label) for label in labels]

    print(int_preds[:10])
    print(labels[:10])

    unbiased_set_ids = []
    biased_set_ids = []
    for index,pred in enumerate(int_preds):
        if pred == labels[index]:
            biased_set_ids.append(index)
        else:
            unbiased_set_ids.append(index)

    return unbiased_set_ids, biased_set_ids









# torchrun --nproc_per_node 1 bias_model_identification.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --num_training_samples 10000 --is_snli True --is_paft True --inc_premise False --start_index 0 --num_run 1
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    num_training_samples: int = 10000,
    is_snli: bool = True,
    is_paft: bool = True,
    inc_premise: bool = False,
    start_index: int = 0,
    num_run: int = 1,
    seed: int = 42
):
    """
    
    run different prompts with zs and save to file
    
    """
    run = num_run
    
    if is_paft:
        prefix = 'paft-snli' if is_snli else 'paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'


    cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"

    peft_model_id = f"au123/LLaMA3-BIAS-MODEL-{prefix.upper()}"

    hf_token = ""
    
    login(
      token=hf_token, # ADD YOUR TOKEN HERE
      add_to_git_credential=False
    )



    

    ############ SECTION 1: Construct training subset to evaluate

    if start_index > 0:
        end_index = start_index + num_training_samples
        if is_snli:
            # train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(start_index,end_index))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/snli_train').shuffle(seed=seed).select(range(start_index,end_index))
        else:
            # train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(start_index,end_index))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/mnli_train').shuffle(seed=seed).select(range(start_index,end_index))
    else:
        if is_snli:
            # train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(num_training_samples))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/snli_train').shuffle(seed=seed).select(range(num_training_samples))
        else:
            # train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(num_training_samples))
            train_set = load_from_disk('/vol/bitbucket/au123/projectenv/my_hf_cache/mnli_train').shuffle(seed=seed).select(range(num_training_samples))
        

    print(len(train_set))
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    # print(len(train_set))

    train_set_labels = [example['label'] for example in train_set]

    print(len(train_set),len(train_set_labels))

    X = train_set.map(lambda sample: prepare_sample(sample,is_paft,inc_premise), remove_columns=train_set.features,batched=False)

    print(X[0])
    
    

    # ############ SECTION 2: Load Identification Model




    
    print("> Building LLaMA-3-8b-Instruct Model...")

    
    model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id,
            token=hf_token,
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id,cache_dir=cache_dir)






    # ############ SECTION 3: Evaluate Training Examples
    


    

    
    print("> Preparing Data and Generating Responses...")

    max_gen_len = 1
    num_test_examples = -1

    LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    # print(formatted_examples)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token # <|eot_id|>
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE


    predictions = []
    inv_count = 0
    for index, ex in enumerate(X):
        # if index == 0:
        #     print(ex["messages"][0])
        #     print(ex["messages"][1])
            
        ex = tokenizer.apply_chat_template(ex["messages"][:2], tokenize=True, add_generation_prompt=True,return_tensors="pt").to(model.device)
        label = train_set_labels[index]

        # if label == -1:
        #     continue

        answer = ""
        iters = 0
        while answer not in ["Ent","Contr","Neutral"]:
            iters += 1
            output = model.generate(ex,
                          max_new_tokens=max_gen_len,
                          temperature=temperature,
                          top_p=top_p,
                          eos_token_id=tokenizer.eos_token_id,
                          pad_token_id=tokenizer.pad_token_id)
        
            response = tokenizer.decode(output[0])
            # response = output[0]['generated_text'][2]['content']
            answer = response[response.find("assistant<|end_header_id|>\n\n")+len("assistant<|end_header_id|>\n\n"):]

            if answer not in ["Ent","Contr","Neutral"]:
                if iters == 10:
                    inv_count += 1
                    answer = random.choice(["Ent","Contr", "Neutral"])
                
        
        predictions.append({"answer":answer})

        if index % 1000 == 0:
            print(X[index],index, {"answer": answer})

        if index != -1:
            if index == num_test_examples-1:
                break


    print(inv_count)
    with open(f"test_results/invalids.txt", 'a') as file:
        file.write(f"Run {run} has {inv_count} invalid samples (random guesses).\n")
        


    
    ############ SECTION 4: Get Error Set, Save to File

    # dir_path = f"test_results/run{run}/"
    
    # predictions = extract_file(f"{dir_path}/preds")
    # train_set_labels = extract_file(f"{dir_path}/labels")
    

    unbiased_set_ids, biased_set_ids = process_results(predictions,train_set_labels)
    print(len(unbiased_set_ids),len(biased_set_ids))

    biased_set = train_set.select(biased_set_ids)

    print("\nDEBUGGING\n")
    print("1111", biased_set[1111])
    print("-1", biased_set[-1])

    dir_path = f"test_results/run{run}"
    
    with open(f"{dir_path}/unbiased_set_ids", 'w') as file:
        for index in unbiased_set_ids:
            file.write(f"{index}\n")

    with open(f"{dir_path}/biased_set_ids", 'w') as file:
        for index in biased_set_ids:
            file.write(f"{index}\n")

    with open(f"{dir_path}/preds", 'w') as file:
        for pred in predictions:
            file.write(f"{pred}\n")

    with open(f"{dir_path}/labels", 'w') as file:
        for label in train_set_labels:
            file.write(f"{label}\n")

    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")


