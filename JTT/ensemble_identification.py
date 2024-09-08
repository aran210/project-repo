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

import random



def extract_file(filename):
    with open(filename, 'r') as file:
        file_contents = file.readlines()
    file_contents = [el.strip() for el in file_contents]
    
    return file_contents



def extract_data(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]

    return premise, hypothesis



def prepare_sample(sample):

    premise, hypothesis = extract_data(sample)

    system_prompt = "Given the premise is factually true. Therefore, it must be entailment or contradiction or neutral that the hypothesis is inferred from it. Please choose exactly one of 'Entailment', 'Neutral' or 'Contradiction' in a single word response."
    
    user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
    
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

    error_set_ids = []
    error_set_labels = []
    for index,pred in enumerate(int_preds):
        if pred != labels[index]:
            error_set_ids.append(index)
            error_set_labels.append(labels[index])

    return error_set_ids, error_set_labels


def main(
    is_snli: bool,
    is_mixed_paft: bool,
    num_training_samples: int, 
    ensemble_run: int,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    
    run different prompts with zs and save to file
    
    """

    # False for MNLI, True for SNLI

    # is_snli = True
    # is_mixed_paft = True
    # num_training_samples = 25000
    num_ensemble = 8

    if is_mixed_paft:
        prefix = 'mixed-paft-snli' if is_snli else 'mixed-paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'

    cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
    peft_model_id = f"au123/LLaMA3-JTT-ID-MODEL-{prefix.upper()}"
    hf_token = ""
    
    login(
      token=hf_token, # ADD YOUR TOKEN HERE
      add_to_git_credential=False
    )


    #  set seed here
    seed = -1
    

    ############ SECTION 1: Construct training subset to evaluate


    if is_snli:
        train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(num_training_samples))
    else:
        train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(num_training_samples))
        

    print(len(train_set))
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    print(len(train_set))

    train_set_labels = [example['label'] for example in train_set]

    print(len(train_set_labels))

    X = train_set.map(prepare_sample, remove_columns=train_set.features,batched=False)

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
    
    invalids = {}
    ensemble_preds = {}
    for ensemble_run in range(num_ensemble):
        ensemble_preds[f"{ensemble_run}"] = []
        invalids[f"{ensemble_run}"] = 0

    for ensemble_run in range(num_ensemble):
        predictions = ensemble_preds[f"{ensemble_run}"]

        for index, ex in enumerate(X):
            ex = tokenizer.apply_chat_template(ex["messages"][:2], tokenize=True, add_generation_prompt=True,return_tensors="pt").to(model.device)

            valid_answer_found = False
            attempts = 0
            while not valid_answer_found and attempts < 10:
                output = model.generate(ex,
                            max_new_tokens=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id)
        
                response = tokenizer.decode(output[0])
                # response = output[0]['generated_text'][2]['content']
                answer = response[response.find("assistant<|end_header_id|>\n\n")+len("assistant<|end_header_id|>\n\n"):]

                if "Ent" in answer or "Neutral" in answer or "Contr" in answer:
                    valid_answer_found = True
                
                attempts += 1
        
            if valid_answer_found:
                predictions.append({"answer":answer})
            else:
                options = ['Ent', 'Neutral', 'Contr']
                # out of "Ent", "Neutral" and "Contr", append random answer using random library
                predictions.append({'answer': random.choice(options)})
                invalids[f"{ensemble_run}"] += 1

            if index % 100 == 0:
                print(X[index],index, {"answer": answer})

            if num_test_examples != -1:
                if index == num_test_examples-1:
                    break



    
    # ############ SECTION 4: Get Error Set, Save to File

    for ensemble_run in range(num_ensemble):
        predictions = ensemble_preds[f"{ensemble_run}"]
        error_set_ids, error_set_labels = process_results(predictions,train_set_labels)
        print(len(error_set_ids),len(error_set_labels))
    
        with open(f"error_set/ensemble_{prefix}_error_set_ids_{ensemble_run}", 'w') as file:
            for index in error_set_ids:
                file.write(f"{index}\n")

        with open(f"error_set/ensemble_{prefix}_error_set_preds_{ensemble_run}", 'w') as file:
            for pred in predictions:
                file.write(f"{pred}\n")

    with open(f"test_results/ensemble/run{ensemble_run}/ensemble_{prefix}_invalids", 'w') as file:
        file.write(f"{invalids}")


    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")


