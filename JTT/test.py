from typing import List, Optional

import fire

import os

from datasets import load_dataset, concatenate_datasets

from datasets.arrow_dataset import Dataset

import time

import json

from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM

from transformers import TrainingArguments

from trl import SFTTrainer, setup_chat_format

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch

from huggingface_hub import login

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



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

    preds = int_preds
    labels = [int(y) for y in labels]

    
    overall_acc = accuracy_score(labels, preds)
    overall_class_report = classification_report(labels, preds)

    return overall_acc, overall_class_report



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

    is_snli = True
    is_mixed_paft = False

    if is_mixed_paft:
        prefix = 'mixed-paft-snli' if is_snli else 'mixed-paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'
    run = 117

    cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
    peft_model_id = f"au123/LLaMA3-JTT-RETRAINED-MODEL-{prefix.upper()}"
    hf_token = "" # set HF token
    
    login(
      token=hf_token, # ADD YOUR TOKEN HERE
      add_to_git_credential=False
    )



    

    ############ SECTION 1: Load Sets

    dataset_test_flags = {
        "snli": True,
        "snli-hard": True,
        "mnli-mm": True,
        # "mnli-m": True,
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


    
    

    # ############ SECTION 2: Load Re-Trained Model




    
    print("> Building LLaMA-3-8b-Instruct Model...")




    model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id,
            token=hf_token,
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id,cache_dir=cache_dir)






    # ############ SECTION 3: Evaluate Validation Set
    


        
    print("> Preparing Data and Generating Responses...")

    max_gen_len = 1
    num_test_examples = 500

    LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    # print(formatted_examples)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token # <|eot_id|>
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE


    for dataset_name in to_test:
        if num_test_examples == -1:
            X_test = test_sets[dataset_name]
        else:
            X_test = test_sets[dataset_name].select(range(num_test_examples))
    
        if dataset_name == "snli":
            X_test = X_test.filter(lambda example: example["label"] in [0,1,2])
    
        X_test = X_test.map(lambda sample: prepare_sample(sample, dataset_name), remove_columns=X_test.features,batched=False)
    

        print(X_test[0])
    
        predictions = []
        labels = []
    
        for index, ex in enumerate(X_test):
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
    
            if index % 1000 == 0:
                print(X_test[index],index, {"answer": answer})
    
            if index != -1:
                if index == num_test_examples-1:
                    break
        
    
            
        with open(f"test_results/run{run}/zs-{prefix}-on-{dataset_name}-preds", 'w') as file:
            for pred in predictions:
                file.write(f"{pred}\n")


    
    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")


