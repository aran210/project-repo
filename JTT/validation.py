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




def process_results(predictions,labels,groups):

    group_accuracies = {
        "negation-0": [],
        "negation-1": [],
        "negation-2": [],
        "no-negation-0": [],
        "no-negation-1": [],
        "no-negation-2": [],
    }
    
    label_mappings = {"Ent":0, "Neutral":1, "Contr":2}
    
    preds = extract_preds(predictions)

    int_preds = [label_mappings[pred] for pred in preds]

    preds = int_preds
    labels = [int(y) for y in labels]

    for group in groups.keys():
        group_idxs = groups[group]

        group_preds = [preds[idx] for idx in group_idxs]
        group_labels = [labels[idx] for idx in group_idxs]

        group_accuracies[group] = accuracy_score(group_labels,group_preds)
        
    
    overall_acc = accuracy_score(labels, preds)
    overall_class_report = classification_report(labels, preds)

    return group_accuracies, overall_acc, overall_class_report


def get_group_dists(is_snli):

    is_snli = is_snli
    is_mixed_paft = True

    if is_mixed_paft:
        prefix = 'mixed-paft-snli' if is_snli else 'mixed-paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'

    num_examples = 25000
    upsample_factor = 8
    seed = -1 # set seed

    if is_snli:
        train_set = load_dataset("stanfordnlp/snli", split="train").shuffle(seed=seed).select(range(num_examples))
    else:
        train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(num_examples))
        
    print(f"Orig train set size: {len(train_set)}")
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    print(f"Cleaned train set size: {len(train_set)}")

    error_set_ids = [int(idx) for idx in extract_file(f"error_set/{prefix}_error_set_ids")]

    print(f"Error set size: {len(error_set_ids)}")

    error_set = train_set.select(error_set_ids)
    upsampled_error_set = [error_set]*(upsample_factor-1)
    upsampled_error_set = concatenate_datasets(upsampled_error_set)
    
    print(f"Upsampled error set size: {len(upsampled_error_set)}")

    final_train_set = concatenate_datasets([upsampled_error_set,train_set]).shuffle(seed=seed)

    negation_words = ['nobody','no','never','nothing']

    groups_orig = {
        "negation-0": [],
        "negation-1": [],
        "negation-2": [],
        "no-negation-0": [],
        "no-negation-1": [],
        "no-negation-2": [],
    }

    groups_new = {
        "negation-0": [],
        "negation-1": [],
        "negation-2": [],
        "no-negation-0": [],
        "no-negation-1": [],
        "no-negation-2": [],
    }


    for index, ex in enumerate(train_set):
        ex_hypothesis = ex['hypothesis'].split(' ')
        ex_label = ex['label']

        if any(negation in ex_hypothesis for negation in negation_words):
            groups_orig[f"negation-{ex_label}"].append(index)
        else:
            groups_orig[f"no-negation-{ex_label}"].append(index)

    for index, ex in enumerate(final_train_set):
        ex_hypothesis = ex['hypothesis'].split(' ')
        ex_label = ex['label']

        if any(negation in ex_hypothesis for negation in negation_words):
            groups_new[f"negation-{ex_label}"].append(index)
        else:
            groups_new[f"no-negation-{ex_label}"].append(index)


    return groups_orig, groups_new


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

    is_snli = True
    is_mixed_paft = True

    if is_mixed_paft:
        prefix = 'mixed-paft-snli' if is_snli else 'mixed-paft-mnli'
    else:
        prefix = 'snli' if is_snli else 'mnli'

    cache_dir="/vol/bitbucket/au123/projectenv/my_hf_cache"
    peft_model_id = f"au123/LLaMA3-JTT-RETRAINED-MODEL-{prefix.upper()}"
    hf_token = "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"
    
    login(
      token=hf_token, # ADD YOUR TOKEN HERE
      add_to_git_credential=False
    )



    

    ############ SECTION 1: Load and Clean Validation Set


    if is_snli:
        val_set = load_dataset("stanfordnlp/snli",split="validation")
    else:
        val_set = load_dataset("nyu-mll/multi_nli",split="validation_matched")

    print(len(val_set))
    
    val_set = val_set.filter(lambda example: example["label"] in [0,1,2])

    print(len(val_set))

    val_set_labels = val_set['label']

    print(len(val_set_labels))

    X_val = val_set.map(prepare_sample, remove_columns=val_set.features,batched=False)

    print(X_val[0])


    
    

    # ############ SECTION 2: Load Re-Trained Model




    
    print("> Building LLaMA-3-8b-Instruct Model...")

    is_new_test = True


    if is_new_test:
        model = AutoPeftModelForCausalLM.from_pretrained(
                peft_model_id,
                token=hf_token,
                device_map="auto",
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id,cache_dir=cache_dir)






    # ############ SECTION 3: Evaluate Validation Set
    


    if is_new_test:
        
        print("> Preparing Data and Generating Responses...")
    
        max_gen_len = 1
        num_test_examples = -1
    
        LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    
        # print(formatted_examples)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token # <|eot_id|>
        tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    
        predictions = []
        labels = []
    
        for index, ex in enumerate(X_val):
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
            labels.append(val_set_labels[index])
    
            if index % 1000 == 0:
                print(X_val[index],index, {"answer": answer})
    
            if index != -1:
                if index == num_test_examples-1:
                    break
    

        
        with open(f"{prefix}_val_set_preds", 'w') as file:
            for pred in predictions:
                file.write(f"{pred}\n")
    
        with open(f"{prefix}_val_set_labels", 'w') as file:
            for label in labels:
                file.write(f"{label}\n")




    
    # ############ SECTION 5: Worst Group Validation

    if not is_new_test:
        predictions = extract_file(f"{prefix}_val_set_preds")
        labels = extract_file(f"{prefix}_val_set_labels")

    negation_words = ['nobody','no','never','nothing']
    groups = {
        "negation-0": [],
        "negation-1": [],
        "negation-2": [],
        "no-negation-0": [],
        "no-negation-1": [],
        "no-negation-2": [],
    }

    # 1. label groups in val set
    for index, ex in enumerate(val_set):
        ex_hypothesis = ex['hypothesis'].split(' ')
        ex_label = ex['label']

        if any(negation in ex_hypothesis for negation in negation_words):
            groups[f"negation-{ex_label}"].append(index)
        else:
            groups[f"no-negation-{ex_label}"].append(index)


    total = 0
    for group in groups.keys():
        print(group, len(groups[group]))
        total += len(groups[group])


    # 2. get accuracy by group
    group_accuracies, overall_acc, class_report = process_results(predictions, labels, groups)

    print("Group Accuracies:")
    for group in groups.keys():
        print(f"  - {group}: {group_accuracies[group]}")

    
    print("Overall Accuracy:", overall_acc)
    # print("Overall Classification Report:\n", class_report)
    





    
    ################### OTHER: ORIGINAL GROUP DISTRIBUTION VS UPSAMPLED GROUP DISTRIBUTION


    orig, new = get_group_dists(is_snli)

    print("\nORIGINAL")
    for group in orig.keys():
        print(group, len(orig[group]))

    print("\nNEW")
    for group in new.keys():
        print(group, len(new[group]))





    
    
    print("> Process Complete")

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Script completed in {time.time()-start_time} seconds.")


