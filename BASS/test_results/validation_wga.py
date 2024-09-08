import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import random
from datasets import load_dataset, concatenate_datasets

def validate_outputs(labels, int_preds, isHans=False):

    # check class size correct after converting to integers
    if isHans:
        assert len(set(int_preds)) == 2
    else:
        assert len(set(int_preds)) == 3
        
    # check labels and predictions equal in size after converting to integers
    assert len(labels) == len(int_preds)


def extract_file(filename):
    with open(filename, 'r') as file:
        file_contents = file.readlines()
    file_contents = [el.strip() for el in file_contents]
    
    return file_contents


def compare_predictions(preds, labels):
    preds = [int(x) for x in preds]
    labels = [int(y) for y in labels]
    
    acc = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    class_report = classification_report(labels, preds)

    return acc, conf_matrix, class_report


def extract_preds(preds):
    results = []
    for i,pred in enumerate(preds):
        answer_idx = pred.find("'answer': '")
        answer = pred[answer_idx+len("'answer': '"):-2]
        results.append(answer)

    return results




# file_endings = {
#     "snli": True,
#     "snli-hard": True,
#     "mnli-mm": True,
#     "mnli-m": True,
#     "hans": True
# }

def process_results(run,is_snli):

    if is_snli:
        val_set = load_dataset("stanfordnlp/snli", split="validation")   
        predictions = extract_file(f"validation/run{run}/zs-val-snli-preds")
        labels = extract_file(f"validation/val-snli-labels.txt")
    else:
        val_set = load_dataset("nyu-mll/multi_nli", split="validation_matched")   
        predictions = extract_file(f"run{run}/zs-mnli-on-mnli-m-preds")
        labels = extract_file(f"test_set_labels/mnli-m-labels.txt")

    
    val_set = val_set.filter(lambda example: example["label"] in [0,1,2])
    
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

    print(index)
    
    total = 0
    for group in groups.keys():
        # print(group, len(groups[group]))
        total += len(groups[group])
    
    
    # 2. get accuracy by group

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

    # int_preds = [label_mappings[pred] for pred in preds]

    int_preds = []
    for pred in preds:
        if pred not in label_mappings.keys():
            int_preds.append(random.choice([0,1,2]))
        else:
            int_preds.append(label_mappings[pred])

    preds = int_preds
    labels = [int(y) for y in labels]

    for group in groups.keys():
        group_idxs = groups[group]


        group_preds = [preds[idx] for idx in group_idxs if idx < len(preds)]
        group_labels = [labels[idx] for idx in group_idxs if idx < len(labels)]

        group_accuracies[group] = accuracy_score(group_labels,group_preds)

        # print(f"  - {group}: {group_accuracies[group]}


    val_scores = []
    weighted_val_scores = []
    weighted_val_size = 0
    ignored_groups = ["negation-2","negation-0","no-negation-0","no-negation-1"]
    for group in groups.keys():
        if group in ignored_groups:
            continue
        else:
            val_scores.append(group_accuracies[group])
            weighted_val_scores.append(group_accuracies[group]*len(groups[group]))
            weighted_val_size += len(groups[group])
            
    
    weighted_val_score = sum(weighted_val_scores)/weighted_val_size
    
        
    
    overall_acc = accuracy_score(labels, preds)


    final_ave = round(((overall_acc+weighted_val_score)/2)*100,2)

    print(f"Run{run}: {final_ave}")





process_results(1,True) 