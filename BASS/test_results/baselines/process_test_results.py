import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import random

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



def process_results(file_endings,run,is_snli=True,is_paft=False):
    
    pred_mappings = {"Ent":0, "Neutral":1, "Contr":2}
    pred_mappings_hans = {"Ent":0, "Neutral":1, "Contr":1}
    label_mappings_snli_hard = {"entailment":0, "neutral":1, "contradiction":2}

    print(f"####################################################################################################")
    print(f"####################################################################################################")


    for file_ending in file_endings.keys():

        if not file_endings[file_ending]: continue
    
        labels = extract_file(f"test_set_labels/{file_ending}-labels.txt")
        
        if file_ending == "snli-hard":
            labels = [label_mappings_snli_hard[label] for label in labels]
            
        if is_paft:
            preds_file = f"{run}/zs-paft-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"
        else:
            preds_file = f"{run}/zs-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"

        if run == "run0": 
            preds_file = f"{run}/zs-base-on-{file_ending}-preds"
            
            # preds_file = f"{run}/zs-bias-model-paft-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"
            # preds_file = f"{run}/preds"
            # labels = extract_file(f"{run}/labels")

        
        print(f"################### {file_ending.upper()} TEST SET RESULTS, {'SNLI MODEL' if is_snli else 'MNLI MODEL'}, {run.upper()} ###################")

        preds = extract_file(preds_file)
        preds = extract_preds(preds)

        is_hans = file_ending == "hans"

        int_preds = []

        if is_hans:
            # int_preds = [pred_mappings_hans[pred] for pred in preds]
            for pred in preds: 
                if pred in pred_mappings_hans.keys():
                    int_preds.append(pred_mappings_hans[pred])
                else:
                    int_preds.append(random.choice([0,1]))
        else:
            # int_preds = [pred_mappings[pred] for pred in preds]
            for pred in preds:
                if pred in pred_mappings.keys():
                    int_preds.append(pred_mappings[pred])
                else:
                    int_preds.append(random.choice([0,1,2]))
        
       # for ind,pr in enumerate(preds):
       #     if 
        

        validate_outputs(labels, int_preds, is_hans)

        acc, conf_matrix, class_report = compare_predictions(int_preds, labels)

        print(f"Accuracy: {acc}")
        print(class_report)
        print(conf_matrix)



file_endings = {
    # "snli": True,
    # "snli-hard": True,
    # "mnli-mm": True,
    # "mnli-m": True,
    "hans": True
}


# is_snli, is_bias, is_paft

is_paft = False

# process_results(file_endings,"run0",False,is_paft) # base model

process_results(file_endings,"run1",True,True) # snli-paft, 10k, seed-42
# process_results(file_endings,"run1",False,True) # mnli-paft, 20k, seed-42

process_results(file_endings,"run2",True,True) # snli-paft, 10k, seed-42
# process_results(file_endings,"run2",True,True) # mnli-paft, 20k, seed-42

# process_results(file_endings,"run2",False,is_paft) # MNLI 20k, no shuffle
# process_results(file_endings,"run3",False,is_paft) # MNLI 20k, seed=1
# process_results(file_endings,"run4",False,is_paft) # MNLI 20k, seed=1

process_results(file_endings,"run2",True,is_paft) # SNLI 10k, no shuffle
process_results(file_endings,"run3",True,is_paft) # SNLI 10k, seed=1
process_results(file_endings,"run4",True,is_paft) # SNLI 10k, seed=2

# process_results(file_endings,"run5",True,is_paft) # SNLI, 30k, LR=2e-05, no shuffle
# process_results(file_endings,"run5",False,is_paft) # MNLI, 30k, LR=2e-05, no shuffle

# process_results(file_endings,"run6",True,is_paft) # SNLI baseline 20k, LR=2e-05



process_results(file_endings,"run7",True,is_paft) # SNLI 10k, LR=2e-06
# process_results(file_endings,"run7",False,is_paft) # SNLI 10k, LR=2e-06