import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

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



def process_results(file_endings,run,is_snli,is_mixed_paft=False):
    
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
            

        if is_snli:
            preds_file = f"{run}/zs-mixed-paft-snli-on-{file_ending}-preds" if is_mixed_paft else f"{run}/zs-snli-on-{file_ending}-preds"
        else:
            preds_file = f"{run}/zs-mixed-paft-mnli-on-{file_ending}-preds" if is_mixed_paft else f"{run}/zs-mnli-on-{file_ending}-preds"
        
        print(f"################### JTT, ZERO-SHOT, {file_ending.upper()} TEST SET RESULTS, {run.upper()} ###################")

        preds = extract_file(preds_file)
        preds = extract_preds(preds)

        is_hans = file_ending == "hans"

        if is_hans:
            int_preds = [pred_mappings_hans[pred] for pred in preds]
        else:
            int_preds = [pred_mappings[pred] for pred in preds]
            
        validate_outputs(labels, int_preds, is_hans)

        acc, conf_matrix, class_report = compare_predictions(int_preds, labels)

        print(f"Accuracy: {acc}")
        print(class_report)



file_endings = {
    "snli": True,
    "snli-hard": True,
    "mnli-mm": True,
    # "mnli-m": True,
    "hans": True
}


# process_results(file_endings,"run1",True)
# process_results(file_endings,"run2",True)
# process_results(file_endings,"run3",True)
# process_results(file_endings,"run4",False)
# process_results(file_endings,"run5",False)
# process_results(file_endings,"run6",False)
# process_results(file_endings,"run7",False)
# process_results(file_endings,"run8",False)
# process_results(file_endings,"run9",False) # best MNLI-JTT for HANS
# process_results(file_endings,"run10",True)
# process_results(file_endings,"run11",False)
process_results(file_endings,"run12",False) # best MNLI-JTT for SNLI-Hard - Probably Overall MNLI-JTT Best So Far
# process_results(file_endings,"run13",False)
# process_results(file_endings,"run14",False)
# process_results(file_endings,"run15",False)
# process_results(file_endings,"run16",False)
process_results(file_endings,"run17",True) # best SNLI
# process_results(file_endings,"run18",True)

print("\n\n MIXED-PAFT \n\n")

# Mixed-PAFT
process_results(file_endings,"run19",False, True) 
process_results(file_endings,"run20",True, True)
process_results(file_endings,"run21",True, True)
process_results(file_endings,"run22",True, True)
