import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import random
from datasets import load_dataset, concatenate_datasets
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import sys
import spacy


# nltk.download('punkt')

def load_data(is_snli):
    
    if is_snli:
        val_set = load_dataset("stanfordnlp/snli", split="test")
        # val_set = load_dataset("stanfordnlp/snli", split="validation")
    else:
        val_set = load_dataset("nyu-mll/multi_nli", split="validation_matched")

    # print(f"Orig train set size: {len(val_set)}")
    
    val_set = val_set.filter(lambda example: example["label"] in [0,1,2])

    # print(f"Cleaned train set size: {len(val_set)}")

    return val_set


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



def syntactic_subsequence(premise, hypothesis):
    premise_words = word_tokenize(premise.lower())
    hypothesis_words = word_tokenize(hypothesis.lower())
    it = iter(premise_words)
    return all(word in it for word in hypothesis_words)


def is_subtree(premise, hypothesis):

    # Load the SpaCy English language model
    nlp = spacy.load("en_core_web_sm")

    premise_doc = nlp(premise)
    hypothesis_doc = nlp(hypothesis)
    hypothesis_tokens = set([token.lemma_ for token in hypothesis_doc])
    for token in premise_doc:
        subtree_tokens = set([subtoken.lemma_ for subtoken in token.subtree])
        if hypothesis_tokens == subtree_tokens:
            return True
    return False


def lexical_overlap(premise, hypothesis, n=1):
    premise_ngrams = set(ngrams(word_tokenize(premise.lower()), n))
    hypothesis_ngrams = set(ngrams(word_tokenize(hypothesis.lower()), n))
    if len(hypothesis_ngrams) == 0:  # Avoid division by zero
        return 0
    overlap = premise_ngrams.intersection(hypothesis_ngrams)
    return len(overlap) / len(hypothesis_ngrams)


def negation_bias(premise, hypothesis):
    negation_words = {'no', 'not', 'none', 'never', 'nothing'}
    premise_words = set(word_tokenize(premise.lower()))
    hypothesis_words = set(word_tokenize(hypothesis.lower()))
    premise_negations = premise_words & negation_words
    hypothesis_negations = hypothesis_words & negation_words

    # Check if negations are added or removed in the hypothesis
    return not premise_negations == hypothesis_negations



def find_challenging_samples(val_set,verbose=False):
    syntactic_subsequence_biases = []
    lexical_overlap_biases = []
    negation_biases = []
    subtree_biases = []  # Subtree biases list
    
    # Data structures to hold counts per class
    class_count_subsequence = {0: 0, 1: 0, 2: 0}  # 0: entailment, 1: neutral, 2: contradiction
    class_count_overlap = {0: 0, 1: 0, 2: 0}
    class_count_negations = {0: 0, 1: 0, 2: 0}
    class_count_subtree = {0: 0, 1: 0, 2: 0}
    
    # Lists to hold indices of biased samples for entailment
    biased_entailment_subsequence = []
    biased_entailment_overlap = []
    biased_entailment_subtree = []
    biased_contradiction_negation = []

    challenging_samples = []
    
    # Iterate over the dataset
    for i, example in enumerate(val_set):
        premise = example['premise']
        hypothesis = example['hypothesis']
        label = example['label']
    
        # Check for syntactic subsequence
        if syntactic_subsequence(premise, hypothesis):
            syntactic_subsequence_biases.append((i, label))
            class_count_subsequence[label] += 1
            if label == 0:
                biased_entailment_subsequence.append(i)
            else:
                challenging_samples.append(i)
        
        # Check for significant lexical overlap (e.g., more than 60%)
        if lexical_overlap(premise, hypothesis, n=1) > 0.6:
            lexical_overlap_biases.append((i, label))
            class_count_overlap[label] += 1
            if label == 0:
                biased_entailment_overlap.append(i)
            else:
                challenging_samples.append(i)
                
        # Check for negations in contradiction examples    
        if negation_bias(premise, hypothesis):  # Check if it's a contradiction case
            negation_biases.append((i, label))
            class_count_negations[label] += 1
            if label == 2:
                biased_contradiction_negation.append(i)
            else:
                challenging_samples.append(i)

        if is_subtree(premise, hypothesis):
            subtree_biases.append((i, label))
            class_count_subtree[label] += 1
            if label == 0:
                biased_entailment_subtree.append(i)
            else:
                challenging_samples.append(i)


    challenging_samples_ids = sorted(set(challenging_samples))

    if verbose:
        print(f"Syntactic Subsequence Biases Found: {len(syntactic_subsequence_biases)}")
        print(f"Lexical Overlap Biases Found: {len(lexical_overlap_biases)}")
        print(f"Negation Biases Found: {len(negation_biases)}")
        print(f"Subtree Biases Found: {len(subtree_biases)}")
        
        # Print the class-specific counts
        print("Syntactic Subsequence Biases by Class:")
        for cls, count in class_count_subsequence.items():
            print(f"Class {cls}: {count}")
        
        print("Lexical Overlap Biases by Class:")
        for cls, count in class_count_overlap.items():
            print(f"Class {cls}: {count}")
    
        print("Negation Biases by Class:")
        for cls, count in class_count_negations.items():
            print(f"Class {cls}: {count}")
        
        print("Subtree Biases by Class:")
        for cls, count in class_count_subtree.items():
            print(f"Class {cls}: {count}")
    
        
        # Print the biased entailment indices
        print(f"Biased Entailment Syntactic Subsequence Indices: {len(biased_entailment_subsequence)}")
        print(f"Biased Entailment Lexical Overlap Indices: {len(biased_entailment_overlap)}")
        print(f"Biased Contradiction Negation Indices: {len(biased_contradiction_negation)}")
        print(f"Biased Entailment Subtree Indices: {len(biased_entailment_subtree)}")
    
    
        # print(biased_entailment_subsequence[:5])
        # print(biased_entailment_overlap[:5])

    # return biased_entailment_subsequence, biased_entailment_overlap, biased_contradiction_negation, biased_entailment_subtree

    return challenging_samples_ids




def get_challenging_ids(is_snli):

    val_set = load_data(is_snli)
    
    biased_entailment_subsequence, biased_entailment_overlap, biased_contradiction_negation, biased_entailment_subtree = find_challenging_samples(val_set)
    
    combined_set = biased_entailment_subsequence + biased_entailment_overlap + biased_contradiction_negation + biased_entailment_subtree
    full_biased_set = set(combined_set)
    
    
    # print(f"Combined Set Size:  {len(combined_set)}")
    print(f"Final Biased Set Size: {len(full_biased_set)}")
    
    full_biased_set = sorted(full_biased_set)

    unbiased_set = []
    for idx in range(len(val_set)):
        if idx not in full_biased_set:
            unbiased_set.append(idx)

    full_unbiased_set = sorted(unbiased_set)

    challenging_samples_ids = find_challenging_samples(val_set)


    filename = "val_challenging_ids_snli" if is_snli else "val_challenging_ids_mnli"
    with open(filename,'w') as file:
        for idx in challenging_samples_ids:
            file.write(f"{idx}\n")
    
    print(full_biased_set[:5])

    counts = {0: 0, 1: 0, 2: 0}
    for idx in unbiased_set:
        counts[int(val_set[idx]['label'])] += 1
    
    print(counts)


    return challenging_samples_ids



def load_challenging_ids(is_snli):

    filename = "simple_classifier/val_challenging_ids_snli" if is_snli else "simple_classifier/val_challenging_ids_mnli"

    challenging_ids = [int(idx) for idx in extract_file(filename)]

    return challenging_ids



def process_results(run,is_snli,is_simple=True):
    
    # extract preds for biased examples, extract labels for biased examples, compute accuracy

    challenging_samples_ids = get_challenging_ids(is_snli)

    
    # challenging_samples_ids = load_challenging_ids(is_snli)
    
    if is_snli:
        preds_file = "zs-val-snli-preds"
        labels_file = f"val-snli-labels.txt"
        predictions = extract_file(f"{'validation/simple_classifier/' if is_simple else ''}run{run}/{preds_file}")
        labels = extract_file(f"{'validation/simple_classifier/' if is_simple else ''}test_set_labels/{labels_file}")
    else:
        preds_file = "zs-mnli-on-mnli-m-preds"
        labels_file = "mnli-m-labels.txt"
        predictions = extract_file(f"{'simple_classifier/' if is_simple else ''}run{run}/{preds_file}")
        labels = extract_file(f"{'simple_classifier/' if is_simple else ''}test_set_labels/{labels_file}")

    preds = extract_preds(predictions)
    
    label_mappings = {"Ent":0, "Neutral":1, "Contr":2}
    int_preds = [label_mappings[pred] for pred in preds]
    int_labels = [int(y) for y in labels]

    challenging_preds = [int_preds[index] for index in challenging_samples_ids]
    challenging_labels = [int_labels[index] for index in challenging_samples_ids]

    acc, conf_matrix, class_report = compare_predictions(challenging_preds,challenging_labels)

    overall_acc,_,_ = compare_predictions(int_preds,int_labels)

    print(f"Run {run}: {round(((overall_acc+acc)/2)*100,2)}")








process_results(1,True)



































































































