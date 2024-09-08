from datasets import load_dataset
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import sys
import spacy

nltk.download('punkt')

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")


def save_to_file(idx_set, is_bias, is_10k):
    if is_bias:
        if is_10k:
            filename = "biased_set_ids_10k.txt"
        else:
            filename = "biased_set_ids_20k.txt"
        # filename = "biased_set_ids_val.txt"
    else:
        if is_10k:
            filename = "unbiased_set_ids_10k.txt"
        else:
            filename = "unbiased_set_ids_20k.txt"
        # filename = "unbiased_set_ids_val.txt"
    
    with open(f"test_results/simple_classifier/{filename}", 'w') as file:
        for idx in idx_set:
            file.write(f"{idx}\n")


def syntactic_subsequence(premise, hypothesis):
    premise_words = word_tokenize(premise.lower())
    hypothesis_words = word_tokenize(hypothesis.lower())
    it = iter(premise_words)
    return all(word in it for word in hypothesis_words)


def is_subtree(premise, hypothesis):
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


def load_data(num_training_samples):
# Load MNLI data
    seed = -1 # set seed 
    train_set = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed).select(range(num_training_samples))
    # train_set = load_dataset("nyu-mll/multi_nli", split="validation_matched")

    print(f"Orig train set size: {len(train_set)}")
    
    train_set = train_set.filter(lambda example: example["label"] in [0,1,2])

    print(f"Cleaned train set size: {len(train_set)}")

    return train_set


def find_biased_samples(train_set):
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
    
    # Iterate over the dataset
    for i, example in enumerate(train_set):
        premise = example['premise']
        hypothesis = example['hypothesis']
        label = example['label']
    
        # Check for syntactic subsequence
        if syntactic_subsequence(premise, hypothesis):
            syntactic_subsequence_biases.append((i, label))
            class_count_subsequence[label] += 1
            if label == 0:
                biased_entailment_subsequence.append(i)
        
        # Check for significant lexical overlap (e.g., more than 60%)
        if lexical_overlap(premise, hypothesis, n=1) > 0.6:
            lexical_overlap_biases.append((i, label))
            class_count_overlap[label] += 1
            if label == 0:
                biased_entailment_overlap.append(i)

        # Check for negations in contradiction examples    
        if negation_bias(premise, hypothesis):  # Check if it's a contradiction case
            negation_biases.append((i, label))
            class_count_negations[label] += 1
            if label == 2:
                biased_contradiction_negation.append(i)

        if is_subtree(premise, hypothesis):
            subtree_biases.append((i, label))
            class_count_subtree[label] += 1
            if label == 0:
                biased_entailment_subtree.append(i)
    
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

    return biased_entailment_subsequence, biased_entailment_overlap, biased_contradiction_negation, biased_entailment_subtree




# simple_classifier_identification.py 10000 0
if __name__ == "__main__":
    num_training_samples = int(sys.argv[1])
    should_save = int(sys.argv[2])

    train_set = load_data(num_training_samples)
    
    biased_entailment_subsequence, biased_entailment_overlap, biased_contradiction_negation, biased_entailment_subtree = find_biased_samples(train_set)

    combined_set = biased_entailment_subsequence + biased_entailment_overlap + biased_contradiction_negation + biased_entailment_subtree
    full_biased_set = set(combined_set)

    
    print(f"Combined Set Size:  {len(combined_set)}")
    print(f"Final Biased Set Size: {len(full_biased_set)}")

    full_biased_set = sorted(full_biased_set)

    unbiased_set = []
    for idx in range(len(train_set)):
        if idx not in full_biased_set:
            unbiased_set.append(idx)

    unbiased_set = sorted(unbiased_set)

    print(full_biased_set[:5])
    # print(len(full_biased_set))
    
    print(unbiased_set[:5])
    # print(len(unbiased_set))

    print("for debugging")
    print(train_set[unbiased_set[100]]['premise'], train_set[unbiased_set[100]]['hypothesis'], train_set[unbiased_set[100]]['label'])
    print(train_set[full_biased_set[100]]['premise'], train_set[full_biased_set[100]]['hypothesis'], train_set[full_biased_set[100]]['label'])


    counts = {0: 0, 1: 0, 2: 0}
    for idx in unbiased_set:
        counts[int(train_set[idx]['label'])] += 1

    print(counts)

    is_10k = num_training_samples == 10000

    if should_save == 1:
        save_to_file(full_biased_set, True, is_10k)
        save_to_file(unbiased_set, False, is_10k)












