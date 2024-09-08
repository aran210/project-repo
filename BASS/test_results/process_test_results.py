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



def process_results(file_endings,run,is_snli=True,is_bias=False,is_paft=False):
    
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
            
        if not is_bias:
            if is_paft:
                preds_file = f"{run}/zs-paft-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"
            else:
                preds_file = f"{run}/zs-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"
        else:
            if is_paft:
                preds_file = f"{run}/zs-bias-model-paft-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"
            else:
                preds_file = f"{run}/zs-bias-model-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"
            # preds_file = f"{run}/zs-bias-model-paft-{'snli' if is_snli else 'mnli'}-on-{file_ending}-preds"
            # preds_file = f"{run}/preds"
            # labels = extract_file(f"{run}/labels")

        
        print(f"################### JTT, ZERO-SHOT, {file_ending.upper()} TEST SET RESULTS, {'SNLI MODEL' if is_snli else 'MNLI MODEL'}, {run.upper()} ###################")

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

        # print(conf_matrix)

        print(f"Accuracy: {acc}")
        print(class_report)



file_endings = {
    "snli": True,
    "snli-hard": True,
    "mnli-mm": True,
    # "mnli-m": True,
    "hans": True
}


# is_snli, is_bias, is_paft

is_bias = False
is_paft = False
# process_results(file_endings,"run24",True,is_bias,is_paft)
# process_results(file_endings,"run25",False,is_bias,is_paft)



############################ MNLI BASES - ALL USELESS, USE WRONG UPSAMPLING - COULD SHOW EFFECT OF TOO HIGH UPSAMPLING MAYBE IN ANALYSIS
# is_bias = False
# is_paft = False
# process_results(file_endings,"run13",False,is_bias,is_paft)
# process_results(file_endings,"run14",False,is_bias,is_paft)
# process_results(file_endings,"run15",False,is_bias,is_paft)


# ############################ SNLI ER BASELINES - hyp-only classifier with 8x upsample - run17 may actually be useful
# is_bias = False
# is_paft = False
# process_results(file_endings,"run16",True,is_bias,is_paft)
# process_results(file_endings,"run17",True,is_bias,is_paft) #not good actually "Run 17, SNLI, Upsample = 8x, LR = 2e-05, Bias Model = 3 epochs training"
# process_results(file_endings,"run18",True,is_bias,is_paft)



############################ MNLI BATCHER - Might be useful for analysis
# is_bias = False
# is_paft = False
# process_results(file_endings,"run20",False,is_bias,is_paft) # 7x upsample - no improvement to robustness 
# process_results(file_endings,"run21",False,is_bias,is_paft) # 9x upsample - no improvement to robustness
# process_results(file_endings,"run22",False,is_bias,is_paft) # halved bias set - improvement to HANS - SHOW IN ANALYSIS?
# process_results(file_endings,"run26",False,is_bias,is_paft) # split epoch training (1-2) - improvement to HANS - SHOW IN ANALYSIS?



############################ SNLI - CUSTOM SAMPLING
# is_bias = False
# is_paft = False
# process_results(file_endings,"run28",True,is_bias,is_paft) # 10k samples in retrain, 8k unbiased, 2k biased - improvement to robustness
# process_results(file_endings,"run29",True,is_bias,is_paft) # 18k samples in retrain, 16k unbiased, 2k biased - similar to 28, but not as good
# process_results(file_endings,"run30",True,is_bias,is_paft) # 10k samples in retrain, 9k unbiased, 1k biased - similar to 28, but better for HANS - BEST?
# process_results(file_endings,"run31",True,is_bias,is_paft) # 30k samples in bias model, 16k samples in retrain, 15k unbiased, 1k biased - brings performance of SNLI close to SNLI-Hard, HANS not good though
#would like to test 31 again but with 9k/1k split



############################ MNLI - CUSTOM SAMPLING
# is_bias = False
# is_paft = False
# process_results(file_endings,"run32",False,is_bias,is_paft) # 10k samples for bias model, 10k samples in retrain, 8k unbiased, 2k biased - good for HANS not SNLI-Hard
# process_results(file_endings,"run33",False,is_bias,is_paft) # 10k samples for bias model, 18k samples in retrain, 16k unbiased, 2k biased - very similar to 32, not quite as good
# process_results(file_endings,"run34",False,is_bias,is_paft)




############################ SNLI - 1 to 6
# is_bias = True
# is_paft = False
process_results(file_endings,"run1",True,is_bias,is_paft)
process_results(file_endings,"run2",True,is_bias,is_paft)
process_results(file_endings,"run3",True,is_bias,is_paft)
process_results(file_endings,"run4",True,is_bias,is_paft) # really good: SNLI-Hard = 79.95, HANS = 73.90
process_results(file_endings,"run5",True,is_bias,is_paft) 
process_results(file_endings,"run6",True,is_bias,is_paft) # really good: SNLI-Hard = 81.97, HANS = 72.35




############################ SNLI - 7 to 11 - BALANCING
# is_bias = False
# is_paft = False
process_results(file_endings,"run7",True,is_bias,is_paft) # solid: SNLI-Hard = 81.97, HANS = 70.82
# process_results(file_endings,"run8",True,is_bias,is_paft)
# process_results(file_endings,"run9",True,is_bias,is_paft)
# process_results(file_endings,"run10",True,is_bias,is_paft)
# process_results(file_endings,"run11",True,is_bias,is_paft) 




############################ SNLI - 35 to 41 - baselines for extensions
# is_bias = False
# is_paft = False
# process_results(file_endings,"run35",True,is_bias,is_paft)
# process_results(file_endings,"run36",True,is_bias,is_paft)
# process_results(file_endings,"run37",True,is_bias,is_paft)
# process_results(file_endings,"run38",True,is_bias,is_paft)
# process_results(file_endings,"run39",True,is_bias,is_paft) 
# process_results(file_endings,"run40",True,is_bias,is_paft)
# process_results(file_endings,"run41",True,is_bias,is_paft)




############################ MNLI - 50 to 54
# is_bias = False
# is_paft = False
# process_results(file_endings,"run50",False,is_bias,is_paft)
# process_results(file_endings,"run51",False,is_bias,is_paft)
# process_results(file_endings,"run52",False,is_bias,is_paft)
# process_results(file_endings,"run53",False,is_bias,is_paft)
# 54 doesn't exist





############################ SNLI - 55 to 56 - bias model 2e-05
# is_bias = False
# is_paft = False
# process_results(file_endings,"run55",True,is_bias,is_paft) # improvements to SNLI-Hard, but not quite as using LR=2e-04
# process_results(file_endings,"run56",True,is_bias,is_paft) # improvements to SNLI-Hard, but not quite as using LR=2e-04






############################ SNLI + MNLI Baselines - 57 and 58
# is_bias = False
# is_paft = False
# process_results(file_endings,"run57",True,is_bias,is_paft)
# process_results(file_endings,"run58",False,is_bias,is_paft)
# process_results(file_endings,"run71",True,is_bias,is_paft) 




############################ MNLI - 59 and 65 - improvement techniques
# process_results(file_endings,"run59",False,is_bias,is_paft)
# process_results(file_endings,"run60",False,is_bias,is_paft)
# process_results(file_endings,"run61",False,is_bias,is_paft)
# process_results(file_endings,"run62",False,is_bias,is_paft)
# process_results(file_endings,"run63",False,is_bias,is_paft)
# process_results(file_endings,"run64",False,is_bias,is_paft)
# process_results(file_endings,"run65",False,is_bias,is_paft) # DIDNT WORK - RERUN





############################ SNLI - 66 to 70 - run 6 variations - all pretty shit
process_results(file_endings,"run66",True,is_bias,is_paft) # 
process_results(file_endings,"run67",True,is_bias,is_paft) # 
process_results(file_endings,"run68",True,is_bias,is_paft) # 
process_results(file_endings,"run69",True,is_bias,is_paft) # 
process_results(file_endings,"run70",True,is_bias,is_paft) # 




############################ SNLI - 71 to 84 - proportion method
# process_results(file_endings,"run71",True,is_bias,is_paft) # 
# process_results(file_endings,"run72",True,is_bias,is_paft) # 
# 73 failed # dont care though really, not useful
# process_results(file_endings,"run74",True,is_bias,is_paft) # 
# process_results(file_endings,"run75",True,is_bias,is_paft) # 
# process_results(file_endings,"run76",True,is_bias,is_paft) # 
# process_results(file_endings,"run77",True,is_bias,is_paft) # 
# process_results(file_endings,"run78",True,is_bias,is_paft) # 
# process_results(file_endings,"run79",True,is_bias,is_paft) # 
# process_results(file_endings,"run80",True,is_bias,is_paft) # 
# process_results(file_endings,"run81",True,is_bias,is_paft) # 
# 82 failed
# process_results(file_endings,"run83",True,is_bias,is_paft) # 
# 84 failed




############################ SNLI - 82 + 85 to 96 - SNLI hyp-only results
# process_results(file_endings,"run82",True,is_bias,is_paft) # 
# process_results(file_endings,"run85",True,is_bias,is_paft) # 
# process_results(file_endings,"run86",True,is_bias,is_paft) # 
# process_results(file_endings,"run87",True,is_bias,is_paft) # 
# process_results(file_endings,"run88",True,is_bias,is_paft) # 
# process_results(file_endings,"run89",True,is_bias,is_paft) # 
# process_results(file_endings,"run90",True,is_bias,is_paft) # 
# process_results(file_endings,"run91",True,is_bias,is_paft) # 
# process_results(file_endings,"run92",True,is_bias,is_paft) # 
# process_results(file_endings,"run93",True,is_bias,is_paft) # 
# process_results(file_endings,"run94",True,is_bias,is_paft) # love
# process_results(file_endings,"run95",True,is_bias,is_paft) # these
# process_results(file_endings,"run96",True,is_bias,is_paft) # 




############################ MNLI - 97 to 100 - modified bias model for hyp-only
# process_results(file_endings,"run97",False,is_bias,is_paft) #
# process_results(file_endings,"run98",False,is_bias,is_paft) #
# process_results(file_endings,"run99",False,is_bias,is_paft) #
# same as run99 because I commented file lol, no HANS available though



############################ SNLI - 101 to 108 - SNLI hyp-only results (gentle resampling mainly + modeified bias model)
# process_results(file_endings,"run101",True,is_bias,is_paft) # 
# process_results(file_endings,"run102",True,is_bias,is_paft) # 
# process_results(file_endings,"run103",True,is_bias,is_paft) # 
# process_results(file_endings,"run104",True,is_bias,is_paft) # 
# process_results(file_endings,"run105",True,is_bias,is_paft) # 
# process_results(file_endings,"run106",True,is_bias,is_paft) # 
# process_results(file_endings,"run107",True,is_bias,is_paft) # 
# process_results(file_endings,"run108",True,is_bias,is_paft) # 

#  should rerun 94 x3 different seeds

############################ MNLI - 97 to 100 - modified bias model for hyp-only
# process_results(file_endings,"run109",False,is_bias,is_paft) # excellent (80.99 SH, 73.2 HANS)
# process_results(file_endings,"run110",False,is_bias,is_paft) # excellent for HANS (79.42, 73.73)
# process_results(file_endings,"run111",False,is_bias,is_paft) # solid (80.42 SH, 72.43 HANS)
# process_results(file_endings,"run112",False,is_bias,is_paft) # very similar to 111 but actually seems to rely on syntactic heuristics (good HANS but bad non-ent)
#  overall runs 109 to 112 use 30k training samples for bias model, and use different splits: 1k/9k, 3k/9k, 5k/9k, 7k/9k
#  in this case, reducing the bias set more, proves to have a greater impact on robustness
#  increasing the bias set size while maintaining the unbiased set size seems to be harmful to HANS even if the overall accuracy is still high
#  note that 9k is the size of the unbiased set, so max variety for the unbiased set is best

# process_results(file_endings,"run113",False,is_bias,is_paft) # 
# process_results(file_endings,"run114",False,is_bias,is_paft) # 
#  these two use 7k/9k*2 or *3, and again shows that upsampling has limits.

#  should now test 109 with upsampling x2. Run over a few different seeds to get final scores. 

############################ MNLI - 115 to 121 - gentle resampling
# process_results(file_endings,"run115",False,is_bias,is_paft) # probably best (best SH at 81.05, SNLI at 88.13, and MNLI-MM at 88.28)
# process_results(file_endings,"run116",False,is_bias,is_paft) # not too bad compared to 115 (80.83 SH, 87.55 SNLI, MNLI-MM at 88.13) - might just be variation differences
# process_results(file_endings,"run117",False,is_bias,is_paft) # very comparable to 116 - again could be variation
#  these runs are the best for MNLI gentle resampling. Run118 is the same as 117 but with 2x, and this worsens performance, so we can show that upsampling the unbiased set is not as effective as downsampling the biased set

# process_results(file_endings,"run118",False,is_bias,is_paft) # 
# process_results(file_endings,"run119",False,is_bias,is_paft) # 
# process_results(file_endings,"run120",False,is_bias,is_paft) # 
# process_results(file_endings,"run121",False,is_bias,is_paft) # 
#  these runs are all shite except 120 which is same as 116 but balances classes


# should probably run 115, 116 and 117 each over a few different seeds (this is 3+2+2=7 experiments)


# we can show that upsampling more and more has a limit to how much it helps, show by validation and test results
# e.g run121 upsamples 3x and does more harm than good
# we can use this to say, it could be the case that we need moere variety in the training data such as a greater number of training samples
# this would expose the model to more diverse data to train on which might prevent it from still overfitting to spurious correlations





#  pattern: often see a trade-off between HANS and SNLI-Hard. If we opt for a more conservative upsampling/downsampling approach, SNLI-hard benefits but HANS suffers.


# seemingly sensitive to hyperparams - could be due to the fact thta upsampling is so dramatic (inc / dec by 1 means thousand of samples)




# OKay NOW we're talking
# run 94 - great for SNLI-Hard (83%) and others except HANS, which is 71.42
# run 95 - solid for SNLI-Hard (82.3%) and HANS (72.45%) but doesn't improve cross-domain robustness - include in analysis but not final table
# run 96 - (is basically run94 but using 30k trained bias model with exclusive split)

# Useful Results:
# run6 - best ER result so far
# run7 - matches run6 for SNLI-Hard but has lower HANS.
# run72 - normal ER, upsample = 2x - better than run6 for SNLI-Hard but lower HANS
# run74 - bias model 2 eps/2e-04, final model 3 eps/2e-05, upsample = 1x, multipler = 0.9, dec_biased - downsampling with minimal upsampling - best SNLI-Hard ever, bad HANS
# run75 - same as run74, but with increasing the unbiased set by amount biased set is decreased - lower SNLI-hard than run74, but much better HANS
# run81 - same as run80, but with inc (so 0.7, dec, inc, balance) - surprisingly decent HANS (71.9), not so much for SHard

# Potentially Useful Baselines / Ablations:
# run4 - somehow great for HANS with hyp-only classifier. Kind of counterintuitive but has highest HANS result seen.
# run17 - acts as ER baseline for SNLI
# run35, 36 - exclusive splits (5k,5k) no custom range - 2/3 epochs, so can act as some kind of baseline or maybe ablation
# runs55, 56 - maybe useful? probably not, but could be seen as ER baselines as well with 2 epoch training (upsample=4,5)
# run66 - run6 with bias model LR=2e-05
# run67 - run6 with bias model LR=2e-05, and balance classes
# run68 - run6 with both models LR=2e-04
# run71 - run6 with bias model LR=2e-05 and 3 epochs
# run76 - same as run75, but with multiplier = 0.7, almost identical results to run75
# run77 - same as run75, but with multiplier = 0.7, almost identical results to run75
# run78 - same as run74, but with multiplier = 0.5, almost identical results to run77 ???
# run79 - same as run78, but with inc - decent SNLI-Hard and HANS tbf


# Not likely useful:
# runs28-30 - custom range (upsampling + downsampling) runs, could be custom range baselines. Not too useful imo.
# runs37-41 - custom range (upsampling + downsampling) runs, again not useful really
# run69 - run6 with both models LR=2e-04 plus balance classes (i.e. 68 with balance classes)
# run70 - run6 with both models LR=2e-04 plus balance classes, plus final train epochs is 2 (i.e. 69 with two epochs for final train)
# run72 - not even sure tbh doesn't seem useful
# run80 - same as run77, but with balance class
# run82 - run6 with LR=2e-06 - forget it babe



# runs 85-96:
# - run7 : Try with less upsampling (e.g. 4 or 5)
# - run6 : Try more and less upsampling (e.g. 3, 5)
# - run6 : Try linear LR scheduler
# - run6 : Try seed 2
# - run6 : Try 12k / 2k split
# - run6 : Try balance classes
# - run74 : Try upsample 2x
# - run74 : Try 0.8
# - run75 : Try balance 
# - run81 : Try 0.8


# Trying next:

