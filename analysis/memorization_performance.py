import os
import argparse
import pickle as pkl
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score
import math
import numpy as np


def get_roc_metrics(real_preds, sample_preds):
    real_preds =  [element for element in real_preds if not math.isnan(element)]
    sample_preds = [element for element in sample_preds if not math.isnan(element)]

    fpr, tpr, thresholds = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)


if __name__ == '__main__':
    base_result_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-{}/lr_ratio_threshold_results.json"
    crit_result_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-{}-{}/crit/group_out.json"
    sizes = ["160m", "410m", "1b", "2.8b", "6.9b", "12b--min_k"]

    # Load perplexities
    lls = {size: {} for size in sizes}
    for size in sizes:
        result_path = base_result_path.format(size)
        with open(result_path, 'r') as f:
            result = json.load(f)
        for i, entry in enumerate(result["nonmember_meta"]):
            lls[size][tuple(entry)] = result["nonmember_lls"][i]
        for i, entry in enumerate(result["member_meta"]):
            lls[size][tuple(entry)] = result["member_lls"][i]
    
    # Find a threshold
    best_nonmember_lls = lls[sizes[-1]].values()
    best_member_lls = lls[sizes[-1]].values()
    threshold = None
    fpr, tpr, thresholds, roc_auc = get_roc_metrics(best_nonmember_lls,  best_member_lls)
    print(fpr[:10])
    print(fpr[-10:])
    print(thresholds[:10])
    print(thresholds[-10:])
    for i, rate in enumerate(fpr):
        if rate > 0.05:
            threshold = thresholds[i-1]
    
    # Determine memorized label v.s. not memorized label
    labels = {}
    for entry, score in lls[sizes[-1]].items():
        print(score)
    for entry, score in lls[sizes[-1]].items():
        labels[tuple(entry)] = score > threshold
    print("Memorized rate for the largest model:{}".format(np.mean(list(labels.values()))))
    
    # Collecting Evaluation
    evaluations = []
    for size, size_result in lls.items():
        predictions = {}
        for entry, score in size_result.items():
            predictions[tuple(entry)] = score > threshold
        precision = precision_score(list(labels.values()), list(predictions.values()))
        recall = recall_score(list(labels.values()), list(predictions.values()))
        f1 = 2 * (precision * recall) / (precision + recall)
        print("Precision for size ({}): {}".format(size, precision))
        print("Recall for size ({}): {}".format(size, recall))
        print("F1 for size ({}): {}".format(size, f1))
        print("=========================================================")


        