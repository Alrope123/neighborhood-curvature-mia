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

def get_correlation(evaluations, key1, key2="performance"):
    return np.corrcoef([evaluation[key1] for evaluation in evaluations], [evaluation[key2] for evaluation in evaluations])[0, 1]

if __name__ == '__main__':
    base_result_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-{}/lr_ratio_threshold_results.json"
    crit_result_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-{}-EleutherAI_pythia-{}/crit/group_output.json"
    sizes = ["14m--min_k", "70m--min_k", "160m", "410m", "1b", "2.8b", "6.9b", "12b--min_k"]

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
    
    evaluations = []
    for target_size in reversed(sizes):
        # Find a threshold
        best_nonmember_lls = lls[target_size].values()
        best_member_lls = lls[target_size].values()
        threshold = None
        fpr, tpr, thresholds, roc_auc = get_roc_metrics(best_nonmember_lls,  best_member_lls)
        for i, rate in enumerate(fpr):
            if rate > 0.05:
                threshold = thresholds[i-1]
                break

        # Determine memorized label v.s. not memorized label
        labels = {}
        for entry, score in lls[target_size].items():
            labels[tuple(entry)] = score > threshold
        print("Memorized rate for the largest model:{}".format(np.mean(list(labels.values()))))
        
        # Collecting Evaluation
        for ref_size, size_result in lls.items():
            correct_labels = []
            predictions = []
            for entry, score in size_result.items():
                predictions.append(score > threshold)
                correct_labels.append(labels[tuple(entry)]) 

            precision = precision_score(correct_labels, predictions)
            recall = recall_score(correct_labels, predictions)
            f1 = 2 * (precision * recall) / (precision + recall)
            peformance_path = crit_result_path.format(target_size, ref_size)
            with open(peformance_path, 'r') as f:
                performance = json.load(f)["average"]["max"]["100"]["MIA"]["best"]["ROC AUC"]
            
            evaluations.append({
                "target_size": target_size,
                "ref_size": ref_size,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "performance": performance
            })
        
    correlations = {
        "evaluations": evaluations,
        "precision_correlation": get_correlation(evaluations, "precision"),
        "recall_correlation": get_correlation(evaluations, "precision"),
        "f1_correlation": get_correlation(evaluations, "precision")
    }

    output_dir = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_analysis/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(result_path, "memorizaiton.json"), 'r') as f:
        json.dump(correlations)
        