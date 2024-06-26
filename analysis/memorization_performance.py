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
    rates = [0.05, 0.15, 0.5, 0.85, 0.95]

    # Load perplexities
    lls = {size: {} for size in sizes}
    lls_nonmember = {size: {} for size in sizes}
    for size in sizes:
        result_path = base_result_path.format(size)
        with open(result_path, 'r') as f:
            result = json.load(f)
        for i, entry in enumerate(result["nonmember_meta"]):
            lls_nonmember[size][tuple(entry)] = result["nonmember_lls"][i]
        for i, entry in enumerate(result["member_meta"]):
            lls[size][tuple(entry)] = result["member_lls"][i]
    
    memorizations = {}
    for cur_rate in rates:
        all_evals = {}
        for target_size in reversed(sizes):
            # Find a threshold
            best_nonmember_lls = lls_nonmember[target_size].values()
            best_member_lls = lls[target_size].values()
            threshold = None
            fpr, tpr, thresholds, roc_auc = get_roc_metrics(best_nonmember_lls,  best_member_lls)
            for i, rate in enumerate(fpr):
                if rate > cur_rate:
                    threshold = thresholds[i-1]
                    break

            # Determine memorized label v.s. not memorized label
            labels = {}
            label_scores = {}
            for entry, score in lls[target_size].items():
                labels[tuple(entry)] = score > threshold
                label_scores[tuple(entry)] = score
            memorization_rate = np.mean(list(labels.values()))
            print("Memorized rate for Eleuther_pythia-{}:{}".format(target_size, memorization_rate))
            
            evaluations = []
            # Collecting Evaluation
            for ref_size in reversed(sizes):
                if sizes.index(ref_size) >= sizes.index(target_size):
                    continue
                size_result = lls[ref_size]
                correct_labels = []
                correct_label_scores = []
                predictions = []
                prediciton_scores = []
                for entry, score in size_result.items():
                    predictions.append(score > threshold)
                    correct_labels.append(labels[tuple(entry)])
                    prediciton_scores.append(score)
                    correct_label_scores.append(label_scores[tuple(entry)]) 

                precision = precision_score(correct_labels, predictions)
                recall = recall_score(correct_labels, predictions)
                f1 = 2 * (precision * recall) / (precision + recall)
                pure_ratio = sum(predictions) / sum(correct_labels)
                confidence = np.mean([score for i, score in enumerate(prediciton_scores) if predictions[i]]) - np.mean([score for i, score in enumerate(correct_label_scores) if correct_labels[i]])
                correct_confidence = np.mean([score for i, score in enumerate(prediciton_scores) if correct_labels[i]]) - np.mean([score for i, score in enumerate(correct_label_scores) if correct_labels[i]])
                peformance_path = crit_result_path.format(target_size, ref_size)
                with open(peformance_path, 'r') as f:
                    # performance = json.load(f)["average"]["max"]["100"]["MIA"]["best"]["ROC AUC"]
                    performance = json.load(f)["average"]["max"]["100"]["ROC AUC Individual"]
                
                evaluations.append({
                    "ref_size": ref_size,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ratio": pure_ratio,
                    "confidence": confidence,
                    "correct_confidence": correct_confidence,
                    "performance": performance
                })
            
            correlations = {
                "precision_correlation": get_correlation(evaluations, "precision") if len(evaluations) > 1 else 0,
                "recall_correlation": get_correlation(evaluations, "recall") if len(evaluations) > 1 else 0,
                "f1_correlation": get_correlation(evaluations, "f1") if len(evaluations) > 1 else 0,
                "ratio_correlation": get_correlation(evaluations, "ratio") if len(evaluations) > 1 else 0,
                "confidence_correlation": get_correlation(evaluations, "confidence") if len(evaluations) > 1 else 0,
                "correct_confidence_correlation": get_correlation(evaluations, "correct_confidence") if len(evaluations) > 1 else 0
            }
            
            all_evals[target_size] = {
                "threshold": threshold,
                "memorization_rate": memorization_rate,
                "evaluations": evaluations,
                "correlations": correlations
            }
        memorizations[cur_rate] = {
            "average_correlations": {
                key: np.mean([all_evals[s]["correlations"]["{}_correlation".format(key)] for s in all_evals.keys()]) for key in ["precision", "recall", "f1", "ratio", "confidence", "correct_confidence"]
            },
            "all_evals": all_evals,
        }

    output_dir = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_analysis/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "memorization.json"), 'w') as f:
        json.dump(memorizations, f, indent=4)
        