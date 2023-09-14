import os
import argparse
import pickle as pkl
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import math


def get_roc_metrics(real_preds, sample_preds):
    real_preds =  [element for element in real_preds if not math.isnan(element)]
    sample_preds = [element for element in sample_preds if not math.isnan(element)]

    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def save_roc_curves(name, fpr, tpr, roc_auc, SAVE_FOLDER=None):
    # first, clear plt
    plt.clf()
    plt.plot(fpr, tpr, label=f"{name}, roc_auc={roc_auc:.3f}", color='#1b9e77')
    # print roc_auc for this experiment
    print(f"{name} roc_auc: {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/lr_ratio_threshold_results.json")
    parser.add_argument('--member_info_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/wikipedia_member.json")
    parser.add_argument('--nonmember_info_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/wikipedia_nonmember.json")
    parser.add_argument('--membership_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl")
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/")
    parser.add_argument('--top_k', type=int, default=50)    

    args = parser.parse_args()

    assert os.path.exists(args.result_path), args.result_path
    assert os.path.exists(args.member_info_path), args.member_info_path
    assert os.path.exists(args.nonmember_info_path), args.nonmember_info_path
    assert os.path.exists(args.membership_path), args.membership_path

    with open(args.result_path, 'r') as f:
        result = json.load(f)
    with open(args.member_info_path, 'r') as f:
        member_info = json.load(f)
    with open(args.nonmember_info_path, 'r') as f:
        nonmember_info= json.load(f)
    with open(args.membership_path, 'rb') as f:
        group_to_documents= pkl.load(f)
        
    print("# of results: {}".format(len(result['raw_results'])))
    print("# of samples: {}".format(result['info']['n_samples']))
    print("Metrics on document level:")
    print(result['metrics']['roc_auc'])

    info_to_group = {}
    for group, infos in group_to_documents.items():
        for filename, i, _ in infos['documents']:
            info_to_group[(filename, i)] = group

    group_results_members = {}
    group_results_nonmembers = {}
    for i, entry in enumerate(result['raw_results']):
        group_member = info_to_group[member_info[i]]
        if group_member not in group_results_members:
            group_results_members[group_member] = []
        group_results_members[group_member].append(entry['member_crit'])
        group_nonmember = info_to_group[nonmember_info[i]]
        if group_nonmember not in group_results_nonmembers:
            group_results_nonmembers[group_nonmember] = []
        group_results_nonmembers[group_nonmember].append(entry['nonmember_crit'])

    print("# of member groups: {}".format(len(group_results_members)))
    print("# of nonmember groups: {}".format(len(group_results_nonmembers)))
    print("Average # document in member group: {}/{}".format(np.mean([len(members) for _, members in group_results_members.items()]), np.std([len(members) for _, members in group_results_members.items()])))
    print("Average # document in nonmember group: {}/{}".format(np.mean([len(members) for _, members in group_results_nonmembers.items()]), np.std([len(members) for _, members in group_results_nonmembers.items()])))

    best_k = None
    best_fpr = None
    best_tpr = None
    best_auc = -1
    all_results = {}
    for top_k in [1, 3, 5, 10, 30, 50]:
        cur_member_predictions = []
        cur_nonmember_predictions = []
        for group, predictions in group_results_members.items():
            cur_member_predictions.append(np.mean(sorted(predictions, reverse=False)[:top_k]))
        for group, predictions in group_results_nonmembers.items():
            cur_nonmember_predictions.append(np.mean(sorted(predictions, reverse=False)[:top_k]))
        fpr, tpr, roc_auc = get_roc_metrics(cur_member_predictions, cur_nonmember_predictions)
        all_results[top_k] = (fpr, tpr, roc_auc)
        if roc_auc > best_auc:
            best_k = top_k
            best_auc = roc_auc
            best_fpr = fpr
            best_tpr = tpr
    
    output = {
        "top_k": best_k,
        "ROC_AUC": best_auc,
        # "fpr": best_fpr,
        # "tpr": best_tpr
    }
    all_results["best"] = output
    print("Final results")
    print(output['top_k'])
    print(output['ROC_AUC'])

    SAVE_FOLDER = args.out_dir

    with open(os.path.join(SAVE_FOLDER, "group_output.json"), 'w') as f:
        json.dump(all_results, f)

    save_roc_curves("neo-3b", best_fpr, best_tpr, best_auc, SAVE_FOLDER)
