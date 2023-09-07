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
    plt.plot(fpr, tpr, label=name, roc_auc="{roc_auc:.3f}", color='#1b9e77')
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
    parser.add_argument('--check_map_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out/check_map.pkl")
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-miaresults/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/")
    parser.add_argument('--top_k', type=int, required=True)    

    args = parser.parse_args()

    assert os.path.exists(args.result_path), args.result_path
    assert os.path.exists(args.member_info_path), args.member_info_path
    assert os.path.exists(args.nonmember_info_path), args.nonmember_info_path
    assert os.path.exists(args.check_map_path), args.check_map_path

    with open(args.result_path, 'r') as f:
        result = json.load(f)
    with open(args.member_info_path, 'r') as f:
        member_info = json.load(f)
    with open(args.nonmember_info_path, 'r') as f:
        nonmember_info= json.load(f)
    with open(args.check_map_path, 'rb') as f:
        check_map= pkl.load(f)
        
    print("# of results: {}".format(len(result['raw_results'])))
    print("# of samples: {}".format(result['info']['n_samples']))
    print("Metrics on document level:")
    print(result['metrics'])

    group_results_members = {}
    group_results_nonmembers = {}
    for i, entry in enumerate(result['raw_results']):
        group_member = check_map[member_info[i]["title"]]['group']
        if group_member not in group_results_members:
            group_results_members[group_member] = []
        group_results_members[group_member].append(entry['member_crit'])
        group_nonmember = check_map[nonmember_info[i]["title"]]['group']
        if group_nonmember not in group_results_nonmembers:
            group_results_nonmembers[group_nonmember] = []
        group_results_nonmembers[group_nonmember].append(entry['nonmember_crit'])

    print("# of member groups: {}".format(len(group_results_members)))
    print("# of nonmember groups: {}".format(len(group_results_nonmembers)))
    print("Average # document in member group: {}/{}".format(np.mean([len(members) for _, members in group_results_members.items()]), np.std([len(members) for _, members in group_results_members.items()])))
    print("Average # document in nonmember group: {}/{}".format(np.mean([len(members) for _, members in group_results_nonmembers.items()]), np.std([len(members) for _, members in group_results_nonmembers.items()])))


