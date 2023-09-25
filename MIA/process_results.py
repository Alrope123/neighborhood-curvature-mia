import os
import argparse
import pickle as pkl
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import math
import random

def generate_topk_array(n):
    base_list = [1, 3, 5]
    output_list = []
    while True:
        for k in base_list:
            if k > n:
                return output_list
            else:
                output_list.append(k)
        base_list = [k * 10 for k in base_list]


def make_dicts_equal(dict1, dict2):
    length_diff = len(dict1) - len(dict2)

    if length_diff > 0:
        keys_to_remove = random.sample(list(dict1.keys()), length_diff)
        for key in keys_to_remove:
            del dict1[key]
    elif length_diff < 0:
        keys_to_remove = random.sample(list(dict2.keys()), abs(length_diff))
        for key in keys_to_remove:
            del dict2[key]

    return dict1, dict2


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
    plt.savefig(f"{SAVE_FOLDER}/roc_curves_{name}.png")


def calculate_group_loss(losses, aggregated_method, direction, top_s):
    if aggregated_method == 'mean':
        return np.mean(sorted(losses, reverse=direction == "min")[:top_s])
    else:
        raise NotImplementedError("The aggregation method is not supported.")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(members, nonmembers, name, bin_width, SAVE_FOLDER):
    # assert len(members) == len(nonmembers)
    # first, clear plt
    plt.clf()
    bins_members = int(max(abs(max(members) - min(members)), abs(min(members) - max(members))) / bin_width)
    if bins_members == 0:
        bins_members = 1
    bins_nonmembers = int(max(abs(max(nonmembers) - min(nonmembers)), abs(min(nonmembers) - max(nonmembers))) / bin_width)
    if bins_nonmembers == 0:
        bins_nonmembers = 1
    
    # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
    plt.figure(figsize=(10, 6))
    plt.hist(members, alpha=0.5, bins=bins_members, label='member')
    plt.hist(nonmembers, alpha=0.5, bins=bins_nonmembers, label='non-member')
    plt.xlabel("Likelihood")
    plt.ylabel('Count')
    plt.title(name)
    plt.legend(loc='upper right')
    plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{name}.png")
    print(f"Plotting at {SAVE_FOLDER}/ll_histograms_{name}.png")


def save_cmap(data, ticks, name):
    # Creating the 2D table plot
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='Blues')

    # Masking nan values to show as white
    # cax.set_bad(color='white')

    # Creating a colorbar for reference
    fig.colorbar(cax)

    # Labeling the axes
    ax.set_xlabel('Top S')
    ax.set_ylabel('K')

    # Setting custom tick labels
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_yticklabels([str(tick) for tick in ticks])

    # Showing the plot
    plt.savefig(f"{SAVE_FOLDER}/cmap_{name}.png")
    print(f"Plotting at {SAVE_FOLDER}/cmap_{name}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/lr_ratio_threshold_results.json")
    parser.add_argument('--membership_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl")
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/")
    parser.add_argument('--key', type=str, default="crit", choices=["crit", "lls", "bff"])
    parser.add_argument('--top_k', type=int, default=50)    

    args = parser.parse_args()

    assert os.path.exists(args.result_path), args.result_path
    assert os.path.exists(args.membership_path), args.membership_path

    with open(args.result_path, 'r') as f:
        result = json.load(f)
    with open(args.membership_path, 'rb') as f:
        group_to_documents= pkl.load(f)
    
    SAVE_FOLDER = args.out_dir
    max_top_k = args.top_k
    random.seed(2023)
    np.random.seed(2023)

    print("# of samples: {}".format(result['n_samples']))

    print("Metrics on document level:")
    nonmember_key = f"nonmember_{args.key}"
    member_key = f"member_{args.key}"

    nonmember_predictions = result[nonmember_key]
    member_predictions = result[member_key]
    compare_length = min(len(nonmember_predictions), len(member_predictions))
    if len(nonmember_predictions) > len(member_predictions):
        nonmember_predictions = np.random.choice(nonmember_predictions, len(member_predictions), replace=False)
    elif len(member_predictions) > len(nonmember_predictions):
        member_predictions = np.random.choice(member_predictions, len(nonmember_predictions), replace=False)
    fpr, tpr, individual_roc_auc = get_roc_metrics(nonmember_predictions, member_predictions)
    # Draw log likehood histogram on individual documents
    save_ll_histograms(member_predictions, nonmember_predictions,f"individual_with_{args.key}", 0.05, SAVE_FOLDER)
    print("Individual AUC-ROC with {}: {}".format(args.key, individual_roc_auc))
    save_roc_curves("Individual_with_{}".format(args.key), fpr, tpr, individual_roc_auc, SAVE_FOLDER)

    info_to_group = {}
    for group, infos in group_to_documents.items():
        for filename, i, _, is_member in infos['documents']:
            info_to_group[(filename, i)] = group

    group_results_members = {}
    group_results_nonmembers = {}
    for i, entry in enumerate(result[member_key]):
        group_member = info_to_group[tuple(result['member_meta'][i])]
        assert group_to_documents[group_member]['group_is_member']
        if group_member not in group_results_members:
            group_results_members[group_member] = []
        group_results_members[group_member].append(entry)
    for i, entry in enumerate(result[nonmember_key]):
        group_nonmember = info_to_group[tuple(result['nonmember_meta'][i])]
        assert not group_to_documents[group_nonmember]['group_is_member']
        if group_nonmember not in group_results_nonmembers:
            group_results_nonmembers[group_nonmember] = []
        group_results_nonmembers[group_nonmember].append(entry)
    print("# of member groups before filtering: {}".format(len(group_results_members)))
    print("# of nonmember groups before filtering: {}".format(len(group_results_nonmembers)))

    # Selecting qualified groups
    qualified_group_results_members = {}
    qualified_group_results_nonmembers = {}
    for group, predictions in group_results_members.items():
        if len(predictions) >= max_top_k:
            random.shuffle(predictions)
            qualified_group_results_members[group] = predictions[:max_top_k]
    for group, predictions in group_results_nonmembers.items():
        if len(predictions) >= max_top_k:
            random.shuffle(predictions)
            qualified_group_results_nonmembers[group] = predictions[:max_top_k]

    group_results_members, group_results_nonmembers = make_dicts_equal(qualified_group_results_members, qualified_group_results_nonmembers)
    print("# of member groups: {}".format(len(group_results_members)))
    print("# of nonmember groups: {}".format(len(group_results_nonmembers)))
    print("Average # document in member group: {}/{}".format(np.mean([len(members) for _, members in group_results_members.items()]), np.std([len(members) for _, members in group_results_members.items()])))
    print("Average # document in nonmember group: {}/{}".format(np.mean([len(members) for _, members in group_results_nonmembers.items()]), np.std([len(members) for _, members in group_results_nonmembers.items()])))

    ROOT_SAVE_FOLDER = SAVE_FOLDER
    for aggregated_method in ['mean']:
        for direction in ['max', "min"]:
            method = f"{aggregated_method}-{direction}"
            SAVE_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "{}-{}".format(args.key, method))
            if not os.path.exists(SAVE_FOLDER):
                os.mkdir(SAVE_FOLDER)
            
            all_results = {}
            for k in generate_topk_array(max_top_k):
                # Randomly k documents from each group 
                cur_group_results_members = {}
                cur_group_results_nonmembers = {}
                for group, predictions in group_results_members.items():
                    random.shuffle(predictions)
                    cur_group_results_members[group] = predictions[:k]
                for group, predictions in group_results_nonmembers.items():
                    random.shuffle(predictions)
                    cur_group_results_nonmembers[group] = predictions[:k]

                best_s = None
                best_fpr = None
                best_tpr = None
                best_auc = -1

                for top_s in generate_topk_array(k):
                    cur_member_predictions = []
                    cur_nonmember_predictions = []

                    cur_member_individual_predictions = []
                    cur_nonmember_individual_predictions = []
                    cur_member_individual_predictions_with_set_label = []
                    cur_nonmember_individual_predictions_with_set_label = []

                    if args.key != 'bff':
                        for group, predictions in cur_group_results_members.items():
                            group_loss = calculate_group_loss(predictions, aggregated_method, direction, top_s)
                            cur_member_predictions.append(group_loss)
                            cur_member_individual_predictions.extend(predictions)
                            cur_member_individual_predictions_with_set_label.extend([group_loss] * len(predictions))
                        for group, predictions in cur_group_results_nonmembers.items():
                            group_loss = calculate_group_loss(predictions, aggregated_method, direction, top_s)
                            cur_nonmember_predictions.append(group_loss)
                            cur_nonmember_individual_predictions.extend(predictions)
                            cur_nonmember_individual_predictions_with_set_label.extend([group_loss] * len(predictions))
                    else:
                        for group, predictions in cur_group_results_members.items():
                            scores = [score for (_, _, score, _) in group_to_documents[group]['documents']]
                            group_loss = calculate_group_loss(scores, aggregated_method, direction, top_s)
                            cur_member_predictions.append(group_loss)
                            cur_member_individual_predictions.extend(predictions)
                            cur_member_individual_predictions_with_set_label.extend([group_loss] * len(predictions))
                        for group, predictions in cur_group_results_nonmembers.items():
                            scores = [score for (_, _, score, _) in group_to_documents[group]['documents']]
                            group_loss = calculate_group_loss(scores, aggregated_method, direction, top_s)
                            cur_nonmember_predictions.append(group_loss)
                            cur_nonmember_individual_predictions.extend(predictions)
                            cur_nonmember_individual_predictions_with_set_label.extend([group_loss] * len(predictions))
                    
                    random.shuffle(cur_member_predictions)
                    random.shuffle(cur_nonmember_predictions)
                    sample_size = min([len(cur_member_predictions), len(cur_nonmember_predictions)])
                    cur_member_predictions = cur_member_predictions[:sample_size]
                    cur_nonmember_predictions = cur_nonmember_predictions[:sample_size]
                    fpr, tpr, roc_auc = get_roc_metrics(cur_nonmember_predictions, cur_member_predictions)
                    # save_ll_histograms(cur_member_predictions, cur_nonmember_predictions, "group_top-k={}".format(top_k), 0.02, SAVE_FOLDER)
                    # save_roc_curves("group_top-k={}".format(top_k), fpr, tpr, roc_auc, SAVE_FOLDER)
                    
                    assert len(cur_member_individual_predictions) == len(cur_nonmember_individual_predictions)
                    fpr, tpr, individual_roc_auc_ = get_roc_metrics(cur_nonmember_individual_predictions, cur_member_individual_predictions)
                    # save_ll_histograms(cur_member_individual_predictions, cur_nonmember_individual_predictions, "individual", 0.05, SAVE_FOLDER)
                    # save_roc_curves("individual", fpr, tpr, roc_auc, SAVE_FOLDER)

                    assert len(cur_member_individual_predictions_with_set_label) == len(cur_nonmember_individual_predictions_with_set_label)
                    fpr, tpr, individual_roc_auc_set = get_roc_metrics(cur_nonmember_individual_predictions_with_set_label, cur_member_individual_predictions_with_set_label)
                    # save_ll_histograms(cur_member_individual_predictions_with_set_label, cur_nonmember_individual_predictions_with_set_label, "individual_top-k={}".format(top_k), 0.05, SAVE_FOLDER)
                    # save_roc_curves("individual_top-k={}".format(top_k), fpr, tpr, roc_auc, SAVE_FOLDER)

                    if k not in all_results:
                        all_results[k] = {}
                    all_results[k][top_s] = {
                        "ROC AUC": roc_auc,
                        "Group size": len(cur_member_predictions),
                        "ROC AUC Individual": individual_roc_auc_,
                        "ROC AUC Individual with set label": individual_roc_auc_set
                    }
                    if roc_auc > best_auc:
                        best_s = top_s
                        best_auc = roc_auc
                        best_fpr = fpr
                        best_tpr = tpr
                
                output = {
                    "s": best_s,
                    "ROC AUC": best_auc,
                }
                all_results[k]["best"] = output

                print("For {} documents: ".format(k))
                print("top s: {}".format(output['s']))
                print("ROC AUC: {}".format(output['ROC AUC']))
                save_roc_curves("group_best", best_fpr, best_tpr, best_auc, SAVE_FOLDER)

            # Graph C map
            ticks = generate_topk_array(max_top_k)
            data = [] 
            for k in ticks:
                cur_row = []
                for s in ticks:
                    if s in all_results[k]:
                        cur_row.append(all_results[k][s]["ROC AUC"])
                    else:
                        cur_row.append(np.nan)
                data.append(cur_row)
            save_cmap(data, ticks, method)


            all_results["Total individual ROC AUC"] = individual_roc_auc
            with open(os.path.join(SAVE_FOLDER, "group_output.json"), 'w') as f:
                json.dump(all_results, f, indent=4)
            
            
            