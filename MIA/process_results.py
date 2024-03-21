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
import zlib

def merge_json(json_objects):
    if all(isinstance(obj, dict) for obj in json_objects):
        merged_avg = {}
        merged_std = {}
        for key in json_objects[0].keys():
            merged_avg[key], merged_std[key] = merge_json([obj[key] for obj in json_objects])
        return merged_avg, merged_std
    
    elif all(isinstance(obj, list) for obj in json_objects):
        max_length = max(len(obj) for obj in json_objects)
        merged_avg = []
        merged_std = []
        for i in range(max_length):
            ith_values = [obj[i] for obj in json_objects if i < len(obj)]
            merged_avg.append(merge_json(ith_values)[0])
            merged_std.append(merge_json(ith_values)[1])
        return merged_avg, merged_std
    
    elif all(isinstance(obj, (int, float)) for obj in json_objects):
        avg = sum(json_objects) / len(json_objects)
        std_dev = math.sqrt(sum((x - avg) ** 2 for x in json_objects) / len(json_objects))
        return avg, std_dev
    
    else:
        raise ValueError("Inconsistent data types in JSON objects")


def generate_topk_array(n):
    base_list = [1, 3, 5, 8]
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

    fpr, tpr, thresholds = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), thresholds.tolist(), float(roc_auc)


def save_roc_curves(name, fpr, tpr, roc_auc, save_dir=None):
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
    plt.savefig(f"{save_dir}/roc_curves_{name}.png")


def calculate_group_loss(losses, aggregated_method, direction, top_s):
    if aggregated_method == 'mean':
        return np.mean(sorted(losses, reverse=direction == "min")[:top_s])
    else:
        raise NotImplementedError("The aggregation method is not supported.")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(members, nonmembers, name, bin_width, save_dir):
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
    plt.hist(members, alpha=0.5, bins=bins_members, weights=np.ones(len(members)) / len(members), label='member')
    plt.hist(nonmembers, alpha=0.5, bins=bins_nonmembers, weights=np.ones(len(nonmembers)) / len(nonmembers), label='non-member')
    plt.xlabel("Likelihood")
    plt.ylabel('Count')
    plt.title(name)
    plt.legend(loc='upper right')
    plt.savefig(f"{save_dir}/ll_histograms_{name}.png")
    print(f"Plotting at {save_dir}/ll_histograms_{name}.png")


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

    ticks = [1] + ticks
    print([str(tick) for tick in ticks])

    # Setting custom tick labels
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_yticklabels([str(tick) for tick in ticks])

    # Showing the plot
    plt.savefig(f"{SAVE_FOLDER}/cmap_{name}.png")
    print(f"Plotting at {SAVE_FOLDER}/cmap_{name}.png")


def calculate_compression_entropy(text):
    # Encode text to bytes
    text_bytes = text.encode('utf-8')
    
    # Compress the bytes
    compressed_data = zlib.compress(text_bytes)
    
    # Calculate the size of the compressed data in bits
    compressed_size_bits = len(compressed_data) * 8
    
    # Calculate the average bits per original symbol
    # This is the entropy rate of the compressed text
    entropy_rate = compressed_size_bits / len(text_bytes)
    
    return entropy_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/lr_ratio_threshold_results.json")
    parser.add_argument('--result_path_ref', type=str, default=None)
    parser.add_argument('--membership_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl")
    parser.add_argument('--out_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/")
    parser.add_argument('--keys', nargs="+", default="crit")
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--random_groups', type=int, default=1)  

    args = parser.parse_args()

    assert os.path.exists(args.result_path), args.result_path
    assert os.path.exists(args.membership_path), args.membership_path
    
    with open(args.result_path, 'r') as f:
        result = json.load(f)
    if args.result_path_ref != None:
        assert "crit" in args.keys
        with open(args.result_path_ref, 'r') as f:
            result_ref = json.load(f)
    with open(args.membership_path, 'rb') as f:
        group_to_documents = pkl.load(f)
    
    SAVE_FOLDER = args.out_dir
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    max_top_k = args.top_k
    random.seed(2023)
    np.random.seed(2023)

    print("# of samples: {}".format(result['n_samples']))

    ROOT_SAVE_FOLDER = SAVE_FOLDER
    for key in args.keys:
        print("Metrics on document level:")
        nonmember_key = f"nonmember_{key}"
        member_key = f"member_{key}"

        # add additional key to the results
        if key == "bff":
            result[nonmember_key] = result["nonmember_lls"]
            result[member_key] = result["member_lls"]
        elif key == "min_k":
            result[member_key] = result["nonmember_crit"]
            print(result["nonmember_crit"][:100])
            result[nonmember_key] = result["member_crit"]
            print(result["member_crit"][:100])
        elif key == "ref_lls":
            result[nonmember_key] = [lls - crit for lls, crit in zip(result["nonmember_lls"], result["nonmember_crit"])]
            result[member_key] = [lls - crit for lls, crit in zip(result["member_lls"], result["member_crit"])]
        elif key == "zlib":
            result[nonmember_key] = [lls - calculate_compression_entropy(text) for lls, text in zip(result["nonmember_lls"], result["nonmember"])]
            result[member_key] = [lls - calculate_compression_entropy(text) for lls, text in zip(result["member_lls"], result["member"])]
        elif key == "crit" and args.result_path_ref != None:
            nonmember_meta_to_index = {}
            for i, entry in enumerate(result_ref["nonmember_meta"]):
                nonmember_meta_to_index[tuple(entry)] = i
            member_meta_to_index = {}
            for i, entry in enumerate(result_ref["member_meta"]):
                member_meta_to_index[tuple(entry)] = i

            result[nonmember_key] = [lls_base - result_ref["nonmember_lls"][nonmember_meta_to_index[tuple(result["nonmember_meta"][i])]] for i, lls_base in enumerate(result["nonmember_lls"])]
            result[member_key] = [lls_base - result_ref["member_lls"][member_meta_to_index[tuple(result["member_meta"][i])]] for i, lls_base in enumerate(result["member_lls"])]
            sets_members = []
            sets_nonmembers = []
            for cur_result in [result, result_ref]:
                sets_members.append(set([(filename, i) for filename, i in cur_result['member_meta']]))
                sets_nonmembers.append(set([(filename, i) for filename, i in cur_result['nonmember_meta']]))
            for set_members in sets_members:
                assert set_members == sets_members[0], [set_members, sets_members[0]]
            for set_nonmembers in sets_nonmembers:
                assert set_nonmembers == sets_nonmembers[0], [set_nonmembers, sets_nonmembers[0]]


        nonmember_predictions = result[nonmember_key]
        member_predictions = result[member_key]
        individual_fpr, individual_tpr, individual_thresholds, individual_roc_auc = get_roc_metrics(nonmember_predictions, member_predictions)
        # Draw log likehood histogram on individual documents
        # compare_length = min(len(nonmember_predictions), len(member_predictions))
        # if len(nonmember_predictions) > len(member_predictions):
        #     nonmember_predictions = np.random.choice(nonmember_predictions, len(member_predictions), replace=False)
        # elif len(member_predictions) > len(nonmember_predictions):
        #     member_predictions = np.random.choice(member_predictions, len(nonmember_predictions), replace=False)
        save_ll_histograms(member_predictions, nonmember_predictions,f"individual_with_{key}", 0.05, ROOT_SAVE_FOLDER)
        print("Individual AUC-ROC with {}: {}".format(key, individual_roc_auc))
        save_roc_curves("Individual_with_{}".format(key), individual_fpr, individual_tpr, individual_roc_auc, ROOT_SAVE_FOLDER)

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
        print("# of member groups: {}".format(len(group_results_members)))
        print("# of nonmember groups: {}".format(len(group_results_nonmembers)))
        print("Average # document in member group: {}/{}".format(np.mean([len(members) for _, members in group_results_members.items()]), np.std([len(members) for _, members in group_results_members.items()])))
        print("Average # document in nonmember group: {}/{}".format(np.mean([len(members) for _, members in group_results_nonmembers.items()]), np.std([len(members) for _, members in group_results_nonmembers.items()])))

        original_group_results_members = {}
        original_group_results_nonmembers = {}
        # Randomly shuffle predictions
        for group, predictions in group_results_members.items():
            if len(predictions) >= max_top_k:
                random.shuffle(predictions)
                original_group_results_members[group] = predictions
        for group, predictions in group_results_nonmembers.items():
            if len(predictions) >= max_top_k:
                random.shuffle(predictions)
                original_group_results_nonmembers[group] = predictions

        all_results = []

        aggregated_method = "mean"
    
        for seed, startinx in enumerate(range(0, args.random_groups*max_top_k, max_top_k)):
            seed_result = {}    
            for direction in ['max', "min"]:
                direction_result = {}

                qualified_group_results_members = {}
                qualified_group_results_nonmembers = {}
                for group, predictions in original_group_results_members.items():
                    if len(predictions) >= max_top_k:
                        qualified_group_results_members[group] = predictions[startinx: startinx+max_top_k]
                        assert len(predictions[startinx: startinx+max_top_k]) >= max_top_k, f"Not enough document at group {seed}"
                for group, predictions in original_group_results_nonmembers.items():
                    if len(predictions) >= max_top_k:           
                        qualified_group_results_nonmembers[group] = predictions[startinx: startinx+max_top_k]
                        assert len(predictions[startinx: startinx+max_top_k]) >= max_top_k, f"Not enough document at group {seed}"
                # group_results_members, group_results_nonmembers = make_dicts_equal(qualified_group_results_members, qualified_group_results_nonmembers)
                group_results_members = qualified_group_results_members
                group_results_nonmembers = qualified_group_results_nonmembers

                for k in generate_topk_array(max_top_k):
                    direction_result[k] = {}

                    # Randomly k documents from each group 
                    cur_group_results_members = {}
                    cur_group_results_nonmembers = {}
                    cur_member_individual_predictions = []
                    cur_nonmember_individual_predictions = []
                    for group, predictions in group_results_members.items():
                        for i in range(0, len(predictions), k):
                            cur_group_results_members[group+f"_{i//k}"] = predictions[i:i+k]
                            cur_member_individual_predictions.extend(predictions[i:i+k])
                    for group, predictions in group_results_nonmembers.items():
                        for i in range(0, len(predictions), k):
                            cur_group_results_nonmembers[group+f"_{i//k}"] = predictions[i:i+k]
                            cur_nonmember_individual_predictions.extend(predictions[i:i+k])
                    fpr, tpr, thresholds, individual_roc_auc_ = get_roc_metrics(cur_nonmember_individual_predictions, cur_member_individual_predictions)
                    direction_result[k] = {
                        "Member size": len(cur_group_results_members),
                        "Nonmember size": len(cur_group_results_nonmembers),
                        "ROC AUC Individual": individual_roc_auc_,
                        "MIA": {}
                    }

                    for top_s in generate_topk_array(k):
                        cur_member_predictions = []
                        cur_nonmember_predictions = []

                        if key != 'bff':
                            for group, predictions in cur_group_results_members.items():
                                group_loss = calculate_group_loss(predictions, aggregated_method, direction, top_s)
                                cur_member_predictions.append(group_loss)
                            
                            for group, predictions in cur_group_results_nonmembers.items():
                                group_loss = calculate_group_loss(predictions, aggregated_method, direction, top_s)
                                cur_nonmember_predictions.append(group_loss)
                        else:
                            for group, predictions in cur_group_results_members.items():
                                scores = [score for (_, _, score, _) in group_to_documents[group]['documents']]
                                group_loss = calculate_group_loss(scores, aggregated_method, direction, top_s)
                                cur_member_predictions.append(group_loss)
                            for group, predictions in cur_group_results_nonmembers.items():
                                scores = [score for (_, _, score, _) in group_to_documents[group]['documents']]
                                group_loss = calculate_group_loss(scores, aggregated_method, direction, top_s)
                                cur_nonmember_predictions.append(group_loss)
                        
                        fpr, tpr, thresholds, roc_auc = get_roc_metrics(cur_nonmember_predictions, cur_member_predictions)
                        cur_nonmember_predictions = [prediction for prediction in cur_nonmember_predictions if prediction]
                        cur_member_predictions = [prediction for prediction in cur_member_predictions if prediction]
                        average_cur_nonmember = np.nanmean(cur_nonmember_predictions)
                        std_cur_nonmember = np.nanstd(cur_nonmember_predictions)
                        average_cur_member = np.nanmean(cur_member_predictions)
                        std_cur_member = np.nanstd(cur_member_predictions)
                        threshold = None
                        tpr_is = None
                        for i, rate in enumerate(fpr):
                            if rate > 0.05:
                                threshold = thresholds[i-1]
                                tpr_is = tpr[i-1]
                                break

                        # direction_result[k]["MIA"][top_s] = (roc_auc, fpr, tpr)
                        direction_result[k]["MIA"][top_s] = {
                            "ROC_AUC": roc_auc,
                            "threshold@FPR=0.05": threshold,
                            "TPR@FPR=0.05": tpr_is,
                            "Member Average": average_cur_member,
                            "Member STD": std_cur_member,
                            "Non-Member Average": average_cur_nonmember,
                            "Non-Member STD": std_cur_nonmember,
                            "Average Diff": average_cur_member - average_cur_nonmember
                        }
                        
                    
                seed_result[direction] = direction_result


            all_results.append(seed_result)

        print(all_results)
        # average_results, std_results = merge_json(all_results)
        average_results, std_results = all_results[0], 0

        for direction in ['max', "min"]:
            SAVE_FOLDER = os.path.join(ROOT_SAVE_FOLDER, "{}".format(key))
            if not os.path.exists(SAVE_FOLDER):
                os.mkdir(SAVE_FOLDER)
            for k in generate_topk_array(max_top_k):
                best_s = None
                # best_fpr = None
                # best_tpr = None
                best_auc = -1
                best_threshold = -1
                best_tpr_is = -1
                for top_s in generate_topk_array(k):
                    roc_auc = average_results[direction][k]['MIA'][top_s]["ROC_AUC"]
                    threshold = average_results[direction][k]['MIA'][top_s]["threshold@FPR=0.05"]
                    tpr_is = average_results[direction][k]['MIA'][top_s]["TPR@FPR=0.05"]
                    if roc_auc > best_auc:
                        best_s = top_s
                        best_auc = roc_auc
                        # best_fpr = fpr
                        # best_tpr = tpr
                        best_threshold = threshold
                        best_tpr_is  = tpr_is
                output = {
                    "s": best_s,
                    "ROC AUC": best_auc,
                    "threshold": best_threshold,
                    "TPR at 5%": best_tpr_is
                }
                average_results[direction][k]["MIA"]["best"] = output
            
            # Graph C map
            ticks = generate_topk_array(max_top_k)
            data = [] 
            for k in ticks:
                cur_row = []
                for s in ticks:
                    if s in average_results[direction][k]["MIA"]:
                        cur_row.append(average_results[direction][k]["MIA"][s]["ROC_AUC"])
                    else:
                        cur_row.append(np.nan)
                data.append(cur_row)
            save_cmap(data, ticks, direction)

        for i, rate in enumerate(individual_fpr):
            if rate > 0.05:
                individual_threshold = individual_thresholds[i-1]
                individual_tpr_is = individual_tpr[i-1]
                break

        average_results["Total individual ROC AUC"] = individual_roc_auc
        average_results["Total TPR@5%"] = individual_tpr_is
        average_results["Total threshold"] = individual_threshold
        final_results = {"average": average_results, "std": std_results}
        with open(os.path.join(SAVE_FOLDER, "group_output.json"), 'w') as f:
            json.dump(final_results, f, indent=4)

        