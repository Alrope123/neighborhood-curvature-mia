import json
import numpy as np
import os
import pickle as pkl
import argparse
import random
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset, concatenate_datasets

def dict_of_lists_to_list_of_dicts(dict_of_lists):
    # Check if the dictionary is empty
    if not dict_of_lists:
        return []

    # Get the length of the lists in the dictionary assuming all lists are of the same length
    list_length = len(next(iter(dict_of_lists.values())))

    # Create a list of dictionaries
    list_of_dicts = []
    for i in range(list_length):
        # Create a new dictionary for each index in the lists
        new_dict = {key: value[i] for key, value in dict_of_lists.items()}
        list_of_dicts.append(new_dict)

    return list_of_dicts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_answer', default=False, action="store_true")
    parser.add_argument('--out_path', type=str, default="/gscratch/h2lab/alrope/data/eval")
    args = parser.parse_args()

    eval_datasets = ["allenai/ai2_arc@ARC-Easy", "google/boolq", "gsm8k@main", "lambada", "natural_questions@default", "openbookqa@main", "piqa"]
    mmlu_subsets = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    eval_datasets.extend(["cais/mmlu@{}".format(subset) for subset in mmlu_subsets])

    choices_list = ["A", "B", "C", "D"]

    np.random.seed(2023)
    random.seed(2023)

    for eval_dataset in eval_datasets:
        print("Processing {}".format(eval_dataset))

        names = eval_dataset.split('@')
        dataset_name = names[0]
        if len(names) > 1:
            subset_name = names[1]
        else:
            subset_name = None

        if subset_name:
            huggingface_datasets = load_dataset(dataset_name, subset_name)
        else:
            huggingface_datasets = load_dataset(dataset_name)
        
        if "test" in huggingface_datasets:
            huggingface_data = huggingface_datasets["test"][:5000]
        elif "validation" in huggingface_datasets:
            huggingface_data = huggingface_datasets["validation"][:5000]
        else:
            raise NotImplementedError('Dataset splits: {}'.format(huggingface_datasets))
        
        huggingface_data = huggingface_data.to_iterable_dataset()

        new_dataset = []
        for i, entry in enumerate(huggingface_data):
            if not args.include_answer:
                if "question" in entry:
                    if type(entry["question"]) == str:
                        text = entry["question"]
                    elif type(entry["question"]) == dict:
                        if "text" in entry["question"]:
                            text = entry["question"]["text"]
                        else:
                            raise NotImplementedError('Entry fields: {}'.format(entry["question"]))
                elif "text" in entry:
                    text = entry["text"]
                elif "question_stem" in entry:
                    text = entry["question_stem"]
                elif "question_stem" in entry:
                    text = entry["question_stem"]
                elif "goal" in entry:
                    text = entry["goal"]
                else:
                    raise NotImplementedError('Entry fields: {}'.format(entry))
            else:
                splitter = '\n'
                if eval_dataset.startswith("allenai/ai2_arc"):
                    text = entry["question"] + splitter + entry["choices"]["text"][entry["answerKey"]]
                elif eval_dataset.startswith("google/boolq"):
                    text = entry["question"] + splitter + entry["question"]
                elif eval_dataset.startswith("gsm8k"):
                    text = entry["question"] + splitter + entry["answer"]
                elif eval_dataset.startswith("lambada"):
                    text = entry["text"]
                elif eval_dataset.startswith("natural_questions"):
                    text = entry["question"]["text"]
                elif eval_dataset.startswith("openbookqa"):
                    text = entry["question_stem"] + " " + entry["choices"]["text"][entry["choices"]["label"].index(entry["answerKey"])]
                elif eval_dataset.startswith("piqa"):
                    text = entry["goal"] + splitter + entry["sol{}".format(entry["label"])]
                elif eval_dataset.startswith("cais/mmlu"):
                    text = entry["question"] + splitter + entry["choices"][choices_list.index(entry["answerKey"])]

            new_entry  = {"group": eval_dataset, "text": text}
            new_dataset.append(new_entry)

        # if not args.include_answer:
        #     if "question" in huggingface_data:
        #         if type(entry["question"]) == str:
        #             text = entry["question"]
        #         elif type(entry["question"]) == dict:
        #             if "text" in entry["question"]:
        #                 text = entry["question"]["text"]
        #             else:
        #                 raise NotImplementedError('Entry fields: {}'.format(entry["question"]))
        #     elif "text" in huggingface_data:
        #         text = entry["text"]
        #     elif "question_stem" in huggingface_data:
        #         text = entry["question_stem"]
        #     elif "question_stem" in huggingface_data:
        #         text = entry["question_stem"]
        #     elif "goal" in huggingface_data:
        #         text = entry["goal"]
        #     else:
        #         raise NotImplementedError('Entry fields: {}'.format(entry))
        #     dataset_key = 
        # for i, entry in enumerate(huggingface_data):
            
        #         if "question" in entry:
        #             if type(entry["question"]) == str:
        #                 text = entry["question"]
        #             elif type(entry["question"]) == dict:
        #                 if "text" in entry["question"]:
        #                     text = entry["question"]["text"]
        #                 else:
        #                     raise NotImplementedError('Entry fields: {}'.format(entry["question"]))
        #         elif "text" in entry:
        #             text = entry["text"]
        #         elif "question_stem" in entry:
        #             text = entry["question_stem"]
        #         elif "question_stem" in entry:
        #             text = entry["question_stem"]
        #         elif "goal" in entry:
        #             text = entry["goal"]
        #         else:
        #             raise NotImplementedError('Entry fields: {}'.format(entry))
        #     else:
        #         splitter = '\n'
        #         if eval_dataset.startswith("allenai/ai2_arc"):
        #             text = entry["question"] + splitter + entry["choices"]["text"][entry["answerKey"]]
        #         elif eval_dataset.startswith("google/boolq"):
        #             text = entry["question"] + splitter + entry["question"]
        #         elif eval_dataset.startswith("gsm8k"):
        #             text = entry["question"] + splitter + entry["answer"]
        #         elif eval_dataset.startswith("lambada"):
        #             text = entry["text"]
        #         elif eval_dataset.startswith("natural_questions"):
        #             text = entry["question"]["text"]
        #         elif eval_dataset.startswith("openbookqa"):
        #             text = entry["question_stem"] + " " + entry["choices"]["text"][entry["choices"]["label"].index(entry["answerKey"])]
        #         elif eval_dataset.startswith("piqa"):
        #             text = entry["goal"] + splitter + entry["sol{}".format(entry["label"])]
        #         elif eval_dataset.startswith("cais/mmlu"):
        #             text = entry["question"] + splitter + entry["choices"][choices_list.index(entry["answerKey"])]

        #     new_entry  = {"group": eval_dataset, "text": text}
        #     new_dataset.append(new_entry)


        with open(os.path.join("/gscratch/h2lab/alrope/data/eval{}/{}.jsonl".format("_full" if args.include_answer else "", eval_dataset)), 'w') as f:
            for entry in new_dataset:
                f.write(json.dumps(entry) + "\n")
