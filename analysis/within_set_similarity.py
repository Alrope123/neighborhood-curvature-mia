import numpy as np
import transformers
import torch
from tqdm import tqdm
import random
import argparse
import os
import json
import time
import fasttext
import math
from huggingface_hub import hf_hub_download
# from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def load_model(name):
    model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin", cache_dir=args.cache_dir)
    model = fasttext.load_model(model_path)
    return model


##################################### Fasttext ##################################################
def get_embeddings(model, documents):
    """Get FastText embedding for a given text."""
    embeddings = []
    for text in documents:
        words = text.split()
        vectors = [model[word] for word in words if word in model]
        if vectors:
            embedding = np.mean(vectors, axis=0)
            embeddings.append(embedding)
        else:
            embedding = np.zeros(300)
            embeddings.append(embedding)
    return embeddings

# def compute_average_cosine_similarity(embeddings):
#     """Compute average cosine sifmilarity among a list of texts."""
#     total_similarity = 0
#     total_pairs = 0
#     for i in range(len(embeddings)):
#         for j in range(i+1, len(embeddings)):
#             similarity = 1 - cosine(embeddings[i], embeddings[j])
#             total_similarity += similarity
#             total_pairs += 1
    
#     average_similarity = total_similarity / total_pairs if total_pairs != 0 else 0
#     return average_similarity

##################################### N-gram ####################################################
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# Function to create n-grams from a list of texts
def create_ngrams(text, n):
    # Remove punctuation and generate n-grams
    n_grams = ngrams(''.join([char for char in text.lower() if char not in string.punctuation]).split(), n)
    return set(n_grams)


def compute_average_cosine_similarity(embeddings):
    similarities = []
    similarity_matrix = cosine_similarity(embeddings)
    assert len(documents) == len(similarity_matrix), (len(documents), len(similarity_matrix))
    assert len(documents) == len(similarity_matrix[0]), (len(documents), len(similarity_matrix[0]))
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            similarities.append(similarity_matrix[i][j])
    return float(np.mean(similarities))


if __name__ == '__main__':
    DEVICE = "cuda"
    random.seed(2023)
    np.random.seed(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/")
    parser.add_argument('--membership_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl")
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--model_name', type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--max_top_k', type=int, default=100)
    parser.add_argument('--downsize_factor', type=float, default=0.1)
    parser.add_argument('--methods', nargs="+", default="fasttext")
    args = parser.parse_args()

    assert os.path.exists(args.result_dir), args.result_dir
    assert os.path.exists(args.membership_path), args.membership_path

    max_top_k = args.max_top_k
    downsize_factor = args.downsize_factor


    # Load selected document
    with open(os.path.join(args.result_dir, "lr_ratio_threshold_results.json"), 'r') as f:
        result = json.load(f)
    with open(args.membership_path, 'rb') as f:
        group_to_documents = pkl.load(f)
    
    if os.path.exists(os.path.join(args.result_dir, "within_set_similarity.json")):
        with open(os.path.join(args.result_dir, "within_set_similarity.json"), 'r') as f:
            results = json.load(f)
    else:
        results = {}

    info_to_group = {}
    for group, infos in group_to_documents.items():
        for filename, i, _, is_member in infos['documents']:
            info_to_group[(filename, i)] = group

    group_results_members = {}
    group_results_nonmembers = {}
    for i, entry in enumerate(result["member"]):
        group_member = info_to_group[tuple(result['member_meta'][i])]
        assert group_to_documents[group_member]['group_is_member']
        if group_member not in group_results_members:
            group_results_members[group_member] = []
        group_results_members[group_member].append(entry)
    for i, entry in enumerate(result["nonmember"]):
        group_nonmember = info_to_group[tuple(result['nonmember_meta'][i])]
        assert not group_to_documents[group_nonmember]['group_is_member']
        if group_nonmember not in group_results_nonmembers:
            group_results_nonmembers[group_nonmember] = []
        group_results_nonmembers[group_nonmember].append(entry)

    original_group_results_members = {}
    original_group_results_nonmembers = {}
    for group, predictions in group_results_members.items():
        if len(predictions) >= max_top_k:
            random.shuffle(predictions)
            original_group_results_members[group] = predictions[:max_top_k]
    for group, predictions in group_results_nonmembers.items():
        if len(predictions) >= max_top_k:
            random.shuffle(predictions)
            original_group_results_nonmembers[group] = predictions[:max_top_k]

    group_results_members = original_group_results_members
    group_results_nonmembers = original_group_results_nonmembers

    if 'fasttext' in args.methods:
        model = load_model(args.model_name)
    
    # Calculate the word embeddings
    group_similarity_member = {method: {} for method in args.methods}
    random_indices = np.random.randint(0, len(group_results_members), size=int(len(group_results_members) * downsize_factor))
    sampled_group_documents = [(group, documents) for i, (group, documents) in enumerate(group_results_members.items()) if i in random_indices]
    for group, documents in tqdm(sampled_group_documents):
        for method in args.methods:
            if method in results:
                continue
            if method == 'fasttext':
                documents_embeddings = get_embeddings(model, documents)
                average_similarity = compute_average_cosine_similarity(documents_embeddings)
            elif method == 'tf-idf':
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(documents)
                average_similarity = compute_average_cosine_similarity(tfidf_matrix)
            elif method.startswith('n-gram'):
                from nltk import ngrams
                from nltk.corpus import stopwords
                import string
                import itertools
                ngrams_list = [create_ngrams(document, method.split('-')[-1]) for document in documents]
                similarities = []
                for text1, text2 in itertools.combinations(range(len(ngrams_list)), 2):
                    similarities.append(jaccard_similarity(ngrams_list[text1], ngrams_list[text2]))
                average_similarity = np.mean(similarities)
            group_similarity_member[method][group] = average_similarity
    
    group_similarity_nonmember = {method: {} for method in args.methods}
    random_indices = np.random.randint(0, len(group_results_nonmembers), size=int(len(group_results_nonmembers) * downsize_factor))
    sampled_group_documents = [(group, documents) for i, (group, documents) in enumerate(group_results_nonmembers.items()) if i in random_indices]
    for group, documents in tqdm(sampled_group_documents):
        for method in args.methods:
            if method in results:
                continue
            if method == 'fasttext':
                documents_embeddings = get_embeddings(model, documents)
                average_similarity = compute_average_cosine_similarity(documents_embeddings)
            elif method == 'tf-idf':
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(documents)
                average_similarity = compute_average_cosine_similarity(tfidf_matrix)
            elif method.startswith('n-gram'):
                from nltk import ngrams
                from nltk.corpus import stopwords
                import string
                import itertools
                ngrams_list = [create_ngrams(document, int(method.split('-')[-1])) for document in documents]
                similarities = []
                for text1, text2 in itertools.combinations(range(len(ngrams_list)), 2):
                    similarities.append(jaccard_similarity(ngrams_list[text1], ngrams_list[text2]))
                average_similarity = np.mean(similarities)
            group_similarity_nonmember[method][group] = average_similarity

    for method in args.methods:
        if method in results:
            continue
        result = {}
        result["final average"] = np.mean([value for value in list(group_similarity_member[method].values()) + list(group_similarity_nonmember[method].values()) if not math.isnan(value)])
        result["member"] = group_similarity_member[method]
        result["nonmember"] = group_similarity_nonmember[method]
        results[method] = result
        print("Final average for {} is: {}".format(method, result["final average"]))

    with open(os.path.join(args.result_dir, "within_set_similarity.json"), 'w') as f:
        json.dump(results, f, indent =4)
    
