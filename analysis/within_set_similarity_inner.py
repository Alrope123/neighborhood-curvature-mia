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

DATASETS = ['rpj-arxiv', 'wikipedia', 'wikipedia_noisy', 'wikipedia_month', 'rpj-arxiv_noisy', 'rpj-arxiv_month', 'rpj-book', 'language']

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def sample_segment(text, tokenizer, max_length):
    # def random_segment(l, length, max_length):
    #     idx_random = random.randint(0, length-max_length)
    #     return l[idx_random: idx_random + max_length]
    segments = []
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_length):
        segments.append(tokenizer.decode(tokens[i: i+max_length]))
    return segments


def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)


def sample_group(membership_info, n_group=100, n_document_per_group=30, train=True):
    groups = set()
    info_list = list(membership_info.items())
    if n_group < 0:
        n_group = len([group for group, infos in info_list if infos['group_is_member'] == train])
    random.shuffle(info_list)
    for group, infos in info_list:
        if len(groups) >= n_group:
            break
        if infos['group_is_member'] == train and len(infos['is_members']) >= n_document_per_group:
            groups.add(group)
    # assert len(groups) == n_group, (len(groups), n_group)

    selected_data = set()
    for group, infos in membership_info.items():
        if group in groups:
            new_added_data = []
            for filename, i, _, _ in infos['documents']:
                new_added_data.append((filename, i))
            # Oversample the documents to give room for unqualified document
            random.shuffle(new_added_data)
            new_added_data = new_added_data[:int(n_document_per_group * 1.2)]
            selected_data.update(new_added_data)
    # assert len(selected_data) >= n_group * n_document_per_group, len(selected_data)
    return selected_data


def load_dataset(membership_info, data_dir=None, train=True, n_group=100, n_document_per_group=30): 
    selected_data = sample_group(membership_info, n_group, n_document_per_group, train)
    
    data = [] 
    meta_data = []
    for file_path, filename in tqdm(iterate_files(data_dir)):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if (filename, i) in selected_data:
                    dp = json.loads(line)      
                    meta_data.append((filename, i))
                    data.append(dp['text'])
    assert len(data) == len(selected_data), (len(data), len(selected_data))
    return data, meta_data


def load(data_dir, membership_path, verbose=False, n_group=100, n_document_per_group=30, train=True):
    with open(membership_path, 'rb') as f:
        membership_info = pkl.load(f)
    return load_dataset(membership_info, data_dir=data_dir, n_group=n_group, n_document_per_group=n_document_per_group, train=train)


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
    return len(intersection) / len(union) if len(union) > 0 else 0

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


def strip_newlines(text):
    return ' '.join(text.split())


if __name__ == '__main__':
    DEVICE = "cuda"
    random.seed(2023)
    np.random.seed(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--result_dir', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/")
    parser.add_argument('--membership_path', type=str, default="/gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl")
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--model_name', type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--max_top_k', type=int, default=100)
    parser.add_argument('--n_group_member', type=int, default=100)
    parser.add_argument('--n_group_nonmember', type=int, default=100)
    parser.add_argument('--downsize_factor', type=float, default=0.1)
    parser.add_argument('--methods', nargs="+", default="fasttext")
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()

    assert os.path.exists(args.result_dir), args.result_dir
    assert os.path.exists(args.membership_path), args.membership_path

    max_top_k = args.max_top_k
    downsize_factor = args.downsize_factor

    if os.path.exists(os.path.join(args.result_dir, "within_set_similarity_inner.json")):
        with open(os.path.join(args.result_dir, "within_set_similarity_inner.json"), 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Load selected document
    member_data, member_metadata = load(data_dir=args.data_dir,
            membership_path=args.membership_path,
            n_group=args.n_group_member, n_document_per_group=-1, train=True)
    nonmember_data, nonmember_metadata = load(data_dir=args.data_dir,
            membership_path=args.membership_path,
            n_group=args.n_group_nonmember, n_document_per_group=-1, train=False)
    
    member_data = [x.strip() for x in member_data]
    member_data = [strip_newlines(x) for x in member_data]
    member_data = [x for x in member_data if len(x.split()) > 0 and len(x) > 2048]

    nonmember_data = [x.strip() for x in nonmember_data]
    nonmember_data = [strip_newlines(x) for x in nonmember_data]
    nonmember_data = [x for x in nonmember_data if len(x.split()) > 0 and len(x) > 2048]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.name, cache_dir=args.cache_dir)

    new_member_data = []
    new_nonmember_data = []
    for text in member_data:
        new_member_data.extend(sample_segment(text, tokenizer, args.max_length))
    for text in nonmember_data:
        new_nonmember_data.extend(sample_segment(text, tokenizer, args.max_length))

    member_data = new_member_data
    nonmember_data = new_nonmember_data

    if 'fasttext' in args.methods:
        model = load_model(args.model_name)
    
    # Calculate the word embeddings
    group_similarity_member = {method: [] for method in args.methods}
    random_indices = np.random.randint(0, len(member_data), size=int(len(member_data) * downsize_factor))
    sampled_documents = [documents for i, documents in enumerate(member_data) if i in random_indices]
    for documents in tqdm(sampled_documents):
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
            group_similarity_member[method].append(average_similarity)
    
    group_similarity_nonmember = {method: [] for method in args.methods}
    random_indices = np.random.randint(0, len(nonmember_data), size=int(len(nonmember_data) * downsize_factor))
    sampled_documents = [documents for i, documents in enumerate(nonmember_data) if i in random_indices]
    for documents in tqdm(sampled_documents):
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
            group_similarity_nonmember[method].append(average_similarity)

    for method in args.methods:
        if method in results:
            continue
        result = {}
        result["final average"] = np.mean([value for value in group_similarity_member[method] + group_similarity_nonmember[method] if not math.isnan(value)])
        result["member"] = group_similarity_member[method]
        result["nonmember"] = group_similarity_nonmember[method]
        results[method] = result
        print("Final average for {} is: {}".format(method, result["final average"]))

    with open(os.path.join(args.result_dir, "within_set_similarity_inner.json"), 'w') as f:
        json.dump(results, f, indent =4)
    
