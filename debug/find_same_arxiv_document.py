from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
import os
import json

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)


if __name__ == '__main__':

    doc_dir = "/data/pile/train/00.jsonl"
    docs = []
    with open(doc_dir, 'r') as f:
        for line in f:
            dp = json.loads(line)
            if dp['meta']['pile_set_name'] == "ArXiv":
                docs.append(dp['text'])
                if len(docs) >= 1:
                    break 
    assert len(docs) == 1

    output = {}
    for j, doc in enumerate(docs):
        print("Processing doc No.1")
        k = 5
        top_k_texts = [None] * k
        top_k_scores = [-float('inf')] * k

        tfidf_vectorizer = TfidfVectorizer().fit([doc])  # Fit on the target doc first
        target_vector = tfidf_vectorizer.transform([doc])

        data_dir = "/gscratch/h2lab/alrope/data/redpajama/arxiv/"
        for i, (data_path, filename) in tqdm(enumerate(iterate_files(data_dir))):
            if i > 1:
                break

            doc_pool = []
            with open(data_path, 'r') as f:
                for line in f:
                    doc_pool.append(json.loads(line)['text'])  

            # Transform the batch using the same vectorizer (this ensures consistent feature space)
            batch_vectors = tfidf_vectorizer.transform(doc_pool)
            
            # Compute similarities for this batch
            cosine_similarities = linear_kernel(target_vector, batch_vectors).flatten()

            for i, score in enumerate(cosine_similarities):
                if score > min(top_k_scores):  # If this score is among the top k scores
                    # Get the index to replace based on the smallest score among top_k_scores
                    replace_index = top_k_scores.index(min(top_k_scores))
                    
                    # Replace the score and text in the top_k lists
                    top_k_scores[replace_index] = score
                    top_k_texts[replace_index] = doc_pool[i]

        candidates = sorted(zip(top_k_texts, top_k_scores), key=lambda x: x[1], reverse=True)
        output[doc] = candidates

    out_dir = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/out"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'same_arxiv_document'), 'w') as f:
        json.dump(output, f)    
