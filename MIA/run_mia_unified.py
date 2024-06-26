import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
import pretraing_datasets
from multiprocessing.pool import ThreadPool
import time
import math


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# @torch.inference_mode()
def recover(
    path_raw,
    path_diff,
    path_tuned=None,
    device="cpu",
    test_inference=True,
    check_integrity_naively=False,
):
    """Recover the original weights from the released weight diff.

    This function is given for you to run.

    Things to do before running this:
        1. Convert Meta's released weights into huggingface format. Follow this guide:
            https://huggingface.co/docs/transformers/main/model_doc/llama
        2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
            https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
        3. Run this function with the correct paths. E.g.,
            python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir>

    Additional notes:
        - If things run too slowly, and you have an 80G GPU lying around, let GPU go brrr by setting `--device "cuda"`.
        - If you want to save the recovered weights, set `--path_tuned <your_path_tuned>`.
            Next time you can load the recovered weights directly from `<your_path_tuned>`.
    """
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token="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",
    )
    model_recovered: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_diff,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw,
        token="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",
    )
    if tokenizer_raw.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )
    tokenizer_recovered: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_diff
    )

    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if check_integrity_naively:
        # This is not a rigorous, cryptographically strong integrity check :)
        allsum = sum(state_dict_recovered[key].sum() for key in state_dict_recovered)
        assert torch.allclose(
            allsum, torch.full_like(allsum, fill_value=50637.1836), atol=1e-2, rtol=0
        ), "Naive integrity check failed. This could imply that some of the checkpoint files are corrupted."

    if path_tuned is not None:
        model_recovered.save_pretrained(path_tuned)
        tokenizer_recovered.save_pretrained(path_tuned)

    if test_inference:
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"Input: {input_text}\nCompletion: {output_text}")

    return model_recovered, tokenizer_recovered


def load_base_model(base_model):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        base_model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        base_model = torch.nn.DataParallel(base_model)
    print(f'DONE ({time.time() - start:.2f}s)')


def load_ref_model():
    print('MOVING ref MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        ref_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    if args.openai_model is None:
        base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'
    
    span_length = min(int(pct*len(tokens)),span_length)
    #avoid div zero:

    span_length = max(1, span_length)

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, max(1,len(tokens) - span_length))
        end =  start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def _openai_sample(p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30, max_length=200):
    # encode each text as a list of token ids
    if args.dataset == 'pubmed':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    if args.openai_model:
        # decode the prefixes back into text
        prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(args.batch_size)

        decoded = pool.map(_openai_sample, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words and tries <  args.max_tries:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            min_length = 50 if args.dataset in ['pubmed'] else 150
            if args.max_length is not None:
                min_length = 10
            #outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=max_length, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)

            outputs = base_model.generate(**all_encoded,  max_length=200,  **sampling_kwargs,  eos_token_id=base_tokenizer.eos_token_id)
            decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    if args.openai_model:
        global API_TOKEN_COUNTER

        # count total number of tokens with GPT2_TOKENIZER
        total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
        API_TOKEN_COUNTER += total_tokens

    return decoded


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


# Get the log likelihood of each text under the base_model
def get_ll(text):
    if args.openai_model:        
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            # Step 1: Extract Maximum Token ID
            max_token_id = tokenized.input_ids.max().item()

            # Step 2: Determine Embedding Size
            # Assuming base_model uses an embedding layer named 'embeddings', adjust as per your model architecture
            embedding_size = base_model.config.vocab_size

            # Step 3: Compare Values
            if max_token_id >= embedding_size or max_token_id < 0:
                raise ValueError(f"Token ID {max_token_id} is out of range for embedding layer of size {embedding_size}.")

            return -base_model(**tokenized, labels=labels).loss.item()


# Get the  likelihood ratio of each text under the base_model -- MIA baseline
def get_lira(text):
    if args.openai_model: 
        print("NOT IMPLEMENTED")
        exit(0)       
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            assert labels.size(1) <= longest_tokenizable_len, labels.size(1)  # Assuming labels is of shape [batch_size, sequence_length]
            assert labels.max().item() <= base_tokenizer.vocab_size, (labels.max().item(), base_tokenizer.vocab_size)
            assert labels.min().item() >= 0, labels.min().item()
            
            # If a reference model is specified 
            if ref_model:
                tokenized_ref = ref_tokenizer(text, return_tensors="pt").to(DEVICE)
                labels_ref = tokenized_ref.input_ids
                assert labels_ref.size(1) <= longest_tokenizable_len, labels_ref.size(1)  # Assuming labels is of shape [batch_size, sequence_length]
                assert labels_ref.max().item() <= ref_tokenizer.vocab_size, (labels_ref.max().item(), ref_tokenizer.vocab_size)
                assert labels_ref.min().item() >= 0, labels_ref.min().item()
                lls_ref = -ref_model(**tokenized_ref, labels=labels_ref).loss.item()
                if base_model:
                    lls = -base_model(**tokenized, labels=labels).loss.item()
                else:
                    lls = lls_ref - lls_ref
                return lls, lls - lls_ref
            else:
                if "silo" in base_model_name:
                    m = torch.nn.LogSoftmax(dim=-1)
                    nll = torch.nn.NLLLoss(reduction='mean')
                    
                    output = base_model(**tokenized, output_hidden_states=True, return_dict=True)
                    logits = output.logits # [batch_size, max_seq_length, n_vocabs]
                    logits = logits.reshape(-1, logits.shape[-1])
                    labels_new = labels.reshape(-1)
                    lls = nll(m(logits), labels_new).item()
                else:
                    output = base_model(**tokenized, labels=labels)
                    lls = -output.loss.item()
                if min_k_prob:
                    logits = output.logits
                    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
                    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    all_prob = []
                    input_ids_processed = labels[0][1:]
                    for i, token_id in enumerate(input_ids_processed):
                        probability = probabilities[0, i, token_id].item()
                        all_prob.append(probability)
                    ratio = 0.2
                    k_length = int(len(all_prob)*ratio)
                    topk_prob = np.sort(all_prob)[:k_length]
                    return lls, -np.mean(topk_prob).item()
                else:
                    return lls, lls-lls
                
            # else: # IF no reference model is specified, use ICL
            #     tokenized_ref = base_tokenizer(text + '\n\n' + text, return_tensors="pt").to(DEVICE)
            #     labels_ref = tokenized_ref.input_ids
            #     assert labels_ref.size(1) <= longest_tokenizable_len, labels_ref.size(1)  # Assuming labels is of shape [batch_size, sequence_length]
            #     assert labels_ref.max().item() <= base_tokenizer.vocab_size, (labels_ref.max().item(), base_tokenizer.vocab_size)
            #     assert labels_ref.min().item() >= 0, labels_ref.min().item()
            #     lls =  -base_model(**tokenized, labels=labels).loss.item()
            #     lls_ref = -base_model(**tokenized_ref, labels=labels_ref).loss.item()
            #     return lls, lls + lls - lls_ref    

def get_lls(texts):
    if not args.openai_model:
        return [get_ll(text) for text in texts]
    else:
        global API_TOKEN_COUNTER

        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(args.batch_size)
        return pool.map(get_ll, texts)


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    assert args.openai_model is None, "get_rank not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
      
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)

        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_roc_metrics(real_preds, sample_preds):
    real_preds =  [element for element in real_preds if not math.isnan(element)]
    sample_preds = [element for element in sample_preds if not math.isnan(element)]

    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    real_preds =  [element for element in real_preds if not math.isnan(element)]
    sample_preds = [element for element in sample_preds if not math.isnan(element)]

    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


# save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors
def save_roc_curves(experiments):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='member')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='nonmember')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass


# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_llr_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='member')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='nonmember')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass


def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=500):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["nonmember"]
    sampled_text = data["member"]

    if args.ceil_pct:
        ceil_pct=True
    else:
        ceil_pct=False

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked,ceil_pct=ceil_pct)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "nonmember": original_text[idx],
            "member": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    load_base_model()
#    load_ref_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["nonmember"])
        res["sampled_ll"] = get_ll(res["member"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['nonmember']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['member']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


# def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
#     torch.manual_seed(0)
#     np.random.seed(0)

#     results = []
#     for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
#         original_text = data["nonmember"][batch * batch_size:(batch + 1) * batch_size]
#         sampled_text = data["member"][batch * batch_size:(batch + 1) * batch_size]

#         for idx in range(len(original_text)):
#             results.append({
#                 "nonmember": original_text[idx],
#                 "nonmember_crit": criterion_fn(original_text[idx]),
#                 "member": sampled_text[idx],
#                 "member_crit": criterion_fn(sampled_text[idx]),
#             })

#     # compute prediction scores for real/sampled passages
#     predictions = {
#         'real': [x["nonmember_crit"] for x in results],
#         'samples': [x["member_crit"] for x in results],
#     }

#     fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
#     p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
#     print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
#     return {
#         'name': f'{name}_threshold',
#         'predictions': predictions,
#         'info': {
#             'n_samples': n_samples,
#         },
#         'raw_results': results,
#         'metrics': {
#             'roc_auc': roc_auc,
#             'fpr': fpr,
#             'tpr': tpr,
#         },
#         'pr_metrics': {
#             'pr_auc': pr_auc,
#             'precision': p,
#             'recall': r,
#         },
#         'loss': 1 - pr_auc,
#     }


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = {
        "nonmember": [],
        "nonmember_crit": [],
        "nonmember_lls": [],
        "nonmember_meta": [],
        "member": [],
        "member_crit": [],
        "member_lls": [],
        "member_meta": []
    }
    for batch in tqdm.tqdm(range(len(data["nonmember"]) // batch_size + 1), desc=f"Computing {name} criterion for nonmember group"):
        original_text = data["nonmember"][batch * batch_size:(batch + 1) * batch_size]
        original_meta = data["nonmember_metadata"][batch * batch_size:(batch + 1) * batch_size]
        for idx in range(len(original_text)):
            lls, crit = criterion_fn(original_text[idx])
            results["nonmember"].append(original_text[idx])
            results["nonmember_crit"].append(crit)
            results["nonmember_lls"].append(lls)
            results["nonmember_meta"].append(original_meta[idx])


    for batch in tqdm.tqdm(range(len(data["member"]) // batch_size + 1), desc=f"Computing {name} criterion for member group"):
        sampled_text = data["member"][batch * batch_size:(batch + 1) * batch_size]
        sampled_meta = data["member_metadata"][batch * batch_size:(batch + 1) * batch_size]
        for idx in range(len(sampled_text)):
            lls, crit = criterion_fn(sampled_text[idx])
            results["member"].append(sampled_text[idx])
            results["member_crit"].append(crit)
            results["member_lls"].append(lls)
            results["member_meta"].append(sampled_meta[idx])

    # compute prediction scores for real/sampled passages
    results['name'] = f'{name}_threshold'
    results['n_samples'] = n_samples 

    return results


# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return ' '.join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    if args.max_length is not None:
        shorter_length = min(shorter_length,args.max_length)
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def generate_samples(raw_data_member, raw_data_non_member, meta_member, meta_non_member, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "nonmember": [],
        "member": [],
        "nonmember_metadata": [],
        "member_metadata": []
    }

    for batch in range(len(raw_data_non_member) // batch_size + 1):
        # print('Generating samples for batch', batch, 'of', len(raw_data_non_member) // batch_size + 1)
        non_member_text = raw_data_non_member[batch * batch_size:(batch + 1) * batch_size]
        non_member_meta = meta_non_member[batch * batch_size:(batch + 1) * batch_size]
        assert len(non_member_text) == len(non_member_meta)
        for o, om in zip(non_member_text, non_member_meta):
            data["nonmember"].append(o)
            data["nonmember_metadata"].append(om)
    for batch in range(len(raw_data_member) // batch_size + 1):
        # print('Generating samples for batch', batch, 'of', len(raw_data_member) // batch_size + 1)
        member_text = raw_data_member[batch * batch_size:(batch + 1) * batch_size]
        member_meta = meta_member[batch * batch_size:(batch + 1) * batch_size]
        assert len(member_text) == len(member_meta)
        for s, sm in zip(member_text, member_meta):
            data["member"].append(s)
            data["member_metadata"].append(sm)

    n_samples = len(data["nonmember"]) + len(data["member"])
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        load_mask_model()
        data["member"] = perturb_texts(data["member"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        load_base_model()

    return data, n_samples


def sample_segment(text, tokenizer_base, tokenizer_ref, max_length, strategy='random'):
    def random_segment(l, length, max_length):
        idx_random = random.randint(0, length-max_length)
        return l[idx_random: idx_random + max_length]

    def split_by_chapter(text):
        text = text[text.rfind("CHAPTER I"): ]
        chapters = text.split("CHAPTER")
        chapters = ["CHAPTER" + chapter for chapter in chapters]
        return chapters
    
    if "facebook/opt" in tokenizer_base.name_or_path or "facebook/opt" in tokenizer_ref.name_or_path \
        or "bigscience/bloom" in tokenizer_base.name_or_path or "bigscience/bloom" in tokenizer_ref.name_or_path \
        or "EleutherAI/pythia" in tokenizer_base.name_or_path or "EleutherAI/pythia" in tokenizer_ref.name_or_path:
        max_length = max_length - 1

    
    if strategy in ['random']:
        # Filter by number of words first to save compute
        n_words = len(text.split())
        if n_words > max_length:
            if strategy == 'random':
                text = random_segment(text, n_words, max_length)
        
        # Tokenize
        tokens_base = tokenizer_base.encode(text)
        tokens_ref = tokenizer_ref.encode(text)
        if len(tokens_base) > max_length or len(tokens_ref) > max_length:
            if len(tokens_base) > max_length:
                if strategy == 'random':
                    tokens_base = random_segment(tokens_base, len(tokens_base), max_length)
                else:
                    raise NotImplementedError()
                text = tokenizer_base.decode(tokens_base)
                tokens_ref = tokenizer_ref.encode(text)
            if len(tokens_ref) > max_length:
                if strategy == 'random':
                    tokens_ref = random_segment(tokens_ref, len(tokens_ref), max_length)
                else:
                    raise NotImplementedError()
                text = tokenizer_ref.decode(tokens_ref)
                tokens_base = tokenizer_base.encode(text)
        if isinstance(text, list):
            return text
        else:
            return [text]
    else:
        segment_length = 1024
        segments = []
        if strategy == 'split':
            tokens_base = tokenizer_base.encode(text)
            for i in range(0, len(tokens_base), segment_length):
                segments.append(tokenizer_base.decode(tokens_base[i: i+segment_length]))
        elif strategy == 'chapter':
            chapters = split_by_chapter(text)
            segments = []
            for chapter in chapters:
                n_words = len(chapter.split())
                if n_words > max_length:
                    if strategy == 'random':
                        chapter = random_segment(chapter, n_words, max_length)
                
                # Tokenize
                tokens_base = tokenizer_base.encode(chapter)
                tokens_ref = tokenizer_ref.encode(chapter)
                while len(tokens_base) >= max_length or len(tokens_ref) >= max_length:
                    if len(tokens_base) >= max_length:
                        if strategy == 'random':
                            tokens_base = random_segment(tokens_base, len(tokens_base), max_length)
                        else:
                            raise NotImplementedError()
                        chapter = tokenizer_base.decode(tokens_base)
                        tokens_ref = tokenizer_ref.encode(chapter)
                    if len(tokens_ref) >= max_length:
                        if strategy == 'random':
                            tokens_ref = random_segment(tokens_ref, len(tokens_ref), max_length)
                        else:
                            raise NotImplementedError()
                        chapter = tokenizer_base.decode(tokens_ref)
                        tokens_base = tokenizer_base.encode(chapter) 
                segments.append(chapter)
        return segments
        


def generate_data(dataset,key,train=True, strategy='random', SAVE_FOLDER=None, data_dir=None, membership_path=None, n_group=100, n_document_per_group=30, max_length=100000, instruct_model=None):
    random.seed(2023)
    np.random.seed(2023)
    metadata = None
    # load data
    data_split = 'train' if train else 'test'
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, cache_dir)
    elif dataset in pretraing_datasets.DATASETS:
        data, metadata = pretraing_datasets.load(dataset, data_dir=data_dir,
            membership_path=membership_path,
            n_group=n_group, n_document_per_group=n_document_per_group, train=train, SAVE_FOLDER=SAVE_FOLDER, instruct_model=instruct_model)
        assert len(data) == len(metadata)
    elif dataset == 'the_pile' and data_split=='train':
        #data_files = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
        #data_files="/home/niloofar/projects/enron_mail_20150507.tar.gz"
        #data_files ="/home/niloofar/projects/maildir"
        #data = datasets.load_dataset("json", data_files=data_files, split="train", cache_dir=cache_dir)[key]
        #data = datasets.load_dataset("json",data_files=data_files, split='train', cache_dir=cache_dir)[key]https://the-eye.eu/public/AI/pile/train/00.jsonl.zst"
        data = datasets.load_dataset("json", data_files="/gscratch/h2lab/sewon/data/the-pile/train-all/00.jsonl.zst",  split=f"{data_split}[:10000]", cache_dir=cache_dir)[key]
    elif dataset == 'the_pile' and data_split=='test':
        print("test")
        data = datasets.load_dataset("json", data_files="/gscratch/h2lab/sewon/data/the-pile/test.jsonl.zst",split=f"train[:10000]", cache_dir=cache_dir)[key]
    else:
        data = datasets.load_dataset(dataset, split=f'train[:10000]', cache_dir=cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    # data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    if not train:
        print(data[:10])

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 100 words
    # if dataset in ['writing', 'squad', 'xsum']:
    # long_data = [x for x in data if len(x.split()) > 100]
    # if len(long_data) > 0:
    #     data = long_data
    print(f"Total number of documents before long: {len(data)}")
    long_datas = [(x, y) for x, y in zip(data, metadata) if len(x.split()) > 0]
    (long_data, long_metadata) = zip(*long_datas)
    if len(long_data) > 0:
        data = long_data
        metadata = long_metadata

    print(f"Total number of documents after long: {len(data)}")
    
    # not_too_long_data = [x for x in data if len(x.split()) < max_length]
    # if len(not_too_long_data) > 0:
    #         data = not_too_long_data

    # random.seed(0)
    # random.shuffle(data)

    # data = data[:5_000]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    # tokenized_data = preproc_tokenizer(data)
    # tokenized_data_base = base_tokenizer(data)["input_ids"]
    # tokenized_data_ref = ref_tokenizer(data)["input_ids"]
    print("Sampling segments from each documents")
    print(f"Total number of documents: {len(data)}")
    print(type(data[0]))
    # data = [x for x, y, z in zip(data, tokenized_data_base, tokenized_data_ref) if len(y) <= max_length and len(z) <= max_length]
    # data = [sample_segment(dp, base_tokenizer, ref_tokenizer, max_length, strategy) for dp in data]
    
    print("Segmenting the dataset.")
    new_data = []
    new_metadata = []
    for dp, metadp in tqdm.tqdm(zip(data, metadata)):
        segments = sample_segment(dp, base_tokenizer, ref_tokenizer, max_length, strategy)
        assert isinstance(segments[0], str), type(segments[0])
        new_data.extend(segments)
        assert isinstance(new_data[0], str), new_data[0]
        new_metadata.extend([metadp] * len(segments))
    data = new_data
    metadata = new_metadata
    assert len(data) == len(metadata), (len(data), len(metadata))
    assert isinstance(data[0], str), data[0]
    # print stats about remainining data
    print(f"Total number of segments: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return data, metadata

    #return generate_samples(data[:n_samples], batch_size=batch_size)


def load_base_model_and_tokenizer(name):
    if "allenai/" in name:
        return recover("meta-llama/Llama-2-{}-hf".format(name.split('-')[-1]), name)


    if args.openai_model is None:
        print(f'Loading BASE model {name}...')
        base_model_kwargs = {'revision':args.revision}
        # if 'gpt-j' in name or 'neox' in name:
        #     base_model_kwargs.update(dict(torch_dtype=torch.float16))
        # if 'gpt-j' in name:
        #     base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    # if "facebook/opt-" in name or "allenai/tulu" in name:
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if "allenai/" in name:
        print("Using non-fast tokenizer for Tulu")
        optional_tok_kwargs['legacy'] = True
    if args.dataset_member in ['pubmed'] or args.dataset_nonmember in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    print(type(base_tokenizer))
    print("PAD TOKEN ID is: {}".format(base_tokenizer.pad_token_id))
    print("Vocab Size is: {}".format(base_tokenizer.vocab_size))
    return base_model, base_tokenizer


def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data['nonmember'], data['member']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        
        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_member', type=str, default="xsum")
    parser.add_argument('--dataset_member_key', type=str, default="document")
    parser.add_argument('--dataset_nonmember', type=str, default="xsum")
    parser.add_argument('--dataset_nonmember_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default=None)
    parser.add_argument('--revision', type=str, default="main")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="/trunk/model-hub")
    parser.add_argument('--ref_model', type=str, default=None)

    parser.add_argument('--tok_by_tok', action='store_true')

    parser.add_argument('--max_tries', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--ceil_pct', action='store_true')

    parser.add_argument('--n_group_member', type=int, default=100)
    parser.add_argument('--n_group_nonmember', type=int, default=100)
    parser.add_argument('--n_document_per_group', type=int, default=30)

    parser.add_argument('--strategy', type=str, default="random")

    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--membership_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="results")

    parser.add_argument('--min_k_prob', default=False, action='store_true')
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()

    API_TOKEN_COUNTER = 0

    if args.openai_model is not None:
        import openai
        assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = args.openai_key

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.base_model_name:
        if args.openai_model is None:
            base_model_name = args.base_model_name.replace('/', '_')
        else:
            base_model_name = "openai-" + args.openai_model.replace('/', '_')
    else:
        base_model_name = "None"
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
#    SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
    if args.ref_model is not None:
        ref_s=args.ref_model.replace('/', '_')
        ref_model_string = f'--ref_{ref_s}'
    else:
        ref_model_string = ""

    
    if args.tok_by_tok:
        tok_by_tok_string = '--tok_true'
    else:
        tok_by_tok_string = '--tok_false'

    if args.span_length ==2 :
        span_length_string = ""
    else:
        span_length_string = f'--{args.span_length}'

    if args.max_length is not None :
        max_length_string = f'-m{args.max_length}'
    else:
        max_length_string = ""

    if args.min_k_prob:
        min_k_prob_string = "--min_k"
    else:
        min_k_prob_string = ""

    if args.name:
        name_string = f'--{args.name}'
    else:
        name_string = ""

    dataset_member_name=args.dataset_member.replace('/', '_')
    dataset_nonmember_name=args.dataset_nonmember.replace('/', '_')

    # SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}-{args.revision}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{dataset_member_name}-{dataset_nonmember_name}-{args.n_group_member}-{args.n_group_nonmember}-{args.n_document_per_group}{ref_model_string}{span_length_string}{max_length_string}{tok_by_tok_string}"
    # SAVE_FOLDER = f"{args.save_dir}/{output_subfolder}{base_model_name}-{args.revision}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{dataset_member_name}-{dataset_nonmember_name}-{args.n_group_member}-{args.n_group_nonmember}-{args.n_document_per_group}{ref_model_string}{span_length_string}{max_length_string}{tok_by_tok_string}"
    SAVE_FOLDER = f"{args.save_dir}/{dataset_member_name}-{args.n_group_member}-{args.n_group_nonmember}-{args.n_document_per_group}{max_length_string}{name_string}/{base_model_name}{min_k_prob_string}"

    # new_folder = SAVE_FOLDER.replace("tmp_results", args.save_dir)
    # ##don't run if exists!!!
    # print(f"{new_folder}")
    # if  os.path.isdir((new_folder)):
    #     print(f"folder exists, not running this exp {new_folder}")
    #     exit(0)

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples
    min_k_prob = args.min_k_prob

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

    # generic generative model
    if args.base_model_name:
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)


    #reference model if we are doing the lr baseline
    if args.ref_model is not None :
        ref_model, ref_tokenizer = load_base_model_and_tokenizer(args.ref_model)
        load_ref_model()
    else:
        ref_model, ref_tokenizer = None, base_tokenizer

    if not args.base_model_name:
        base_model, base_tokenizer = None, ref_tokenizer

    # mask filling t5 model
    if not args.baselines_only and not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
    # if args.dataset in ['english', 'german']:
    #     preproc_tokenizer = mask_tokenizer

    if args.base_model_name:
        load_base_model(base_model)

    print(f'Loading dataset {args.dataset_member} and {args.dataset_nonmember}...')
    # data, seq_lens, n_samples = generate_data(args.dataset_member,args.dataset_member_key)
    try:
        base_model_n_position = base_model.config.max_position_embeddings
    except AttributeError:
        try:
            base_model_n_position = base_model.config.n_positions
        except AttributeError:
            base_model_n_position = 2048
    if args.ref_model is not None:
        try:
            ref_model_n_position = ref_model.config.max_position_embeddings
        except AttributeError:
            try:
                ref_model_n_position = ref_model.config.n_positions
            except AttributeError:
                ref_model_n_position = 2048
    else:
        ref_model_n_position = base_model_n_position
    longest_tokenizable_len = min(base_model_n_position, ref_model_n_position)
    print(f'The longest tokenizable length of is {longest_tokenizable_len}.')
    if args.max_length:
        assert args.max_length <= longest_tokenizable_len

    if args.dataset_member == "tuning":
        instruct_model_member = args.base_model_name
    else:
        instruct_model_member = None
    if args.dataset_nonmember == "tuning":
        instruct_model_nonmember = args.base_model_name
    else:
        instruct_model_nonmember = None
        
    
    data_member, metadata_member = generate_data(args.dataset_member,args.dataset_member_key, train=True, 
                                                 strategy=args.strategy, n_group=args.n_group_member, 
                                                 n_document_per_group=args.n_document_per_group, 
                                                 SAVE_FOLDER=SAVE_FOLDER, data_dir=args.data_dir, membership_path=args.membership_path, 
                                                 max_length=args.max_length if args.max_length else longest_tokenizable_len,
                                                 instruct_model=instruct_model_member)
    data_nonmember, metadata_nonmember = generate_data(args.dataset_nonmember, args.dataset_nonmember_key, train=False, 
                                                       strategy=args.strategy, n_group=args.n_group_nonmember, 
                                                       n_document_per_group=args.n_document_per_group, 
                                                       SAVE_FOLDER=SAVE_FOLDER, data_dir=args.data_dir, membership_path=args.membership_path, 
                                                       max_length=args.max_length if args.max_length else longest_tokenizable_len,
                                                       instruct_model=instruct_model_nonmember)

    # assert len(data_member) == len(data_nonmember)
    print(f'Loaded {len(data_member)} members and {len(data_nonmember)} non-members.')
    
    # n_samples = min([len(data_member), len(data_nonmember), n_samples])
    # data, seq_lens, n_samples = generate_samples(data_member[:n_samples], data_nonmember[:n_samples], batch_size=batch_size)
    data, n_samples = generate_samples(data_member, data_nonmember, metadata_member, metadata_nonmember, batch_size=batch_size)

    print("Total number of datapoints: {} (members: {}, non-members: {})".format(n_samples, len(data["member"]), len(data["nonmember"])))
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

    if args.scoring_model_name:
        print(f'Loading SCORING model {args.scoring_model_name}...')
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.scoring_model_name)
        load_base_model()  # Load again because we've deleted/replaced the old model

    # write the data to a json file in the save folder
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(data, f)

    # with open(os.path.join(SAVE_FOLDER, "raw_data_lens.json"), "w") as f:
    #     print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data_lens.json')}")
    #     json.dump(seq_lens, f)

    if not args.skip_baselines:
        # baseline_outputs = [run_baseline_threshold_experiment(get_ll, "likelihood", n_samples=n_samples)]
        baseline_outputs = []

        if args.openai_model is None:
            # rank_criterion = lambda text: -get_rank(text, log=False)
            # baseline_outputs.append(run_baseline_threshold_experiment(rank_criterion, "rank", n_samples=n_samples))
            # logrank_criterion = lambda text: -get_rank(text, log=True)
            # baseline_outputs.append(run_baseline_threshold_experiment(logrank_criterion, "log_rank", n_samples=n_samples))
            # entropy_criterion = lambda text: get_entropy(text)
            # baseline_outputs.append(run_baseline_threshold_experiment(entropy_criterion, "entropy", n_samples=n_samples))
            # if args.ref_model is not None:
            baseline_outputs.append(run_baseline_threshold_experiment(get_lira, "lr_ratio", n_samples=n_samples))

        # baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
        # baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))

    outputs = []

    if not args.baselines_only:
        # run perturbation experiments
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results(args.span_length, n_perturbations, n_samples)
            for perturbation_mode in ['d', 'z']:
                output = run_perturbation_experiment(
                    perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
                outputs.append(output)
                with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "w") as f:
                    json.dump(output, f)

    if not args.skip_baselines:
        # write likelihood threshold results to a file
        # with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
        #     json.dump(baseline_outputs[0], f)

        if args.openai_model is None:
            # write rank threshold results to a file
            # with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
            #     json.dump(baseline_outputs[1], f)

            # # write log rank threshold results to a file
            # with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
            #     json.dump(baseline_outputs[2], f)

            # # write entropy threshold results to a file
            # with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
            #     json.dump(baseline_outputs[3], f)
            # if args.ref_model is not None:
            with open(os.path.join(SAVE_FOLDER, f"lr_ratio_threshold_results.json"), "w") as f:
                # json.dump(baseline_outputs[4], f)
                json.dump(baseline_outputs[0], f)


        # # write supervised results to a file
        # with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
        #     json.dump(baseline_outputs[-2], f)
        
        # # write supervised results to a file
        # with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
        #     json.dump(baseline_outputs[-1], f)

        outputs += baseline_outputs

    # save_roc_curves(outputs)
    # save_ll_histograms(outputs)
    # save_llr_histograms(outputs)

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
    # new_folder = SAVE_FOLDER.replace("tmp_results", args.save_dir)
    # if not os.path.exists(os.path.dirname(new_folder)):
    #     os.makedirs(os.path.dirname(new_folder))
    # os.rename(SAVE_FOLDER, new_folder)

    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")