from collections import defaultdict, OrderedDict
import glob
import json
import mmap
import numpy as np
import os
import random
# from scipy.stats import entropy
import sys
import pickle as pkl
import time
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXTokenizerFast

assert sys.byteorder == 'little'
PAGESIZE = int(os.sysconf(os.sysconf_names['SC_PAGESIZE']))

with open("hf_access_token.txt") as f:
    access_token = f.read().strip()

'''
Variable Symbol Value/Range Meaning
         S                  Number of shards of the datastore
         s      [0, S)      Index of datastore
tot_cnt         T           Number of tokens in the ds
ds_size         2 * T       Number of bytes in the ds
sa_cnt          T           Number of elements in the sa
sa_size         P * T       Number of bytes in the sa
ptr_size P      [1, 8]      Number of bytes per pointer
rank            [0, T)      Rank in the sa
ptr             [0, 2 * T)  Pointer into the ds, in bytes (must be even)
offset          [0, 2 * T)  Offset into the ds, in bytes (must be even)
'''

def load_tokenizer(tokenizer_type):
    if tokenizer_type=="llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=access_token)
    elif tokenizer_type=="neox":
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    return tokenizer


class NGramLanguageModeling(object):

    def __init__(self,
                 cache_dir,
                 tokenizer_type,
                 force_cache=False,
                 dstore_size_reduction=1.0,
                 dstore_domain=None,
                 ):

        '''
        tokenizer_type: either llama or gpt2
        data_dir (str)
            - If you'd like to use the V2 datastore, please set it to '/private/home/ljc/ha-memotrap/data/pile_v2'
            - If you'd like to use the V3 datastore, please leave it as None
        dstore_size_reduction (float/int)
            - 1.0 means the full dstore, 2.0 means 2.0x smaller dstore, etc
        dstore_domain (str)
        '''

        self.tokenizer_type = tokenizer_type
        self.tokenizer = load_tokenizer(tokenizer_type)
        '''
        combinations we are likely to need
        - dstore_size_reduction=1.0, dstore_domain=None (the default setting)
        - dstore_size_reduction=1.0, dstore_domain={placeholder} (Gary: this might be useful)
        - dstore_size_reduction={2.0, 4.0, 8.0, 16.0, 32.0}, dstore_domain=None (for dstore size ablations)
        - dstore_size_reduction={2.0, 4.0, 8.0, 16.0, 32.0}, dstore_domain={placeholder}
            (for the graph we discussed last Monday - probably 3-4 domains are enough)
        '''

        if dstore_domain is None:
            if tokenizer_type == 'gpt2':
                if dstore_size_reduction <= 2.0:
                    self.data_dir = '/checkpoint/ljc/ha-memotrap/data/pile_v3_c2'
                else:
                    self.data_dir = '/checkpoint/ljc/ha-memotrap/data/pile_v3_c32'
            elif tokenizer_type == 'llama':
                if dstore_size_reduction <= 2.0:
                    self.data_dir = '/checkpoint/ljc/ha-memotrap/data/pile_v3_c2_llama2'
                elif dstore_size_reduction <= 32:
                    self.data_dir = "/checkpoint/ljc/ha-memotrap/data/pile_v3_c32_llama2"
                else:
                    self.data_dir = "/checkpoint/sewonmin/data/pile_v3_c360_llama2"
            elif tokenizer_type == "neox":
                if dstore_size_reduction <= 4.0:
                    self.data_dir = "/checkpoint/sewonmin/data/pile-neox"
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif dstore_domain=="rpj":
            assert tokenizer_type=="llama"
            assert dstore_size_reduction <= 8
            self.data_dir = "/large_experiments/cmd/ngram_datastore/redpajama_bff_v3_c8_llama2"
        elif tokenizer_type == "gpt2":
            if dstore_size_reduction == 1.0:
                domain = dstore_domain.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                # self.data_dir = f'/checkpoint/ljc/ha-memotrap/data/pile_v3_domain/{domain}'
                self.data_dir = "/gscratch/h2lab/sewon/data/"
            else:
                raise NotImplementedError
        elif tokenizer_type == "llama":
            domain = dstore_domain.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            self.data_dir = f'/checkpoint/ljc/ha-memotrap/data/pile_v3_c32_domain_llama2/{domain}'
        else:
            raise NotImplementedError()

        self.cache = {}
        self.n_cache = 0
        assert os.path.isdir(self.data_dir), self.data_dir

        ####################################################################

        print ("Loading dstore from", self.data_dir)

        self.datastores = [] # Each datastore represents a chunk of the entire corpus
        if 'v2' in self.data_dir:
            ds_name = 'pile.train'
            ds_path = f'{self.data_dir}/{ds_name}'
            sa_path = f'{ds_path}.table.bin'

            f_ds = open(ds_path, 'rb')
            ds = mmap.mmap(f_ds.fileno(), 0, prot=mmap.PROT_READ)
            f_sa = open(sa_path, 'rb')
            sa = mmap.mmap(f_sa.fileno(), 0, prot=mmap.PROT_READ)

            # Get the size of pointers
            ds_size = os.path.getsize(ds_path)
            sa_size = os.path.getsize(sa_path)
            assert sa_size % ds_size == 0
            ptr_size = sa_size // ds_size
            tot_cnt = ds_size // 2 # total number of tokens
            sa_cnt = ds_size # number of elements in the suffix array

            datastore = { 'ds': ds, 'sa': sa, 'tot_cnt': tot_cnt, 'ds_size': ds_size, 'sa_cnt': sa_cnt, 'ptr_size': ptr_size }
            self.datastores.append(datastore)
        else: # v3 or above
            ds_path_base = os.path.join(self.data_dir, 'tokenized')
            sa_path_base = os.path.join(self.data_dir, 'table')

            ds_paths = sorted(glob.glob(f'{ds_path_base}*'))
            sa_paths = sorted(glob.glob(f'{sa_path_base}*'))
            assert len(ds_paths) == len(sa_paths), (ds_paths, sa_paths)

            for (ds_path, sa_path) in zip(ds_paths, sa_paths):
                f_ds = open(ds_path, 'rb')
                ds = mmap.mmap(f_ds.fileno(), 0, prot=mmap.PROT_READ)
                ds.madvise(mmap.MADV_RANDOM)
                # ds = f_ds.read()
                f_sa = open(sa_path, 'rb')
                sa = mmap.mmap(f_sa.fileno(), 0, prot=mmap.PROT_READ)
                sa.madvise(mmap.MADV_RANDOM)

                ds_size = os.path.getsize(ds_path)
                sa_size = os.path.getsize(sa_path)
                assert ds_size % 2 == 0 # 2 bytes per token
                tot_cnt = ds_size // 2 # total number of tokens
                assert sa_size % tot_cnt == 0
                ptr_size = sa_size // tot_cnt # size of each pointer
                sa_cnt = tot_cnt # number of elements in the suffix array

                datastore = { 'ds': ds, 'sa': sa, 'tot_cnt': tot_cnt, 'ds_size': ds_size, 'sa_cnt': sa_cnt, 'ptr_size': ptr_size }
                self.datastores.append(datastore)

        if dstore_size_reduction != 1.0:
            # Take the few datastore shards
            assert int(dstore_size_reduction) == dstore_size_reduction
            dstore_size_reduction = int(dstore_size_reduction)
            assert len(self.datastores) % dstore_size_reduction == 0, (len(self.datastores), dstore_size_reduction)
            self.datastores = self.datastores[:len(self.datastores) // dstore_size_reduction]

        def add_commas_to_number(number):
            number_str = str(number)

            # Determine the position where the first comma should be inserted
            first_comma_position = len(number_str) % 3 if len(number_str) % 3 != 0 else 3

            # Initialize the result string with the first part of the number
            result = number_str[:first_comma_position]

            # Iterate through the remaining parts of the number, adding commas
            for i in range(first_comma_position, len(number_str), 3):
                result += ',' + number_str[i:i + 3]

            return result

        # print ("Finished loading the datastore with %s tokens" % add_commas_to_number(np.sum([ds["tot_cnt"] for ds in self.datastores])))

        ####################################################################
        self.cache_dir = cache_dir
        self.vocab_size = len(self.tokenizer)-1 # excluding the padding token
        self.force_cache = force_cache

    def load_cache(self, cache_prefix):
        self.cache_path = os.path.join(self.cache_dir, "{}ngram-language-modeling-{}-cache.pkl".format(
                        "" if cache_prefix is None else cache_prefix + "_", self.tokenizer_type))
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.cache = pkl.load(f)
            print ("Loaded %d cache items from %s" % (len(self.cache), self.cache_path))
        else:
            self.cache = {}
        self.n_cache = 0

    def save_cache(self):
        if self.n_cache > 0:
            # other processes might have updated the cache already
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    self.cache.update(pkl.load(f))

            with open(self.cache_path, "wb") as f:
                pkl.dump(self.cache, f)
            self.n_cache = 0

    def compute_prob(self, prompt_ids, proposed_ids, eps, prompt_cnt_threshold=0, exclude_eos=False):
        '''
        Inputs:
        - prompt_ids [int]
        - proposed_ids [int] (len=1)
        Outputs:
        - prob float
        - prompt_cnt int
        - cont_cnt int
        '''
        assert type(prompt_ids)==type(proposed_ids)==list
        assert len(prompt_ids)>=0 and len(proposed_ids)==1

        prompt_find_outputs = self.find(prompt_ids)
        if exclude_eos:
            exclude_outputs = self.find(prompt_ids + [65535], hint_segments=prompt_find_outputs['segments'])
            prompt_find_outputs['cnt'] -= exclude_outputs['cnt']

            if type(prompt_find_outputs["segments"])==type(exclude_outputs["segments"])==dict:
                prompt_find_outputs['segments'] = {
                        k: [(segment[0], exclude_segment[0])
                            for (segment, exclude_segment)
                            in zip(prompt_find_outputs["segments"][k], exclude_outputs['segments'][k])]
                        for k in prompt_find_outputs["segments"]}
            else:
                prompt_find_outputs['segments'] = [(segment[0], exclude_segment[0]) for (segment, exclude_segment) in zip(prompt_find_outputs['segments'], exclude_outputs['segments'])]

        prompt_cnt = prompt_find_outputs['cnt']
        cont_find_outputs = self.find(prompt_ids + proposed_ids, hint_segments=prompt_find_outputs['segments'])
        cont_cnt = cont_find_outputs['cnt']
        prob = (cont_cnt + eps) / (prompt_cnt + self.vocab_size * eps) if prompt_cnt > prompt_cnt_threshold else None
        return {"prob": prob, "prompt_cnt": prompt_cnt, "cont_cnt": cont_cnt}

    def get_next_token_distribution(self, prompt_ids, eps, exclude_eos=False):
        '''
        Inputs:
        - prompt_ids [int]
                            i v in
        Outputs:
        - prompt_cnt int: total number of occurrences of the prompt
        - freq_by_token_id dict[int->int]: frequency of each token
        - prob_by_token_id dict[int->float]: probability of each token
        Note that it is possible that cnt == 0, in which case the other two outputs are empty dicts.
        '''
        assert type(prompt_ids)==list
        assert len(prompt_ids)>=0

        freq_by_token_id = defaultdict(int)

        find_outputs = self.find(prompt_ids)
        if exclude_eos:
            assert type(find_outputs["segments"])==list
            exclude_outputs = self.find(prompt_ids + [65535], hint_segments=find_outputs['segments'])
            find_outputs['cnt'] -= exclude_outputs['cnt']
            find_outputs['segments'] = [(segment[0], exclude_segment[0]) for (segment, exclude_segment) in zip(find_outputs['segments'], exclude_outputs['segments'])]

        segments = find_outputs['segments']
        for datastore, segment in zip(self.datastores, segments):
            ds, sa, ds_size, ptr_size = datastore['ds'], datastore['sa'], datastore['ds_size'], datastore['ptr_size']
            (start, end) = segment
            for rank in range(start, end):
                ptr = self.convert_rank_to_ptr(sa, rank, ptr_size)
                offset = ptr + 2 * len(prompt_ids)
                if offset >= ds_size:
                    continue
                token_id = self.convert_offset_to_token_id(ds, offset)
                freq_by_token_id[token_id] += 1

        prompt_cnt = sum(freq_by_token_id.values())
        prob_by_token_id = {token_id: (freq + eps) / (prompt_cnt + self.vocab_size * eps) for token_id, freq in freq_by_token_id.items()}
        return dict(prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)

    def get_next_token_distribution_fast(self, prompt_ids, eps, exclude_eos=False):
        '''
        The fast version of get_next_token_distribution()
        '''
        find_outputs = self.find(prompt_ids)
        if exclude_eos:
            assert type(find_outputs["segments"])==list
            exclude_outputs = self.find(prompt_ids + [65535], hint_segments=find_outputs['segments'])
            find_outputs['cnt'] -= exclude_outputs['cnt']
            find_outputs['segments'] = [(segment[0], exclude_segment[0]) for (segment, exclude_segment) in zip(find_outputs['segments'], exclude_outputs['segments'])]
        segments = find_outputs['segments']
        freq_by_token_id = defaultdict(int)
        for datastore, segment in zip(self.datastores, segments):
            ds, sa, ds_size, ptr_size = datastore['ds'], datastore['sa'], datastore['ds_size'], datastore['ptr_size']
            result = self._get_freq_by_token_id(prompt_ids, segment, ds, sa, ds_size, ptr_size)
            for token_id, freq in result.items():
                freq_by_token_id[token_id] += freq
        prompt_cnt = sum(freq_by_token_id.values())
        prob_by_token_id = {token_id: (freq + eps) / (prompt_cnt + self.vocab_size * eps) for token_id, freq in freq_by_token_id.items()}

        freq_by_token_id["others"] = 0
        prob_by_token_id["others"] = eps / (prompt_cnt + self.vocab_size * eps)

        return dict(prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)

    def _get_freq_by_token_id(
            self, prompt_ids, segment, ds, sa, ds_size, ptr_size,
            token_start=None, token_end=None,
            num_nonzero_threshold=None):

        freq_by_token_id = OrderedDict()
        (start, end) = segment


        def prefetch(lo, hi, depth=0):
            mi = (lo + hi) // 2 # sa index to inspect
            if mi <= 0: # prefetching when mi <= 0 will cause page errors
                return
            if depth == 1: # fetch ds
                ptr = self.convert_rank_to_ptr(sa, mi-1, ptr_size)
                offset = ptr + 2 * len(prompt_ids)
                ds.madvise(mmap.MADV_WILLNEED, offset - offset % PAGESIZE, 2 + offset % PAGESIZE)
                ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                offset = ptr + 2 * len(prompt_ids)
                ds.madvise(mmap.MADV_WILLNEED, offset - offset % PAGESIZE, 2 + offset % PAGESIZE)
            elif depth == 3: # fetch sa
                sa.madvise(mmap.MADV_WILLNEED, (mi-1)*ptr_size - (mi-1)*ptr_size % PAGESIZE, 2*ptr_size + (mi-1)*ptr_size % PAGESIZE) # since we need both mi-1 and mi
                return
            prefetch(lo, mi, depth+1)
            prefetch(mi, hi, depth+1)
        prefetch(start, end)

        # Trivial case
        if end - start < 4:
            for rank in range(start, end):
                ptr = self.convert_rank_to_ptr(sa, rank, ptr_size)
                offset = ptr + 2 * len(prompt_ids)
                if offset >= ds_size:
                    continue
                token_id = self.convert_offset_to_token_id(ds, offset)
                if token_id not in freq_by_token_id:
                    freq_by_token_id[token_id] = 0
                freq_by_token_id[token_id] += 1
            return freq_by_token_id

        # If start and end-1 has the same token, then we know this segment is all the same token
        if token_start is None:
            ptr_start = self.convert_rank_to_ptr(sa, start, ptr_size)
            token_start = self.convert_offset_to_token_id(ds, ptr_start + 2 * len(prompt_ids))
        if token_end is None:
            ptr_end = self.convert_rank_to_ptr(sa, end-1, ptr_size)
            token_end = self.convert_offset_to_token_id(ds, ptr_end + 2 * len(prompt_ids))
        if token_start == token_end:
            freq_by_token_id[token_start] = end - start
            return freq_by_token_id

        # Otherwise, we do divide and conquer
        mi = (start + end) // 2
        left_freq_by_token_id = self._get_freq_by_token_id(prompt_ids, (start, mi), ds, sa, ds_size, ptr_size, token_start=token_start)
        right_freq_by_token_id = self._get_freq_by_token_id(prompt_ids, (mi, end), ds, sa, ds_size, ptr_size, token_end=token_end)
        if next(reversed(left_freq_by_token_id)) != next(iter(right_freq_by_token_id)):
            freq_by_token_id = left_freq_by_token_id
            freq_by_token_id.update(right_freq_by_token_id)
        else:
            token_id, freq = left_freq_by_token_id.popitem(last=True)
            token_id2, freq2 = right_freq_by_token_id.popitem(last=False)
            assert token_id == token_id2
            freq_by_token_id = left_freq_by_token_id
            freq_by_token_id[token_id] = freq + freq2
            freq_by_token_id.update(right_freq_by_token_id)

        # for token_id, freq in left_freq_by_token_id.items():
        #     freq_by_token_id[token_id] += freq
        # for token_id, freq in right_freq_by_token_id.items():
        #     freq_by_token_id[token_id] += freq

        return freq_by_token_id

    def get_next_token_distribution_onehot(self, prompt_ids, eps, exclude_eos=False):
        '''
        Faster version of get_next_token_distribution() assuming that will only succeed when the next token is unique.
        Inputs:
        - prompt_ids [int]
        Outputs:
        - result str: one of 'zero', 'one', or 'multi', indicating the number of distinct next tokens (Check this field first! If it is not 'one', then the other fields are missing.)
        - prompt_cnt int: total number of occurrences of the prompt
        - freq_by_token_id dict[int->int]: frequency of each token
        - prob_by_token_id dict[int->float]: probability of each token
        '''
        assert type(prompt_ids)==list
        assert len(prompt_ids)>=0

        find_outputs = self.find(prompt_ids)
        if exclude_eos:
            assert type(find_outputs["segments"])==list
            exclude_outputs = self.find(prompt_ids + [65535], hint_segments=find_outputs['segments'])
            find_outputs['cnt'] -= exclude_outputs['cnt']
            find_outputs['segments'] = [(segment[0], exclude_segment[0]) for (segment, exclude_segment) in zip(find_outputs['segments'], exclude_outputs['segments'])]
        segments = find_outputs['segments']
        onehot_token_id = None
        is_onehot = True
        found = False
        for s, segment in enumerate(segments):
            (start, end) = segment
            if start == end:
                continue
            found = True
            first_token_id = self.get_ds_token_id(s, start, length=len(prompt_ids))
            last_token_id = self.get_ds_token_id(s, end-1, length=len(prompt_ids))
            if onehot_token_id is None:
                onehot_token_id = first_token_id
            else:
                if onehot_token_id != first_token_id:
                    is_onehot = False
            if onehot_token_id is None:
                onehot_token_id = last_token_id
            else:
                if onehot_token_id != last_token_id:
                    is_onehot = False
        if not found:
            return dict(result='zero')
        elif not is_onehot:
            return dict(result='multi')

        freq_by_token_id = defaultdict(int)
        freq_by_token_id[onehot_token_id] = find_outputs['cnt']
        freq_by_token_id["others"] = 0

        cnt = find_outputs["cnt"]
        prob_by_token_id = defaultdict(float)
        prob_by_token_id[onehot_token_id] = (cnt + eps) / (cnt + self.vocab_size * eps)
        prob_by_token_id["others"] = eps / (cnt + self.vocab_size * eps)

        return dict(
                result='one',
                prompt_cnt=find_outputs['cnt'],
                freq_by_token_id=freq_by_token_id,
                prob_by_token_id=prob_by_token_id)

    def get_original_documents(self, prompt_ids, MAX_DOCUMENTS=10):
        '''
        Inputs:
        - prompt_ids [int]
        Outputs:
        - documents [Document]: Each document is a dict with the following fields:
            - string str: the string of the document
            - char_offset int: the character offset of the prompt in the document
            - token_ids (uint16, len=doc_length): the token IDs of the document
            - token_offset int: the token offset of the prompt in the document
        '''

        find_outputs = self.find(prompt_ids)
        segments = find_outputs['segments']
        documents = []
        for datastore, segment in zip(self.datastores, segments):
            ds, sa, ptr_size = datastore['ds'], datastore['sa'], datastore['ptr_size']
            (start, end) = segment
            for rank in range(start, end):
                ptr = self.convert_rank_to_ptr(sa, rank, ptr_size)

                start_ptr = ds.rfind(b'\xff\xff', 0, ptr)
                assert start_ptr != -1 # because the beginning of each ds should be \xff\xff
                # In rare occasions, the document UID may contain \xff\xff, and our marker search may end up there
                # For example, when the UID is 0x0000ffff, the document begins with \xff\xff\xff\xff\x00\x00
                # as another example, when the UID is 0x00ffff12, the document begins with \xff\xff\x12\xff\xff\x00
                # For now, we assume that the UID cannot be 0xffff****
                # The following lines account for such corner case
                if start_ptr % 2 == 1 and start_ptr >= 3 and ds[start_ptr-3:start_ptr-1] == b'\xff\xff':
                    start_ptr -= 3
                elif start_ptr % 2 == 0 and start_ptr >= 2 and ds[start_ptr-2:start_ptr] == b'\xff\xff':
                    start_ptr -= 2
                elif start_ptr % 2 == 1 and start_ptr >= 1 and ds[start_ptr-1:start_ptr+1] == b'\xff\xff':
                    start_ptr -= 1
                start_ptr += 6 # '\xff\xff' + 4 bytes for the UID
                assert start_ptr % 2 == 0

                end_ptr = ds.find(b'\xff\xff', ptr)
                if end_ptr == -1:
                    end_ptr = len(ds)
                assert end_ptr % 2 == 0

                token_buf = np.frombuffer(ds[start_ptr : end_ptr], dtype=np.uint8)
                token_ids = token_buf.view(np.uint16)
                string = self.tokenizer.decode(token_ids)
                token_offset = (ptr - start_ptr) // 2
                prefix_string = self.tokenizer.decode(token_ids[:token_offset])
                char_offset = len(prefix_string)
                documents.append(dict(string=string, char_offset=char_offset, token_ids=token_ids, token_offset=token_offset))
                if len(documents) >= MAX_DOCUMENTS:
                    return documents
        return documents

    def get_all_documents(self, start_idx, end_idx, hint=None):
        '''
        return (start_idx)-th to (end_idx)-th documents in the datastore
        Inputs:
        - start_idx int
        - end_idx int
        - hint (int, int): hint returned by the previous query
            - The first number is the index of the datastore shard
            - The second number is the byte offset in the datastore shard
            - hint should be None when start_idx == 0
            - If hint is not None, then it must be the hint returned by the previous query whose end_idx is this query's start_idx
        Outputs:
        - documents [(uint16, len=end_idx-start_idx)]: Each document is a np array of token IDs
            - Note: the length of documents may be less than expected when done=True
        - hint (int, int): hint returned by the current query
        - done bool: indicates whether all documents have been returned. This function should not be called again if done is True
        Usage:
        - hint = None
        - start_idx = 0
        - while True:
        -    end_idx = start_idx + 1000
        -    output = lm.get_all_documents(start_idx, end_idx, hint)
        -    # process output['documents']
        -    if output['done']:
        -        break
        -    hint = output['hint']
        -    start_idx = end_idx
        '''
        documents = []
        if hint is None:
            s, start_ptr = 0, 0
            idx = 0
        else:
            (s, start_ptr) = hint
            idx = start_idx

        if len(self.datastores) <= s:
            return dict(documents=[], hint=None, done=True)

        ds = self.datastores[s]['ds']
        for i in range(idx, end_idx):
            ptr = start_ptr + 6
            end_ptr = ds.find(b'\xff\xff', ptr)
            if end_ptr == -1:
                end_ptr = len(ds)
            if start_idx <= i < end_idx:
                token_buf = np.frombuffer(ds[ptr : end_ptr], dtype=np.uint8)
                token_ids = token_buf.view(np.uint16)
                documents.append(token_ids)
            if end_ptr == len(ds):
                s += 1
                start_ptr = 0
                if s == len(self.datastores):
                    return dict(documents=documents, hint=(s, start_ptr), done=True)
                ds = self.datastores[s]['ds']
            else:
                start_ptr = end_ptr
        return dict(documents=documents, hint=(s, start_ptr), done=False)

    def get_ds_tokens(self, s, rank, length):
        '''
        Inputs:
        - s int: index of the datastore shard
        - rank int: index in the suffix array
        - length int: number of tokens to return
        Outputs:
        - buf (uint8, len=2*length): the raw bytes in ds
        - token_ids (uint16, len=length): the token ids
        '''
        datastore = self.datastores[s]
        ds, sa, ptr_size = datastore['ds'], datastore['sa'], datastore['ptr_size']
        ptr = self.convert_rank_to_ptr(sa, rank, ptr_size)
        token_buf = np.frombuffer(ds[ptr : (ptr + 2 * length)], dtype=np.uint8)
        token_ids = token_buf.view(np.uint16)
        return dict(token_buf=token_buf, token_ids=token_ids)

    def get_ds_token_id(self, s, rank, length):
        '''
        Inputs:
        - s int: index of the datastore shard
        - rank int: index in the suffix array
        - length int: number of tokens to read before the token is taken
        Outputs:
        - token_id int: the token id desired
        '''
        datastore = self.datastores[s]
        ds, sa, ptr_size = datastore['ds'], datastore['sa'], datastore['ptr_size']
        ptr = self.convert_rank_to_ptr(sa, rank, ptr_size)
        offset = ptr + 2 * length
        token_id = self.convert_offset_to_token_id(ds, offset)
        return token_id

    def find(self, prompt_ids, hint_segments=None, return_boolean=False, cnt_threshold=0):
        assert self.cache is not None

        cache_key = " ".join([str(i) for i in prompt_ids])

        if return_boolean:
            if cache_key in self.cache:
                return self.cache[cache_key]["cnt"] > cnt_threshold
            elif cache_key + ":bool" in self.cache:
                result = self.cache[cache_key + ":bool"]
            else:
                assert not self.force_cache
                self.cache[cache_key + ":bool"] = self._find(
                        prompt_ids,
                        hint_segments=hint_segments,
                        return_boolean=True,
                        cnt_threshold=cnt_threshold)
                self.n_cache += 1
                result = self.cache[cache_key + ":bool"]

        else:
            if cache_key not in self.cache:
                assert not self.force_cache, cache_key
                self.cache[cache_key] = self._find(prompt_ids, hint_segments=hint_segments)
                self.n_cache += 1

            result = self.cache[cache_key]

        assert result["cnt"] >= 0
        return {k: v.copy() if k=="segments" else v for k, v in result.items()}


    def _find(self, prompt_ids, hint_segments=None, return_boolean=False, cnt_threshold=0):
        '''
        Inputs:
        - prompt_ids [int]
        - hint_segments: [(int, int) * S]: hint from output of a previous search, whose prompt_ids was a prefix of the current prompt_ids
        Outputs:
        - segments [(int, int) * S]: starting and ending ranks in the sa, per datastore shard
        - cnt int: total number of occurrences of the prompt
        '''
        assert type(prompt_ids) == list
        assert len(prompt_ids) >= 0
        if hint_segments is None:
            hint_segments = [None] * len(self.datastores)
        assert type(hint_segments) == list
        assert len(hint_segments) == len(self.datastores)

        prompt_buf = np.array(prompt_ids, dtype=np.uint16).view(np.uint8).tobytes()
        segments = []
        cnt = 0

        for datastore, hint_segment in zip(self.datastores, hint_segments):
            ds, sa, tot_cnt, sa_cnt, ptr_size = datastore['ds'], datastore['sa'], datastore['tot_cnt'], datastore['sa_cnt'], datastore['ptr_size']
            if len(prompt_ids) == 0:

                if return_boolean:
                    return True

                segments.append((0, sa_cnt))
                cnt += tot_cnt
                continue

            def prefetch(lo, hi, depth=0):
                mi = (lo + hi) // 2 # sa index to inspect
                if mi == -1: # this may happen when lo=-1 and hi=0, and we skip prefetching
                    return
                if depth == 1: # fetch ds
                    ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                    ds.madvise(mmap.MADV_WILLNEED, ptr - ptr % PAGESIZE, len(prompt_buf) + ptr % PAGESIZE)
                elif depth == 3: # fetch sa
                    sa.madvise(mmap.MADV_WILLNEED, mi*ptr_size - mi*ptr_size % PAGESIZE, ptr_size + mi*ptr_size % PAGESIZE)
                    return
                prefetch(lo, mi, depth+1)
                prefetch(mi, hi, depth+1)

            # Search for the leftmost sa index that IS >= the prompt
            if hint_segment is None:
                lo, hi = -1, sa_cnt # lo is always < the prompt, hi is always >= the prompt
            else:
                lo, hi = hint_segment[0] - 1, hint_segment[1]
            while hi - lo > 1:
                prefetch(lo, hi)
                mi = (lo + hi) // 2 # sa index to inspect
                ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                if ds[ptr : ptr + len(prompt_buf)] < prompt_buf:
                    lo = mi
                else:
                    hi = mi
            start = hi

            # Search for the leftmost sa index that IS > the prompt
            if hint_segment is None:
                lo, hi = -1, sa_cnt # lo is always <= the prompt, hi is always > the prompt
            else:
                lo, hi = hint_segment[0] - 1, hint_segment[1]
            while hi - lo > 1:
                prefetch(lo, hi)
                mi = (lo + hi) // 2 # sa index to inspect
                ptr = self.convert_rank_to_ptr(sa, mi, ptr_size)
                if ds[ptr : ptr + len(prompt_buf)] <= prompt_buf:
                    lo = mi
                else:
                    hi = mi
            end = hi

            segments.append((start, end))
            cnt += end - start

            if return_boolean and cnt > cnt_threshold:
                return True

        if return_boolean:
            assert cnt == 0
            return False

        assert cnt >= 0, cnt
        return dict(segments=segments, cnt=cnt)

    def convert_rank_to_ptr(self, sa, rank, ptr_size):
        ptr_buf = np.frombuffer(sa[rank*ptr_size:(rank+1)*ptr_size], dtype=np.uint8)
        # Add right padding due to little-endianness
        ptr_buf = np.pad(ptr_buf, (0, 8 - ptr_size), 'constant', constant_values=0)
        ptr = int(ptr_buf.view(np.uint64)[0])
        return ptr

    def convert_offset_to_token_id(self, ds, offset):
        if offset >= len(ds):
            # This happens when we matched the very end of the ds.
            return self.tokenizer.eos_token_id
        token_buf = np.frombuffer(ds[offset:offset+2], dtype=np.uint8)
        token_id = int(token_buf.view(np.uint16)[0])
        # If you see \xff\xff, this actually means we're at the very end of a document.
        if token_id == 65535:
            token_id = self.tokenizer.eos_token_id
        return token_id

def main():
    lm = NGramLanguageModeling(cache_dir='./cache', force_cache=False, tokenizer_type='gpt2')
    lm.load_cache(cache_prefix='test')
    tokenizer = lm.tokenizer

    # Test find()
    prompt = 'Today is a beautiful day'
    # prompt = 'Today is your lucky day'
    # prompt = 'dskhcs' # nonexist
    # random string with letters
    import string
    prompt = ''.join(random.choice(string.ascii_lowercase) for i in range(20))
    prompt_ids = tokenizer.encode(prompt)
    # prompt_ids = [8053, 286, 23167, 9587, 430, 315, 12579, 22861, 198]
    print(f'prompt_ids: {prompt_ids}')
    start_time = time.time()
    find_result = lm.find(prompt_ids)
    end_time = time.time()
    print(f'find took {end_time - start_time} seconds')
    print(f'find_result: {find_result}')
    for s, segment in enumerate(find_result['segments']):
        print(f'Datastore shard #{s}')
        (start, end) = segment
        result = lm.get_ds_tokens(s, start - 1, len(prompt_ids) + 1)
        print(f'\tds at (start-1): token_ids = {result["token_ids"]}, token_buf = {result["token_buf"]}')
        result = lm.get_ds_tokens(s, start, len(prompt_ids) + 1)
        print(f'\tds at (start): token_ids = {result["token_ids"]}, token_buf = {result["token_buf"]}')
        result = lm.get_ds_tokens(s, end - 1, len(prompt_ids) + 1)
        print(f'\tds at (end-1): token_ids = {result["token_ids"]}, token_buf = {result["token_buf"]}')
        result = lm.get_ds_tokens(s, end, len(prompt_ids) + 1)
        print(f'\tds at (end): token_ids = {result["token_ids"]}, token_buf = {result["token_buf"]}')
    print()

    # Test get_next_token_distribution()
    prompt = 'It was a beautiful day'
    prompt_ids = tokenizer.encode(prompt)
    print(f'prompt: {prompt}')
    print(f'prompt_ids: {prompt_ids}')
    find_result = lm.find(prompt_ids)
    print(f'prompt_cnt = {find_result["cnt"]}')
    start_time = time.time()
    next_token_distribution_fast = lm.get_next_token_distribution_fast(prompt_ids, eps=1e-8)
    end_time = time.time()
    print(f'Number of distinct token types: {len(next_token_distribution_fast["freq_by_token_id"])}')
    print(f'get_next_token_distribution_fast took {end_time - start_time} seconds')
    start_time = time.time()
    next_token_distribution = lm.get_next_token_distribution(prompt_ids, eps=1e-8)
    end_time = time.time()
    print(f'get_next_token_distribution took {end_time - start_time} seconds')
    if next_token_distribution == next_token_distribution_fast:
        print('Results are the same!')
    else:
        print('Results are different!')
    print()

    # Test get_next_token_distribution_onehot()
    # prompt = 'It was a'
    # prompt = 'widhciuwefclewifbewibfdew'
    prompt = ' of the Kirghiz SS'
    prompt_ids = tokenizer.encode(prompt)
    print(f'prompt: {prompt}')
    print(f'prompt_ids: {prompt_ids}')
    next_token_distribution_onehot = lm.get_next_token_distribution_onehot(prompt_ids)
    print(next_token_distribution_onehot)
    print()

    # Test get_original_documents()
    prompt = ' of the Kirghiz SS'
    prompt_ids = tokenizer.encode(prompt)
    print(f'prompt: {prompt}')
    print(f'prompt_ids: {prompt_ids}')
    documents = lm.get_original_documents(prompt_ids)
    document = documents[0]
    string = document['string']
    char_offset = document['char_offset']
    print(f'string: {string[:50]} ... {string[char_offset-50:char_offset+50]} ... {string[-50:]}')
    print()

    # Test get_all_documents()
    output = lm.get_all_documents(start_idx=0, end_idx=5)
    documents = output['documents']
    # for document in documents:
    #     print(lm.tokenizer.decode(document))
    # print(documents)
    print()

    # hint = None
    # start_idx = 0
    # while True:
    #     end_idx = start_idx + 1000
    #     output = lm.get_all_documents(start_idx=start_idx, end_idx=end_idx, hint=hint)
    #     documents = output['documents']
    #     print(len(documents))
    #     hint = output['hint']
    #     start_idx = end_idx
    #     if output['done']:
    #         break
    # print(start_idx)
    # print()

    # Test exclude_eos
    prompt = ' of a legal, medical, or any other professional.'
    prompt_ids = tokenizer.encode(prompt)
    print(f'prompt: {prompt}')
    print(f'prompt_ids: {prompt_ids}')
    prob = lm.compute_prob(prompt_ids, proposed_ids=[1439], eps=1e-8)
    print(f'prob (exclude_eos=False): {prob}')
    prob = lm.compute_prob(prompt_ids, proposed_ids=[1439], eps=1e-8, exclude_eos=True)
    print(f'prob (exclude_eos=True): {prob}')
    next_token_distribution_fast = lm.get_next_token_distribution_fast(prompt_ids, eps=1e-8)
    del next_token_distribution_fast['prob_by_token_id']
    print(next_token_distribution_fast)
    next_token_distribution_fast = lm.get_next_token_distribution_fast(prompt_ids, eps=1e-8, exclude_eos=True)
    del next_token_distribution_fast['prob_by_token_id']
    print(next_token_distribution_fast)
    print()


class NGramLanguageModelingUnion(NGramLanguageModeling):

    def __init__(self, ngram_lm_1, ngram_lm_2):

        assert isinstance(ngram_lm_1, NGramLanguageModeling)
        assert isinstance(ngram_lm_2, NGramLanguageModeling)
        assert ngram_lm_1.tokenizer_type == ngram_lm_2.tokenizer_type

        self.ngram_lm_1 = ngram_lm_1
        self.ngram_lm_2 = ngram_lm_2
        self.tokenizer = ngram_lm_1.tokenizer
        self.vocab_size = ngram_lm_1.vocab_size

    def load_cache(self, cache_prefix):
        self.ngram_lm_1.load_cache(cache_prefix)
        self.ngram_lm_2.load_cache(cache_prefix)

    def save_cache(self):
        self.ngram_lm_1.save_cache()
        self.ngram_lm_2.save_cache()

    def find(self, prompt_ids, hint_segments=None, return_boolean=False, cnt_threshold=0):

        if hint_segments is not None:
            assert type(hint_segments)==dict, type(hint_segments)
            hint_segments1 = hint_segments["1"]
            hint_segments2 = hint_segments["2"]
        else:
            hint_segments1 = None
            hint_segments2 = None

        result1 = self.ngram_lm_1.find(prompt_ids, hint_segments1, return_boolean, cnt_threshold)
        result2 = self.ngram_lm_2.find(prompt_ids, hint_segments2, return_boolean, cnt_threshold)

        cnt = result1["cnt"] + result2["cnt"]
        segments = {"1": result1["segments"], "2": result2["segments"]}

        return {"cnt": cnt, "segments": segments}

    def get_next_token_distribution(self, prompt_ids, eps, exclude_eos=False):
        result1 = self.ngram_lm_1.get_next_token_distribution(prompt_ids, eps, exclude_eos)
        result2 = self.ngram_lm_2.get_next_token_distribution(prompt_ids, eps, exclude_eos)

        prompt_cnt = result1["prompt_cnt"] + result2["prompt_cnt"]

        freq_by_token_id = {}
        prob_by_token_id = {}
        keys = set(list(result1["freq_by_token_id"]) + list(result2["freq_by_token_id"])) - set(["others"])

        for key in keys:
            cnt = result1["freq_by_token_id"][key] + result2["freq_by_token_id"][key]
            assert cnt > 0
            freq_by_token_id[key] = cnt
            prob_by_token_id[key] = (cnt + eps) / (prompt_cnt + self.vocab_size * eps)

        freq_by_token_id["others"] = 0
        prob_by_token_id["others"] = eps / (prompt_cnt + self.vocab_size * eps)

        return dict(prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)

    def get_next_token_distribution_fast(self, prompt_ids, eps, exclude_eos=False):
        result1 = self.ngram_lm_1.get_next_token_distribution_fast(prompt_ids, eps, exclude_eos)
        result2 = self.ngram_lm_2.get_next_token_distribution_fast(prompt_ids, eps, exclude_eos)

        prompt_cnt = result1["prompt_cnt"] + result2["prompt_cnt"]

        freq_by_token_id = {}
        prob_by_token_id = {}
        keys = set(list(result1["freq_by_token_id"]) + list(result2["freq_by_token_id"])) - set(["others"])

        for key in keys:
            cnt = result1["freq_by_token_id"][key] + result2["freq_by_token_id"][key]
            assert cnt > 0
            freq_by_token_id[key] = cnt
            prob_by_token_id[key] = (cnt + eps) / (prompt_cnt + self.vocab_size * eps)

        freq_by_token_id["others"] = 0
        prob_by_token_id["others"] = eps / (prompt_cnt + self.vocab_size * eps)

        return dict(prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)

    def get_next_token_distribution_onehot(self, prompt_ids, eps, exclude_eos=False):
        result1 = self.ngram_lm_1.get_next_token_distribution_onehot(prompt_ids, eps, exclude_eos)
        result2 = self.ngram_lm_2.get_next_token_distribution_onehot(prompt_ids, eps, exclude_eos)

        assert result1["result"] in ["zero", "one", "multi"]
        assert result2["result"] in ["zero", "one", "multi"]

        if result1["result"]=="multi" or result2["result"]=="multi":
            return dict(result="multi")

        elif result1["result"]==result2["result"]=="zero":
            return dict(result="zero")

        elif result1["result"]=="zero" and result2["result"]=="one":
            return result2

        elif result1["result"]=="one" and result2["result"]=="zero":
            return result1

        assert result1["result"]==result2["result"]=="one"

        prompt_cnt = result1["prompt_cnt"] + result2["prompt_cnt"]

        freq_by_token_id = {}
        prob_by_token_id = {}
        keys = set(list(result1["freq_by_token_id"]) + list(result2["freq_by_token_id"])) - set(["others"])

        assert len(keys) in [1, 2], keys

        if len(keys)==2:
            return dict(result="multi")

        for key in keys:
            cnt = result1["freq_by_token_id"][key] + result2["freq_by_token_id"][key]
            assert cnt > 0
            freq_by_token_id[key] = cnt
            prob_by_token_id[key] = (cnt + eps) / (prompt_cnt + self.vocab_size * eps)

        freq_by_token_id["others"] = 0
        prob_by_token_id["others"] = eps / (prompt_cnt + self.vocab_size * eps)

        return dict(result="one", prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)


class NGramLanguageModelingDifference(NGramLanguageModeling):

    def __init__(self, ngram_lm_1, ngram_lm_2):

        assert isinstance(ngram_lm_1, NGramLanguageModeling)
        assert isinstance(ngram_lm_2, NGramLanguageModeling)
        assert ngram_lm_1.tokenizer_type == ngram_lm_2.tokenizer_type

        self.ngram_lm_1 = ngram_lm_1
        self.ngram_lm_2 = ngram_lm_2
        self.tokenizer = ngram_lm_1.tokenizer
        self.vocab_size = ngram_lm_1.vocab_size

    def load_cache(self, cache_prefix):
        self.ngram_lm_1.load_cache(cache_prefix)
        self.ngram_lm_2.load_cache(cache_prefix)

    def save_cache(self):
        self.ngram_lm_1.save_cache()
        self.ngram_lm_2.save_cache()

    def find(self, prompt_ids, hint_segments=None, return_boolean=False, cnt_threshold=0):

        if hint_segments is not None:
            assert type(hint_segments)==dict
            hint_segments1 = hint_segments["1"]
            hint_segments2 = hint_segments["2"]
        else:
            hint_segments1 = None
            hint_segments2 = None

        result1 = self.ngram_lm_1.find(prompt_ids, hint_segments1, return_boolean, cnt_threshold)
        result2 = self.ngram_lm_2.find(prompt_ids, hint_segments2, return_boolean, cnt_threshold)

        cnt = result1["cnt"] - result2["cnt"]
        assert cnt >= 0
        segments = {"1": result1["segments"], "2": result2["segments"]}

        return {"cnt": cnt, "segments": segments}

    def get_next_token_distribution(self, prompt_ids, eps, exclude_eos=False):
        result1 = self.ngram_lm_1.get_next_token_distribution(prompt_ids, eps, exclude_eos)
        result2 = self.ngram_lm_2.get_next_token_distribution(prompt_ids, eps, exclude_eos)

        prompt_cnt = result1["prompt_cnt"] - result2["prompt_cnt"]
        assert prompt_cnt >= 0

        freq_by_token_id = {}
        prob_by_token_id = {}
        keys = set(list(result1["freq_by_token_id"]) + list(result2["freq_by_token_id"])) - set(["others"])

        for key in keys:
            cnt = result1["freq_by_token_id"][key] - result2["freq_by_token_id"][key]
            assert cnt >= 0
            freq_by_token_id[key] = cnt
            prob_by_token_id[key] = (cnt + eps) / (prompt_cnt + self.vocab_size * eps)

        freq_by_token_id["others"] = 0
        prob_by_token_id["others"] = eps / (prompt_cnt + self.vocab_size * eps)

        return dict(prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)

    def get_next_token_distribution_fast(self, prompt_ids, eps, exclude_eos=False):
        result1 = self.ngram_lm_1.get_next_token_distribution_fast(prompt_ids, eps, exclude_eos)
        result2 = self.ngram_lm_2.get_next_token_distribution_fast(prompt_ids, eps, exclude_eos)

        prompt_cnt = result1["prompt_cnt"] - result2["prompt_cnt"]
        assert prompt_cnt >= 0

        freq_by_token_id = {}
        prob_by_token_id = {}
        keys = set(list(result1["freq_by_token_id"]) + list(result2["freq_by_token_id"])) - set(["others"])

        for key in keys:
            cnt = result1["freq_by_token_id"][key] - result2["freq_by_token_id"][key]
            assert cnt >= 0
            assert cnt > 0
            freq_by_token_id[key] = cnt
            prob_by_token_id[key] = (cnt + eps) / (prompt_cnt + self.vocab_size * eps)

        freq_by_token_id["others"] = 0
        prob_by_token_id["others"] = eps / (prompt_cnt + self.vocab_size * eps)

        return dict(prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)

    def get_next_token_distribution_onehot(self, prompt_ids, eps, exclude_eos=False):
        result1 = self.ngram_lm_1.get_next_token_distribution_onehot(prompt_ids, eps, exclude_eos)
        result2 = self.ngram_lm_2.get_next_token_distribution_onehot(prompt_ids, eps, exclude_eos)

        assert result1["result"] in ["zero", "one", "multi"]
        assert result2["result"] in ["zero", "one", "multi"]

        if result1["result"]==result2["result"]=="zero":
            return dict(result="zero")

        elif result2["result"]=="zero":
            return result1

        elif result1["result"]=="zero":
            raise NotImplementedError()

        elif result1["result"]==result2["result"]=="one":
            prompt_cnt = result1["prompt_cnt"] + result2["prompt_cnt"]

            freq_by_token_id = {}
            prob_by_token_id = {}
            keys1 = set(list(result1["freq_by_token_id"])) - set(["others"])
            keys2 = set(list(result2["freq_by_token_id"])) - set(["others"])

            assert len(keys2-keys1)==0
            assert len(keys1)==1
            assert len(keys2)==1

            prompt_cnt = result1["prompt_cnt"] - result2["prompt_cnt"]
            assert prompt_cnt >= 0

            freq_by_token_id = {}
            prob_by_token_id = {}

            for k in keys1:
                cnt = result1["freq_by_token_id"][k]-result2["freq_by_token_id"][k]
                assert cnt > 0
                freq_by_token_id[k] = cnt
                prob_by_token_id[k] = (cnt + eps) / (prompt_cnt + self.vocab_size * eps)

            freq_by_token_id["others"] = 0
            prob_by_token_id["others"] = eps / (prompt_cnt + self.vocab_size * eps)

            return dict(result="one", prompt_cnt=prompt_cnt, freq_by_token_id=freq_by_token_id, prob_by_token_id=prob_by_token_id)

        """When any of result1 and result2 is `multi`,
        no trivial way to do this fast.
        Need to use `get_next_token_distribution_fast`"""
        result = self.get_next_token_distribution_fast(prompt_ids, eps, exclude_eos)

        assert result["prompt_cnt"] > 0, "currently this func is called only when prompt_ids is found"
        assert len(result["freq_by_token_id"]) >= 2

        if len(result["freq_by_token_id"])==2:
            result["result"] = "one"
        else:
            result["result"] = "multi"
        return result




if __name__ == '__main__':
    main()
