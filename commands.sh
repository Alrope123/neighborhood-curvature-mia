# srun -p gpu-rtx6k -A h2lab --time=24:00:00 -n 1 --gpus=1 --mem=32G --pty /bin/bash
# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only --max_length 2000 --n_samples 5000 --n_group 100 --n_document_per_group 30
# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/pythia-2.8b-v0 --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only --max_length 2000 --n_samples 5000 --n_group 100 --n_document_per_group 30
# python inspect_redpajama.py --dataset redpajama --data_dir /gscratch/h2lab/alrope/LLM-generations/cache/datasets/datasets--togethercomputer--RedPajama-Data-1T-Sample/snapshots/98f93b765c118b999b1af570c3e399c207d08da7
# python inspect_redpajama.py --dataset pile --data_dir /gscratch/h2lab/sewon/data/the-pile/val.jsonl.zst
# python inspect_redpajama.py --dataset pile --data_dir /gscratch/h2lab/sewon/data/the-pile/train-all/00.jsonl.zst

# target/release/bff \
#   --bloom-filter-file pile_filter.bff \
#   --bloom-filter-size 1000000000 \
#   --expected-ngram-count 390794801 \
#   --output-directory pile_output/ \
#   --filtering-threshold 0.0 \
#   --whole-document \
#   /gscratch/h2lab/sewon/data/the-pile/train-all/*.jsonl.zst

# target/release/bff \
#   --bloom-filter-file /gscratch/h2lab/micdun/bff/full_pile_train_filter_0.bff  \
#   --output-directory /gscratch/h2lab/alrope/data/bff/redpajama-arxiv+pile/filter_0 \
#   --no-update-bloom-filter \
#   --reader-mode 2 \
#   --writer-mode 2 \
#   --bloom-filter-size 1000000000 \
#   --expected-ngram-count 390794801 \
#   --annotate-attribute-only \
#   /gscratch/h2lab/alrope/data/redpajama/arxiv/arxiv_023827cd-7ee8-42e6-aa7b-661731f4c70f.jsonl

#   target/release/bff \
#   --bloom-filter-file /gscratch/h2lab/micdun/bff/full_pile_train_filter_1.bff  \
#   --output-directory /gscratch/h2lab/alrope/data/bff/redpajama-arxiv+pile/filter_1 \
#   --no-update-bloom-filter \
#   --reader-mode 2 \
#   --writer-mode 2 \
#   --bloom-filter-size 1000000000 \
#   --expected-ngram-count 390794801 \
#   --annotate-attribute-only \
#   /gscratch/h2lab/alrope/data/redpajama/arxiv/arxiv_023827cd-7ee8-42e6-aa7b-661731f4c70f.jsonl /gscratch/h2lab/alrope/data/redpajama/arxiv/arxiv_fd572627-cce7-4667-a684-fef096dfbeb7.jsonl

# python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_pythia-2.8b-v0-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/lr_ratio_threshold_results.json \
#   --member_info_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_pythia-2.8b-v0-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/wikipedia_member.json \
#   --nonmember_info_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_pythia-2.8b-v0-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/wikipedia_nonmember.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_pythia-2.8b-v0-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/
# git add results/unified_mia/EleutherAI_pythia-2.8b-v0-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/*.png
# git add results/unified_mia/EleutherAI_pythia-2.8b-v0-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/*/*.png
# git add results/unified_mia/EleutherAI_pythia-2.8b-v0-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-5000--ref_gpt2-xl--m2000--tok_false/*/group_output.json

# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only --max_length 2000 --n_samples 5000 --n_group 100 --n_document_per_group 30

# python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-5000--ref_gpt2-xl--m2000--tok_false/lr_ratio_threshold_results.json \
#   --member_info_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-5000--ref_gpt2-xl--m2000--tok_false/wikipedia_member.json \
#   --nonmember_info_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-5000--ref_gpt2-xl--m2000--tok_false/wikipedia_nonmember.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-5000--ref_gpt2-xl--m2000--tok_false/
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-5000--ref_gpt2-xl--m2000--tok_false/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-5000--ref_gpt2-xl--m2000--tok_false/*/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-5000--ref_gpt2-xl--m2000--tok_false/*/group_output.json

# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --ref_model gpt2-xl --baselines_only --max_length 2000 --n_samples 5000 --n_group 100 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl

# python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-2-100--ref_gpt2-xl--m2000--tok_false/lr_ratio_threshold_results.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-2-100--ref_gpt2-xl--m2000--tok_false/

# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only  --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/pythia-2.8b-v0 --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only  --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl
# python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --ref_model gpt2-xl --baselines_only --n_group 100 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl

# python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/\
#   --top_k 100\
#   --key lls

# python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/\
#   --top_k 30\
#   --key lls

# python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/\
#   --top_k 30\
#   --key lls


# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/*/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/*/group_output.json

# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/*/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/*/group_output.json

# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/*/*.png
# git add results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/*/group_output.json

# git add results/unified_mia/*/*/*.png
# git add results/unified_mia/*/*/*/*.png
# git add results/unified_mia/*/*/*/group_output.json

python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-125m --baselines_only --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-1.3B --baselines_only --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-125m --baselines_only --n_group 100 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-1.3B --baselines_only --n_group 100 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl
