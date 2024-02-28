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

python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only  --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/pythia-2.8b-v0 --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only  --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --ref_model gpt2-xl  --baselines_only --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --ref_model gpt2-xl --baselines_only --n_group 100 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/\
  --top_k 100\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/\
  --top_k 30\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/\
  --top_k 30\
  --key lls


python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-125m --baselines_only --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-1.3B --baselines_only --n_group 100 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-125m --baselines_only --n_group 100 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --ref_model EleutherAI/gpt-neo-1.3B --baselines_only --n_group 100 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/\
  --top_k 100\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/\
  --top_k 30\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/\
  --top_k 30\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-1.3B--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-1.3B--tok_false/\
  --top_k 30\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-125m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-125m--tok_false/\
  --top_k 30\
  --key lls


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-100-100--ref_gpt2-xl--tok_false/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-100-30--ref_gpt2-xl--tok_false/\
  --top_k 30\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_gpt2-xl--tok_false/\
  --top_k 30\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-1.3B--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-1.3B--tok_false/\
  --top_k 30\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-125m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-100-30--ref_EleutherAI_gpt-neo-125m--tok_false/\
  --top_k 30\
  --key crit


python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_month --dataset_member_key text --dataset_nonmember rpj-arxiv_month --dataset_nonmember_key text --ref_model gpt2-xl --n_group -1 --n_document_per_group 100 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_month/group_to_member.pkl
python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_month-rpj-arxiv_month--1-100--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_month/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_month-rpj-arxiv_month--1-100--ref_gpt2-xl--tok_false/\
  --top_k 100\
  --key lls
python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_month-rpj-arxiv_month--1-100--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_month/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_month-rpj-arxiv_month--1-100--ref_gpt2-xl--tok_false/\
  --top_k 100\
  --key crit


python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_month --dataset_member_key text --dataset_nonmember wikipedia_month --dataset_nonmember_key text --ref_model gpt2-xl --n_group -1 --n_document_per_group 500 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_month/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member rpj-arxiv_month --dataset_member_key text --dataset_nonmember rpj-arxiv_month --dataset_nonmember_key text --ref_model gpt2-xl --n_group -1 --n_document_per_group 500 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_month/group_to_member.pkl

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_bigscience_bloom-560m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_bigscience_bloom-560m--tok_false/\
  --top_k 100\
  --key lls


python analysis/within_set_similarity.py \
  --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_bigscience_bloom-560m--tok_false/ \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl

python analysis/within_set_similarity.py \
  --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-926-74-100--ref_gpt2-xl--tok_false/ \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl

python analysis/within_set_similarity.py \
  --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_gpt2-xl--tok_false/ \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl



python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-1.3B--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-1.3B--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--m511--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--m511--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-125m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-125m--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-160m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-160m--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-410m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-410m--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-1b--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-1b--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_facebook_opt-350m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_facebook_opt-350m--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_facebook_opt-1.3b--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_facebook_opt-1.3b--tok_false/\
  --top_k 30\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_facebook_opt-125m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_facebook_opt-125m--tok_false/\
  --top_k 30\
  --key crit ref_lls


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-125m--m511--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-125m--m511--tok_false/\
  --top_k 30\
  --key lls crit ref_lls

python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --ref_model EleutherAI/pythia-1b --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl 

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_gpt2-xl--tok_false/\
  --top_k 30\
  --key lls crit ref_lls bff

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-926-74-100--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-926-74-100--ref_gpt2-xl--tok_false/\
  --top_k 100\
  --key lls crit ref_lls bff

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_bigscience_bloom-560m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_bigscience_bloom-560m--tok_false/\
  --top_k 100\
  --key lls crit ref_lls bff

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-book-rpj-book-968-32-1--ref_EleutherAI_gpt-neo-125m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-book-rpj-book-968-32-1--ref_EleutherAI_gpt-neo-125m--tok_false/\
  --top_k 100\
  --key crit ref_lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-926-74-100--ref_EleutherAI_gpt-neo-125m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-rpj-arxiv_noisy-rpj-arxiv_noisy-926-74-100--ref_EleutherAI_gpt-neo-125m--tok_false/\
  --top_k 100\
  --key crit ref_lls

# Language
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name bigscience/bloom-3b --cache_dir cache --dataset_member language --dataset_member_key text --dataset_nonmember language --dataset_nonmember_key text --ref_model bigscience/bloom-560m --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/redpajama/wikipedia/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name bigscience/bloom-3b --cache_dir cache --dataset_member language --dataset_member_key text --dataset_nonmember language --dataset_nonmember_key text --ref_model gpt2-xl --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/redpajama/wikipedia/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl

# # Debug
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --ref_model EleutherAI/pythia-1b --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl 

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_bigscience_bloom-560m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_bigscience_bloom-560m--tok_false/\
  --top_k 1000\
  --key lls crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_gpt2-xl--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/bigscience_bloom-3b-main-t5-large-temp/fp32-0.3-1-language-language--1--1-1000--ref_gpt2-xl--tok_false/\
  --top_k 1000\
  --key lls crit ref_lls

python MIA/run_mia_unified.py --output_name unified_mia --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --ref_model EleutherAI/pythia-160m --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/beaker_results/unified_mia/EleutherAI_pythia-6.9b-deduped-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-125m--tok_false/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/None-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-770-230-30--ref_EleutherAI_pythia-160m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_pythia-6.9b-deduped-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-160m--tok_false/lr_ratio_threshold_results.json\
  --top_k 30\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/beaker_results/unified_mia/EleutherAI_pythia-6.9b-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_gpt-neo-125m--tok_false/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/None-main-t5-large-temp/fp32-0.3-1-wikipedia-wikipedia-770-230-30--ref_EleutherAI_pythia-160m--tok_false/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_pythia-6.9b-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_EleutherAI_pythia-160m--tok_false/lr_ratio_threshold_results.json\
  --top_k 30\
  --key crit

python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --ref_model EleutherAI/pythia-1b --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl 
python MIA/run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --ref_model EleutherAI/pythia-1b --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl 


git add results/unified_mia/*/*/*.png
git add results/unified_mia/*/*/*/*.png
git add results/unified_mia/*/*/*/group_output.json

git add results/unified_mia/*/*/within_set_similarity.json


# Wikipedia_Noisy
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-2.7B --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-1.3B --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-125m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-410m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-1b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name facebook/opt-1.3b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name facebook/opt-350m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name gpt2-xl --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new

# Arxiv
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-2.7B --max_length 1024 --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --baselines_only --n_group_member 926 --n_group_nonmember 74 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/arxiv/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-125m --max_length 1024 --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --baselines_only --n_group_member 926 --n_group_nonmember 74 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/arxiv/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name gpt2-xl --max_length 1024 --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --baselines_only --n_group_member 926 --n_group_nonmember 74 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/arxiv/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --save_dir results_new

# Book
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-2.7B --max_length 1024 --cache_dir cache --dataset_member rpj-book --dataset_member_key text --dataset_nonmember rpj-book --dataset_nonmember_key text --strategy split --baselines_only --n_group_member 968 --n_group_nonmember 32 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/book/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-125m --max_length 1024 --cache_dir cache --dataset_member rpj-book --dataset_member_key text --dataset_nonmember rpj-book --dataset_nonmember_key text --strategy split --baselines_only --n_group_member 968 --n_group_nonmember 32 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/book/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name gpt2-xl --max_length 1024 --cache_dir cache --dataset_member rpj-book --dataset_member_key text --dataset_nonmember rpj-book --dataset_nonmember_key text --strategy split --baselines_only --n_group_member 968 --n_group_nonmember 32 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/book/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl --save_dir results_new

# Language
python MIA/run_mia_unified.py --base_model_name bigscience/bloom-3b --max_length 1024 --cache_dir cache --dataset_member language --dataset_member_key text --dataset_nonmember language --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/redpajama/wikipedia/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name bigscience/bloom-560m --max_length 1024 --cache_dir cache --dataset_member language --dataset_member_key text --dataset_nonmember language --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/redpajama/wikipedia/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl --save_dir results_new
python MIA/run_mia_unified.py --base_model_name gpt2-xl --max_length 1024 --cache_dir cache --dataset_member language --dataset_member_key text --dataset_nonmember language --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/redpajama/wikipedia/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl --save_dir results_new


python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-2.7B --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-2.7B --max_length 1024 --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --baselines_only --n_group_member 926 --n_group_nonmember 74 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/arxiv/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-2.7B --max_length 1024 --cache_dir cache --dataset_member rpj-book --dataset_member_key text --dataset_nonmember rpj-book --dataset_nonmember_key text --strategy split --baselines_only --n_group_member 968 --n_group_nonmember 32 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/book/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name bigscience/bloom-3b --max_length 1024 --cache_dir cache --dataset_member language --dataset_member_key text --dataset_nonmember language --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/redpajama/wikipedia/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl --save_dir results_new --min_k_prob

python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-1b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-410m-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_gpt-neo-2.7B--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_gpt-neo-2.7B--min_k/\
  --top_k 100\
  --key min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B--min_k/\
  --top_k 100\
  --key min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-1b/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-1b/\
  --top_k 100\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-410m/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-410m/\
  --top_k 100\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m/\
  --top_k 100\
  --key lls



python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-2.8b-deduped--min_k/\
  --top_k 100\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-1b-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-1b-deduped--min_k/\
  --top_k 100\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-410m-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-410m-deduped--min_k/\
  --top_k 100\
  --key lls

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m-deduped--min_k/\
  --top_k 100\
  --key lls


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-1b/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m/lr_ratio_threshold_results.json\
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-1b-EleutherAI_pythia-160m/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-1b/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-410m/lr_ratio_threshold_results.json\
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-1b-EleutherAI_pythia-410m/\
  --top_k 100\
  --key crit


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-410m/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m/lr_ratio_threshold_results.json\
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-410m-EleutherAI_pythia-160m/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m/lr_ratio_threshold_results.json\
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-2.8b-deduped--min_k-EleutherAI_pythia-160m/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-1b-deduped--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-160m/lr_ratio_threshold_results.json\
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-1b-deduped--min_k-EleutherAI_pythia-160m/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B/\
  --top_k 100\
  --key lls

for size1 in 160m 410m 1b 2.8b 6.9b 12b--min_k 
do  
  for size2 in 160m 410m 1b 2.8b 6.9b 12b--min_k 
  do
    python MIA/process_results.py \
    --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size1}/lr_ratio_threshold_results.json \
    --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size2}/lr_ratio_threshold_results.json\
    --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
    --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-${size1}-EleutherAI_pythia-${size2}/\
    --top_k 100\
    --key crit
  done
done
/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-1-m1024/EleutherAI_gpt-neo-125m--min_k
/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-100-m1024/EleutherAI_gpt-neo-125m/lr_ratio_threshold_results.json


HF_DATASETS_CACHE=/gscratch/h2lab/alrope/neighborhood-curvature-mia/cache/ HF_HOME=/gscratch/h2lab/alrope/neighborhood-curvature-mia/cache/ TRANSFORMERS_CACHE=/gscratch/h2lab/alrope/neighborhood-curvature-mia/cache/ python src/run.py --target_model EleutherAI/pythia-2.8b --ref_model EleutherAI/pythia-70m --data swj0419/WikiMIA --length 128

python analysis/within_set_similarity.py --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_gpt-neo-2.7B/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --downsize_factor 0.1 --method fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13
python analysis/within_set_similarity.py --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --downsize_factor 0.1 --method fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13
python analysis/within_set_similarity.py --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/language--1--1-1000-m1024/bigscience_bloom-3b--min_k/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl --downsize_factor 0.1 --method fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-125m --max_length 1024 --cache_dir cache --dataset_member rpj-book --dataset_member_key text --dataset_nonmember rpj-book --dataset_nonmember_key text --strategy split --baselines_only --n_group_member 968 --n_group_nonmember 32 --n_document_per_group 1 --data_dir /gscratch/h2lab/alrope/data/redpajama/book/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl --save_dir results_new --min_k_prob
python analysis/within_set_similarity.py --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-1-m1024/EleutherAI_gpt-neo-125m--min_k/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl --downsize_factor 0.1 --method fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13
python analysis/within_set_similarity.py --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/language--1--1-1000-m1024/bigscience_bloom-3b--min_k/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl --downsize_factor 1.0 --method fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13
python MIA/run_mia_unified.py --base_model_name EleutherAI/gpt-neo-2.7B --max_length 1024 --cache_dir cache --dataset_member rpj-book --dataset_member_key text --dataset_nonmember rpj-book --dataset_nonmember_key text --strategy split --baselines_only --n_group_member 968 --n_group_nonmember 32 --n_document_per_group 1 --data_dir /gscratch/h2lab/alrope/data/redpajama/book/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl --save_dir results_new --min_k_prob


python analysis/within_set_similarity.py --result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --downsize_factor 0.005 --method fasttext

python analysis/within_set_similarity_inner.py \
--name EleutherAI/gpt-neo-2.7B \
--data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ \
--result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_gpt-neo-2.7B/ \
--membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl \
--n_group_member 770 \
--n_group_nonmember 230 \
--downsize_factor 0.1 \
--methods fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13

python analysis/within_set_similarity_inner.py \
--name EleutherAI/gpt-neo-2.7B \
--data_dir /gscratch/h2lab/alrope/data/redpajama/arxiv/ \
--result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B/ \
--membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl \
--n_group_member 926 \
--n_group_nonmember 74 \
--downsize_factor 0.1 \
--methods fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13

python analysis/within_set_similarity_inner.py \
--name EleutherAI/gpt-neo-2.7B \
--data_dir /gscratch/h2lab/alrope/data/redpajama/book/ \
--result_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-1-m1024/EleutherAI_gpt-neo-125m--min_k/ \
--membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl \
--n_group_member 968 \
--n_group_nonmember 32 \
--downsize_factor 1.0 \
--methods fasttext tf-idf n-gram-2 n-gram-3 n-gram-7 n-gram-13 \
--cross_document 


for size1 in 160m-deduped--min_k 410m-deduped--min_k 1b-deduped--min_k 2.8b-deduped--min_k 6.9b-deduped--min_k 12b-deduped--min_k 
do  
  for size2 in 160m-deduped--min_k 410m-deduped--min_k 1b-deduped--min_k 2.8b-deduped--min_k 6.9b-deduped--min_k 12b-deduped--min_k
  do
    python MIA/process_results.py \
    --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size1}/lr_ratio_threshold_results.json \
    --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size2}/lr_ratio_threshold_results.json\
    --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
    --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-${size1}-EleutherAI_pythia-${size2}/\
    --top_k 100\
    --key crit
  done
done

for size1 in 70m-deduped--min_k 160m-deduped--min_k 410m-deduped--min_k 1b-deduped--min_k 2.8b-deduped--min_k 6.9b-deduped--min_k 12b-deduped--min_k
do  
  for size2 in 6.9b-deduped--min_k
  do
    python MIA/process_results.py \
    --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size1}/lr_ratio_threshold_results.json \
    --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size2}/lr_ratio_threshold_results.json\
    --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
    --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-${size1}-EleutherAI_pythia-${size2}/\
    --top_k 100\
    --key crit
  done
done

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-100-m1024/EleutherAI_gpt-neo-2.7B--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-book/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-100-m1024/EleutherAI_gpt-neo-2.7B--min_k/\
  --top_k 100\
  --key lls


python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-70m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-14m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-70m-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-14m-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob

mv beaker_results/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-6.9b/* results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-6.9b


HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-70m --max_length 1024 --cache_dir cache --dataset_member instruction_v1 --dataset_member_key text --dataset_nonmember instruction_v1 --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/instruction/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_v1/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name allenai/tulu-v1-llama2-7b --max_length 1024 --cache_dir cache --dataset_member instruction --dataset_member_key text --dataset_nonmember instruction --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/instruction/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction/group_to_member.pkl --save_dir results_new --min_k_prob

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/allenai_tulu-v1-llama2-7b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_v1/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/allenai_tulu-v1-llama2-7b--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/allenai_tulu-v1-llama2-13b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_v1/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/allenai_tulu-v1-llama2-13b--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/allenai_open-instruct-human-mix-7b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_human/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/allenai_open-instruct-human-mix-7b--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/allenai_open-instruct-human-mix-13b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_human/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/allenai_open-instruct-human-mix-13b--min_k/\
  --top_k 100\
  --key lls min_k zlib


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/allenai_tulu-v1-llama2-7b--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/meta-llama_Llama-2-7b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_v1/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/lira-allenai_tulu-v1-llama2-7b--min_k-meta-llama_Llama-2-7b-hf--min_k/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/allenai_tulu-v1-llama2-13b--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/meta-llama_Llama-2-13b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_v1/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/lira-allenai_tulu-v1-llama2-13b--min_k-meta-llama_Llama-2-13b-hf--min_k/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/allenai_open-instruct-human-mix-7b--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/meta-llama_Llama-2-7b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_human/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/lira-allenai_open-instruct-human-mix-7b--min_k-meta-llama_Llama-2-7b-hf--min_k/\
  --top_k 100\
  --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/allenai_open-instruct-human-mix-13b--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/meta-llama_Llama-2-13b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_human/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/lira-allenai_open-instruct-human-mix-13b--min_k-meta-llama_Llama-2-13b-hf--min_k/\
  --top_k 100\
  --key crit


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/meta-llama_Llama-2-7b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_v1/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/meta-llama_Llama-2-7b-hf--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/meta-llama_Llama-2-13b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_v1/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_v1--1--1-1000-m1024/meta-llama_Llama-2-13b-hf--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/meta-llama_Llama-2-7b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_human/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/meta-llama_Llama-2-7b-hf--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/meta-llama_Llama-2-13b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/instruction_human/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/instruction_human--1--1-1000-m1024/meta-llama_Llama-2-13b-hf--min_k/\
  --top_k 100\
  --key lls min_k zlib


python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 128 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 2048 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 512 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 256 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 16 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 32 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 64 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob


python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 128 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 2048 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 512 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 256 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 16 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 32 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-160m-deduped --max_length 64 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob

HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name kernelmachine/silo-pdswby-1.3b --max_length 1024 --cache_dir cache --dataset_member license_ccby --dataset_member_key text --dataset_nonmember license_ccby --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/openlicense/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/license_ccby/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name kernelmachine/silo-pdsw-1.3b --max_length 1024 --cache_dir cache --dataset_member license_sw --dataset_member_key text --dataset_nonmember license_sw --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/openlicense/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/license_sw/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name kernelmachine/silo-pd-1.3b --max_length 1024 --cache_dir cache --dataset_member license_pd --dataset_member_key text --dataset_nonmember license_pd --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 1000 --data_dir /gscratch/h2lab/alrope/data/openlicense/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/license_pd/group_to_member.pkl --save_dir results_new --min_k_prob


python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob --document_strategy diverse_100
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob --document_strategy diverse_500
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob --document_strategy diverse_1000 &> output0.txt
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob --document_strategy similar_100
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob --document_strategy similar_500
python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob --document_strategy similar_1000 &> output1.txt


for size in 1024 #16 32 64 128 256 512 2048 
do  
  # python MIA/process_results.py \
  # --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m${size}/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
  # --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  # --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m${size}/EleutherAI_pythia-2.8b-deduped--min_k/\
  # --top_k 100\
  # --key lls min_k zlib

  # python MIA/process_results.py \
  # --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m${size}/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
  # --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  # --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m${size}/EleutherAI_pythia-2.8b-deduped--min_k/\
  # --top_k 100\
  # --key lls min_k zlib

  python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m${size}/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m${size}/EleutherAI_pythia-160m-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m${size}/lira-EleutherAI_pythia-2.8b-deduped--min_k-EleutherAI_pythia-160m-deduped--min_k/\
  --top_k 100\
  --key crit
done

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/license_sw--1--1-1000-m1024/kernelmachine_silo-pdsw-1.3b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/license_sw/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/results_new/license_sw--1--1-1000-m1024/kernelmachine_silo-pdsw-1.3b--min_k//\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/license_pd--1--1-1000-m1024/kernelmachine_silo-pd-1.3b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/license_pd/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/results_new/license_pd--1--1-1000-m1024/kernelmachine_silo-pd-1.3b--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/license_ccby--1--1-1000-m1024/kernelmachine_silo-pdswby-1.3b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/license_ccby/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/results_new/license_ccby--1--1-1000-m1024/kernelmachine_silo-pdswby-1.3b--min_k/\
  --top_k 100\
  --key lls min_k zlib



python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name facebook/opt-1.3b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name gpt2-xl --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob

python MIA/run_mia_unified.py --base_model_name facebook/opt-350m --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name facebook/opt-2.7b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name meta-llama/Llama-2-7b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
python MIA/run_mia_unified.py --base_model_name stabilityai/stablelm-base-alpha-3b --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob

HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name stabilityai/stablelm-base-alpha-3b-v2 --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob


 EleutherAI/gpt-neo-125m --max_length 511 --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl
 facebook/opt-1.3b --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl
 facebook/opt-350m --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl
 facebook/opt-125m --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 30 --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-12b-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-12b-deduped--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-12b-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-12b-deduped--min_k/\
  --top_k 100\
  --key lls min_k zlib


python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-12b-deduped--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-160m-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/lira-EleutherAI_pythia-12b-deduped--min_k-EleutherAI_pythia-160m-deduped--min_k/\
  --top_k 100\
  --key crit



python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/language--1--1-1000-m1024/bigscience_bloom-7b1--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/language--1--1-1000-m1024/bigscience_bloom-7b1--min_k\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/language--1--1-1000-m1024/bigscience_bloom-7b1--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/language--1--1-1000-m1024/bigscience_bloom-560m--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/language/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/language--1--1-1000-m1024/lira-bigscience_bloom-7b1--min_k-bigscience_bloom-560m--min_k\
  --top_k 100\
  --key crit

for size in 12b 6.9b 2.8b 1b 410m 160m 70m
do  
  python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size}-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-${size}-deduped--min_k/\
  --top_k 100\
  --key lls min_k zlib
done

for model in gpt2-large--min_k gpt2-medium--min_k gpt2--min_k distilgpt2--min_k  # gpt2-xl facebook_opt-125m--min_k  facebook_opt-350m--min_k facebook_opt-1.3b--min_k facebook_opt-2.7b--min_k gpt2-xl--mink
do  
  # python MIA/run_mia_unified.py --base_model_name $model --max_length 1024 --cache_dir cache --dataset_member wikipedia_noisy --dataset_member_key text --dataset_nonmember wikipedia_noisy --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
  python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-12b-deduped--min_k/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/${model}/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/lira-EleutherAI_pythia-12b-deduped--min_k-${model}/\
  --top_k 100\
  --key crit
done


EleutherAI_gpt-neo-2.7B
EleutherAI_gpt-neo-1.3B
EleutherAI_gpt-neo-125m
EleutherAI_pythia-14m-deduped--min_k
EleutherAI_pythia-70m-deduped--min_k
EleutherAI_pythia-160m-deduped--min_k
EleutherAI_pythia-410m-deduped--min_k
EleutherAI_pythia-1b-deduped--min_k
EleutherAI_pythia-2.8b-deduped--min_k
EleutherAI_pythia-6.9b-deduped--min_k
facebook_opt-125m--min_k
facebook_opt-350m--min_k 
facebook_opt-1.3b--min_k 
facebook_opt-2.7b--min_k  

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024-similar_1000/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024-similar_1000/EleutherAI_pythia-2.8b-deduped--min_k/\
  --top_k 100\
  --key lls min_k zlib

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-6.9b--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_noisy-770-230-100-m1024/EleutherAI_pythia-6.9b--min_k/\
  --top_k 100\
  --key lls min_k zlib

for size in 2.8b 1b 410m 160m 70m
do  
  python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-${size}-deduped --max_length 1024 --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --baselines_only --n_group_member 926 --n_group_nonmember 74 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/arxiv/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
  # python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-${size} --max_length 1024 --cache_dir cache --dataset_member rpj-arxiv_noisy --dataset_member_key text --dataset_nonmember rpj-arxiv_noisy --dataset_nonmember_key text --baselines_only --n_group_member 926 --n_group_nonmember 74 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/redpajama/arxiv/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl --save_dir results_new --min_k_prob
done


# for size in 12b 6.9b 2.8b 1b 410m 160m 70m
# do
#   python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-${size}--min_k/lr_ratio_threshold_results.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-${size}--min_k/\
#   --top_k 100\
#   --key lls min_k zlib

#   python MIA/process_results.py \
#   --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-${size}-deduped--min_k/lr_ratio_threshold_results.json \
#   --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
#   --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-${size}-deduped--min_k/\
#   --top_k 100\
#   --key lls min_k zlib
# done

for size1 in 70m-deduped 160m-deduped 410m-deduped 1b-deduped 2.8b-deduped 6.9b-deduped 12b-deduped
do  
  for size2 in 70m-deduped 160m-deduped 410m-deduped 1b-deduped 2.8b-deduped 6.9b-deduped 12b-deduped
  do
    python MIA/process_results.py \
    --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-${size1}--min_k/lr_ratio_threshold_results.json \
    --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_pythia-${size2}--min_k/lr_ratio_threshold_results.json \
    --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
    --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/lira-EleutherAI_pythia-${size1}-EleutherAI_pythia-${size2}/\
    --top_k 100\
    --key crit
  done
done



mv /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/results_new/license_sw--1--1-1000-m1024/kernelmachine_silo-pdsw-1.3b--min_k/* /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/license_sw--1--1-1000-m1024/kernelmachine_silo-pdsw-1.3b--min_k/

HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member wikipedia_anchor --dataset_member_key text --dataset_nonmember wikipedia_anchor --dataset_nonmember_key text --baselines_only --n_group_member 980 --n_group_nonmember 20 --n_document_per_group 30 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed2/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_anchor/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member lyrics --dataset_member_key text --dataset_nonmember lyrics --dataset_nonmember_key text --baselines_only --n_group_member 500 --n_group_nonmember 500 --n_document_per_group 80 --data_dir /gscratch/h2lab/alrope/data/lyrics/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/lyrics/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member nytimes --dataset_member_key text --dataset_nonmember nytimes --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/nytimes/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/nytimes/group_to_member.pkl --save_dir results_new --min_k_prob

python MIA/process_results.py \
    --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_anchor-500-500-100-m1024/meta-llama_Llama-2-13b-hf--min_k/lr_ratio_threshold_results.json \
    --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_anchor-500-500-100-m1024/facebook_opt-1.3b--min_k/lr_ratio_threshold_results.json \
    --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_anchor/group_to_member.pkl\
    --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_anchor-500-500-100-m1024/lira-meta-llama_Llama-2-13b-hf--min_k-facebook_opt-1.3b--min_k/\
    --top_k 100\
    --key crit

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_anchor-500-500-100-m1024/meta-llama_Llama-2-13b-hf--min_k/lr_ratio_threshold_results.json \
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_anchor/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/wikipedia_anchor-500-500-100-m1024/meta-llama_Llama-2-13b-hf--min_k\
  --top_k 100\
  --key lls min_k zlib

python -m wikiextractor.WikiExtractor /gscratch/h2lab/alrope/data/wikipedia/enwiki-20240201-pages-articles-multistream.xml.bz2 --output /gscratch/h2lab/alrope/data/wikipedia/processed2 --bytes 100M --json --no-templates 

python wikipedia/label_members.py --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed2/ --pile_set_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out2/all_pile_set.pkl --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out2

python wikipedia/obtain_creation_date.py --set_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out3 --set_name pile_member_text.pkl --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/wikipedia/out3 --downsample_factor 0.1


for ratio in 0.6 ; do # 0.1 0.2 0.3 0.4 0.5 0.6 0.7
    HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/mixture/100-${ratio}-0.0-770-230/group_to_member.pkl --name "${ratio}-0.0" --save_dir results_controlled --min_k_prob
done

for ratio in 0.7 0.8 0.9 ; do # 0.1 0.2 0.3 0.4 0.5 0.6 0.7
    HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/mixture/100-0.0-${ratio}-770-230/group_to_member.pkl --name "0.0-${ratio}" --save_dir results_controlled --min_k_prob
done

for ratio in 0.5 ; do # 0.0 0.1 0.2 0.3 0.4 0.5
    HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name EleutherAI/pythia-2.8b-deduped --max_length 1024 --cache_dir cache --dataset_member wikipedia --dataset_member_key text --dataset_nonmember wikipedia --dataset_nonmember_key text --baselines_only --n_group_member 770 --n_group_nonmember 230 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/mixture/100-${ratio}-${ratio}-770-230/group_to_member.pkl --name "${ratio}-${ratio}" --save_dir results_controlled --min_k_prob
done


python -m wikiextractor.WikiExtractor /gscratch/h2lab/alrope/data/wikipedia/enwiki-20231120-pages-articles-multistream.xml.bz2 --output /gscratch/h2lab/alrope/data/wikipedia/processed3 --bytes 100M --json --no-templates 

for ratio1 in 0.6 ; do
  for ratio2 in 0.0 ; do
    python MIA/process_results.py \
      --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_controlled/wikipedia-770-230-100-m1024--${ratio1}-${ratio2}/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
      --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/mixture/100-${ratio1}-${ratio2}-770-230/group_to_member.pkl\
      --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_controlled/wikipedia-770-230-100-m1024--${ratio1}-${ratio2}/EleutherAI_pythia-2.8b-deduped--min_k/\
      --top_k 100\
      --key lls min_k zlib
  done
done

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 ; do
  python MIA/process_results.py \
    --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_controlled/wikipedia-770-230-100-m1024--${ratio}-${ratio}/EleutherAI_pythia-2.8b-deduped--min_k/lr_ratio_threshold_results.json \
    --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/mixture/100-${ratio}-${ratio}-770-230/group_to_member.pkl\
    --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_controlled/wikipedia-770-230-100-m1024--${ratio}-${ratio}/EleutherAI_pythia-2.8b-deduped--min_k/\
    --top_k 100\
    --key lls min_k zlib
done

HF_HOME="cache" python case_study/sample_instruction_dataset.py 


HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member wikipedia_anchor --dataset_member_key text --dataset_nonmember wikipedia_anchor --dataset_nonmember_key text --baselines_only --n_group_member 950 --n_group_nonmember 50 --n_document_per_group 100 --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed2/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia_anchor/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member contamination --dataset_member_key text --dataset_nonmember contamination --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group -1 --data_dir /gscratch/h2lab/alrope/data/eval/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/contamination/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member contamination_full --dataset_member_key text --dataset_nonmember contamination_full --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group -1 --data_dir /gscratch/h2lab/alrope/data/eval_full/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/contamination_full/group_to_member.pkl --save_dir results_new --min_k_prob
HF_HOME="cache" python MIA/run_mia_unified.py --base_model_name facebook/opt-125m --max_length 1024 --cache_dir cache --dataset_member tuning --dataset_member_key text --dataset_nonmember tuning --dataset_nonmember_key text --baselines_only --n_group_member -1 --n_group_nonmember -1 --n_document_per_group -1 --data_dir /gscratch/h2lab/alrope/data/instruction_v2/ --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/tuning/group_to_member.pkl --save_dir results_new --min_k_prob

git add results_new/*/*/*.png
git add results_new/*/*/within_set_similarity.json
git add results_new/*/*/within_set_similarity_inner.json
git add results_new/*/*/*/*.png
git add results_new/*/*/*/group_output.json

git add results_controlled/*/*/*.png
git add results_controlled/*/*/within_set_similarity.json
git add results_controlled/*/*/within_set_similarity_inner.json
git add results_controlled/*/*/*/*.png
git add results_controlled/*/*/*/group_output.json