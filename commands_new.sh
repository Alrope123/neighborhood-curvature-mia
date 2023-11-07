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

python MIA/process_results.py \
  --result_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-2.7B/lr_ratio_threshold_results.json \
  --result_path_ref /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/EleutherAI_gpt-neo-125m/lr_ratio_threshold_results.json\
  --membership_path /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/rpj-arxiv_noisy/group_to_member.pkl\
  --out_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-arxiv_noisy-926-74-100-m1024/lira-EleutherAI_gpt-neo-2.7B-EleutherAI_gpt-neo-125m/\
  --top_k 100\
  --key crit

/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-1-m1024/EleutherAI_gpt-neo-125m--min_k
/gscratch/h2lab/alrope/neighborhood-curvature-mia/results_new/rpj-book-968-32-100-m1024/EleutherAI_gpt-neo-125m/lr_ratio_threshold_results.json

git add results_new/*/*/*.png
git add results_new/*/*/within_set_similarity.json
git add results_new/*/*/*/*.png
git add results_new/*/*/*/group_output.json
