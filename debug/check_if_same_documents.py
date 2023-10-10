import os
import argparse
import pickle as pkl
import json



if __name__ == '__main__':
    base_result_path = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/results/unified_mia/EleutherAI_gpt-neo-2.7B-main-t5-large-temp/fp32-0.3-1-wikipedia_noisy-wikipedia_noisy-770-230-30--ref_{}--tok_false/lr_ratio_threshold_results.json"
    models = ["EleutherAI_pythia-160m", "EleutherAI_gpt-neo-125m", "EleutherAI_pythia-410m", "gpt2-xl"]

    sets_members = []
    sets_nonmembers = []
    for model in models:
        result_path = base_result_path.format(model)
        with open(result_path, 'r') as f:
            result = json.load(f)
        sets_members.append(set([(filename, i) for filename, i in result['member_meta']]))
        sets_nonmembers.append(set([(filename, i) for filename, i in result['nonmember_meta']]))
    
    for set_members in sets_members:
        assert set_members == sets_members[0], [set_members, sets_members[0]]
    for set_nonmembers in sets_nonmembers:
        assert set_nonmembers == sets_nonmembers[0], [set_nonmembers, sets_nonmembers[0]]
        