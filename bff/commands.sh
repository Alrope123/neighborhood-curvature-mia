# for threshold in 0.5 0.3 0.1 0.05 0.03 0.01 0.005 0.003 0.001 ; do
#     python bff/determine_membership.py --threshold $threshold
# done

for threshold in 0.5 0.3 0.1 0.05 0.03 0.01 0.005 0.003 0.001 ; do
    python bff/determine_membership.py --threshold $threshold --data_dir /gscratch/h2lab/alrope/data/wikipedia/processed/ --overlap_dir /gscratch/h2lab/alrope/data/bff/wikipedia+pile --save_dir /gscratch/h2lab/alrope/neighborhood-curvature-mia/bff/wikipedia --data_type wikipedia
done