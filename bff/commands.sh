for threshold in 0.5 0.3 0.1 0.05 0.03 0.01 0.005 0.003 0.001 ; do
    python bff/determine_membership.py --threshold $threshold
done