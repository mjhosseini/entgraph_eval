#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="EG_eval"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

prem_enc=$1
hypo_enc=$2
enc_dim=$3

echo "prem_enc: ""$prem_enc"
echo "hypo_enc: ""$hypo_enc"
echo "embedding dimension: ""$enc_dim"

python -u eval_chinese.py --gpath ../../contextual_linkpred_entgraph/entgraphs_contextual_bsz_24_3370000_3370000_"$prem_enc"_"$hypo_enc"_"$enc_dim" --dev --sim_suffix _sim.txt --method global_scores_orig_dev_apooling_binc --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --write --exactType --backupAvg --no_lemma_baseline --no_constraints --outDir results/pr_rec_orig_dev_exhaust_clp3_3_70000_"$prem_enc"_"$hypo_enc"_"$enc_dim" --eval_range orig_exhaust --avg_pooling
python -u eval_chinese.py --gpath ../../contextual_linkpred_entgraph/entgraphs_contextual_bsz_24_3370000_3370000_"$prem_enc"_"$hypo_enc"_"$enc_dim" --test --sim_suffix _sim.txt --method global_scores_orig_test_apooling_binc --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --write --exactType --backupAvg --no_lemma_baseline --no_constraints --outDir results/pr_rec_orig_test_exhaust_clp3_3_70000_"$prem_enc"_"$hypo_enc"_"$enc_dim" --eval_range orig_exhaust --avg_pooling

echo "Finished"