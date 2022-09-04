#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="EG_eval"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

prem_enc=$1
hypo_enc=$2
enc_dim=$3
sidenote=$4

echo "prem_enc: ""$prem_enc"
echo "hypo_enc: ""$hypo_enc"
echo "embedding dimension: ""$enc_dim"
echo "sidenote: ""$sidenote"

python -u eval_chinese.py --gpath ../../contextual_linkpred_entgraph/entgraphs_dir/entgraphs_contextual_bsz_32_nfill_100_2270000_2270000_"$prem_enc"_"$hypo_enc"_"$enc_dim"_"$sidenote" \
--dev --sim_suffix _sim.txt --method local_"$prem_enc"_"$hypo_enc"_"$enc_dim"_"$sidenote"_2_2_70000_dev --CCG 1 \
--typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --write --exactType --backupAvg --no_lemma_baseline \
--no_constraints --outDir results/pr_rec_clpeg_2_2_70000_dev --eval_range orig_exhaust --avg_pooling

python -u eval_chinese.py --gpath ../../contextual_linkpred_entgraph/entgraphs_dir/entgraphs_contextual_bsz_32_nfill_100_2270000_2270000_"$prem_enc"_"$hypo_enc"_"$enc_dim"_"$sidenote" \
--test --sim_suffix _sim.txt --method local_"$prem_enc"_"$hypo_enc"_"$enc_dim"_"$sidenote"_2_2_70000_test --CCG 1 \
--typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --write --exactType --backupAvg --no_lemma_baseline \
--no_constraints --outDir results/pr_rec_clpeg_2_2_70000_test --eval_range orig_exhaust --avg_pooling

echo "Finished"