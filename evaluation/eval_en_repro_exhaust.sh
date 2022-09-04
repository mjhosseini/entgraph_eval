#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="EG_eval"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL



python -u eval.py --gpath ../../contextual_linkpred_entgraph/entgraphs_dir_en/entgraphs_contextual_bsz_32_triplesplit_enbert_newdata_reproduce_300 \
--dev --sim_suffix _sim.txt --method local_en_3_3_70000_newdata_reproduce_dev --CCG 1 \
--typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --write --exactType --backupAvg --no_lemma_baseline \
--no_constraints --outDir results/pr_rec_clpeg_en_3_3_70000_dev --eval_range orig_exhaust --avg_pooling

python -u eval.py --gpath ../../contextual_linkpred_entgraph/entgraphs_dir_en/entgraphs_contextual_bsz_32_triplesplit_enbert_newdata_reproduce_300 \
--test --sim_suffix _sim.txt --method local_en_3_3_70000_newdata_reproduce_test --CCG 1 \
--typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --write --exactType --backupAvg --no_lemma_baseline \
--no_constraints --outDir results/pr_rec_clpeg_en_3_3_70000_test --eval_range orig_exhaust --avg_pooling

echo "Finished"