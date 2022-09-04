#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="QA_eval"
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL


eval_set=$1
version=$2
device_name=$3
bert_dir=$4
sscont_num_patts=$5
sscont_num_toks_per_patt=$6
# These flags below can include ``debug'', `no_ref_cache`'' and ``no_triple_cache''
flag1=$7
flag2=$8
flag3=$9

pwd
python -u qaeval_chinese_scripts.py --eval_set "$eval_set" --version "$version" \
--fpath_base ../../QAEval/nc_final_samples_%s_%s.json \
--wh_fpath_base ../../QAEval/nc_wh_final_samples_%s_%s.json \
--negi_fpath_base ../../QAEval/nc_negi_final_samples_%s_%s.json \
--sliced_triples_dir ../../QAEval/nc_time_slices/ --slicing_method disjoint --time_interval 3 \
--sliced_triples_base_fn nc_typed_triples_%s_%s.json --eval_mode boolean \
--eval_method sscont --eg_root ../gfiles --eg_name global_graphs --eg_suff _gsim.txt --eg_feat_idx 1 \
--max_spansize 64 --result_dir ../gfiles/qaeval_en_results_prd/%s_%s/ --backupAvg --device_name "$device_name" \
--min_graphsize 20480 --max_context_size 1000 --bert_dir "$bert_dir" --mt5_dir t5-base --batch_size 10 \
--refs_cache_dir ./cache_dir_en/refs_%s.json --triples_cache_dir ./cache_dir_en/triples_%s.json --tfidf_path XXX \
--articleIds_dict_path XXX --num_refs_bert1 5 --sscont_num_patts "$sscont_num_patts" \
--sscont_num_toks_per_patt "$sscont_num_toks_per_patt" --lang en ${flag1} ${flag2} ${flag3}


