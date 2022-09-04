import os
from pathlib import Path
import argparse
from qaeval_utils import DateManager
from qaeval_chinese_general_functions import mask_entities_for_entries, mask_negi_entities_for_entries
from qaeval_chinese_wh_functions import qa_eval_wh_main
from qaeval_chinese_boolean_functions import qa_eval_boolean_main


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval_set', type=str, default='dev')
	parser.add_argument('--version', type=str, default='15_30_triple_doc_disjoint_1400000_2_lexic_wordnet')
	parser.add_argument('--fpath_base', type=str,
						default='../../QAEval/clue_final_samples_%s_%s.json')
	parser.add_argument('--wh_fpath_base', type=str,
						default='../../QAEval/clue_wh_final_samples_%s_%s.json')
	parser.add_argument('--negi_fpath_base', type=str,
						default='../../QAEval/clue_negi_final_samples_%s_%s.json')
	parser.add_argument('--sliced_triples_dir', type=str,
						default='../../QAEval/clue_time_slices/')
	parser.add_argument('--slicing_method', type=str, default='disjoint')
	parser.add_argument('--time_interval', type=int, default=3)
	parser.add_argument('--sliced_triples_base_fn', type=str, default='clue_typed_triples_%s_%s.json')
	parser.add_argument('--eval_mode', type=str, default='boolean', help='[boolean / wh-questions]')
	parser.add_argument('--eval_method', type=str, required=True)
	parser.add_argument('--eg_root', type=str, default='../gfiles', help='root directory to entailment graphs.')
	parser.add_argument('--eg_name', type=str, default='typedEntGrDir_Chinese2_2_V3',
						help='name of the desired entailment graph')
	parser.add_argument('--eg_suff', type=str, default='_sim.txt',
						help='suffix corresponding to the EG files of interest.')
	parser.add_argument('--eg_feat_idx', type=int, required=True,
						help='feature index, local graph: {cos: 0, weeds: 1, etc.}, global graph: {init: 0, globalized: 1}')
	parser.add_argument('--max_spansize', type=int, default=300, help='maximum span size for Bert inputs.')
	# parser.add_argument('--store_skip_idxes', action='store_true')
	# parser.add_argument('--skip_idxes_fn', type=str, default='./skip_idxes_%s_%s.json')
	parser.add_argument('--result_dir', type=str, default='../gfiles/qaeval_results_prd/%s_%s/')
	parser.add_argument('--pr_rec_fn', type=str, default='%s_prt_vals.tsv')
	parser.add_argument('--boolean_predictions_fn', type=str, default='%s_predictions.txt')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--backupAvg', action='store_true')
	parser.add_argument('--ignore_0_for_Avg', action='store_true',
						help='whether or not to ignore the zero entailment scores for averages, or to take them in in the denominator.')
	parser.add_argument('--keep_same_rel_sents', action='store_true')
	parser.add_argument('--device_name', type=str, default='cpu')
	parser.add_argument('--min_graphsize', type=int, default=20480)
	parser.add_argument('--max_context_size', type=int, default=3200,
						help='the maximum number of context sentences/rels to look at when answering each query')
	parser.add_argument('--bert_dir', type=str, default='/home/s2063487/bert_checkpoints/bert_base_chinese')
	parser.add_argument('--mt5_dir', type=str, default='/home/s2063487/bert_checkpoints/mt5_small')
	parser.add_argument('--batch_size', type=int, default=64)

	parser.add_argument('--refs_cache_dir', type=str, default='./cache_dir/refs_%s.json')
	parser.add_argument('--triples_cache_dir', type=str, default='./cache_dir/triples_%s.json')

	# flags for `wh' evaluation setting only.
	parser.add_argument('--mask_only_objs', action='store_true')
	parser.add_argument('--ignore_ftscr', action='store_true')
	parser.add_argument('--ignore_uscr', action='store_true')
	parser.add_argument('--ftscr_backupOnly', action='store_true')
	parser.add_argument('--uscr_backupOnly', action='store_true')

	parser.add_argument('--rtscr_w', type=float, default=1.0,
						help='Weight of restricted t-scores.')
	parser.add_argument('--ftscr_w', type=float, default=1.0,
						help='Weight of scores by averaging all sub-graphs where typed-matches for the index arguments'
							 'are found.')
	parser.add_argument('--uscr_w', type=float, default=1.0,
						help='Weight of scores by averaging all sub-graphs where matches are found.')

	parser.add_argument('--assert_indexarg_type', action='store_true',
						help='flag controlling whether the type of the index argument in context triples should be '
							 'asserted to remain the same as in the query triple.')
	parser.add_argument('--wh_predictions_fn', type=str, default='%s_wh_%s_predictions.txt')
	parser.add_argument('--wh_results_fn', type=str, default='%s_wh_%s_evalresults.txt')
	parser.add_argument('--max_t5_seq_length', type=int, default=600,
						help='maximum sequence length for the B type BERT baselines: the maximum length of the concatenated '
							 'context sentences with the appended query.')
	parser.add_argument('--wh_label', type=str, default='positive', help='')

	parser.add_argument('--no_triple_cache', action='store_true')
	parser.add_argument('--no_ref_cache', action='store_true')
	parser.add_argument('--no_write_individual_preds', action='store_true')

	# flags below are for the TF-IDF ranker.
	parser.add_argument('--tfidf_path', type=str,
						default='/Users/teddy/PycharmProjects/DrQA/scripts/retriever/clue_doc_db-tfidf-ngram=2-hash=16777216-tokenizer=spacy-chinese.npz')
	parser.add_argument('--articleIds_dict_path', type=str,
						default='/Users/teddy/PycharmProjects/DrQA/scripts/retriever/articleIds_by_partition.json')
	parser.add_argument('--num_refs_bert1', type=int, default=3,
						help='the number of reference articles to retrieve with TF-IDF for each query sentence.')

	# flag below are for the sscont method.
	parser.add_argument('--sscont_num_patts', type=int, default=5, help='the number of continuous patterns.')
	parser.add_argument('--sscont_num_toks_per_patt', type=int, default=2, help='the number of tokens for each pattern.')

	# flags below are put here for the graph initializer, but generally they should not be changed.
	parser.add_argument('--saveMemory', action='store_true')
	parser.add_argument('--threshold', type=int, default=None)
	parser.add_argument('--maxRank', type=int, default=None)

	parser.add_argument('--lang', type=str, required=True)

	# flags below are (so far) only useful for English
	parser.add_argument('--all_preds_set_path', type=str, default='')
	parser.add_argument('--backoff_to_predstr', action='store_true')
	parser.add_argument('--threshold_samestr', type=int, default=1)

	parser.add_argument('--T5_size', type=str, default='small')

	parser.add_argument('--data_parallel', action='store_true', help="Used in eval method SS")

	parser.add_argument('--lm_graph_embs_dir', type=str, default='/Users/teddy/PycharmProjects/lm_embedded_EGs/javad_tacl_bb_3_3_global_binc/')
	parser.add_argument('--smooth_p', type=str, default=None, help='For EG smoothing, whether to use [lm / wn / None] smoothing for the premise.')
	parser.add_argument('--smooth_h', type=str, default=None, help='For EG smoothing, whether to use [lm / wn / None] smoothing for the hypothesis.')
	parser.add_argument('--smooth_sim_order', type=float, default=1.0, help='The order to bring the very small similarity values closer to 1, this value should be <= 1.0')
	# parser.add_argument('--ablate', type=str, default='', help='For EG LM smoothing, whether to smooth the p / h even when it is found in the graph.')
	parser.add_argument('--smoothing_k', type=int, default=4, help='How many nearest neighbours to extract.')
	# parser.add_argument('--smoothing_method', type=str, default='lm', help='[lm / wn]')

	parser.add_argument('--num_prems_to_check', type=int, default=None)

	args = parser.parse_args()
	args.CCG = True
	assert args.eval_set in ['dev', 'test']
	assert args.slicing_method in ['disjoint', 'sliding']
	assert args.eval_mode in ['boolean', 'wh', 'wh_masking', 'negi_masking']
	assert args.eval_method in ['bert1A', 'bert2A', 'bert3A', 'bert1B', 'bert2B', 'bert3B', 'T51A', 'T53A', 'eg', 'ss', 'sscont', 'null']
	assert args.wh_label in ['positive', 'negative', 'both']
	assert args.lang in ['zh', 'en']
	assert args.smooth_p is None or args.smooth_p in ['lm', 'wn']
	assert args.smooth_h is None or args.smooth_h in ['lm', 'wn']
	# assert args.ablate in ['', 'p', 'h', 'ph']

	if args.threshold_samestr == 0:
		args.threshold_samestr = False
	elif args.threshold_samestr == 1:
		args.threshold_samestr = True
	else:
		raise AssertionError

	args.fpath = args.fpath_base % (args.version, args.eval_set)
	args.wh_fpath = args.wh_fpath_base % (args.version, args.eval_set)
	args.negi_fpath = args.negi_fpath_base % (args.version, args.eval_set)

	args.eg_dir = os.path.join(args.eg_root, args.eg_name)
	# args.skip_idxes_fn = args.skip_idxes_fn % (args.version, args.eval_set)
	args.result_dir = args.result_dir % (args.version, args.eval_set)
	if not os.path.exists(args.result_dir):
		Path(args.result_dir).mkdir(parents=True, exist_ok=True)
	args.pr_rec_path = os.path.join(args.result_dir, args.pr_rec_fn)
	args.boolean_predictions_path = os.path.join(args.result_dir, args.boolean_predictions_fn)
	args.wh_predictions_path = os.path.join(args.result_dir, args.wh_predictions_fn)
	args.wh_results_path = os.path.join(args.result_dir, args.wh_results_fn)

	datemngr = DateManager(args.lang)
	if args.slicing_method == 'disjoint':
		date_slices, _ = datemngr.setup_dateslices(args.time_interval)
	elif args.slicing_method == 'sliding':
		date_slices, _ = datemngr.setup_dates(args.time_interval)
	else:
		raise AssertionError

	print(args)

	if args.eval_mode in ['wh']:
		qa_eval_wh_main(args, date_slices)
	elif args.eval_mode in ['wh_masking']:  # this is masking the positive entries for wh-QA
		mask_entities_for_entries(args)
	elif args.eval_mode in ['negi_masking']:  # this is masking the negative entries for wh-QA
		mask_negi_entities_for_entries(args)
	elif args.eval_mode in ['boolean']:
		qa_eval_boolean_main(args, date_slices)

	print(f"Finished.")
