from qaeval_utils import DateManager, parse_rel, calc_simscore, duration_format_print, simple_lemmatize, rel2concise_str, \
read_all_preds_set, get_predstr2preds
import sys
sys.path.append("..")
sys.path.append("/Users/teddy/eclipse-workspace/EntGraph_nick/evaluate")
sys.path.append('/disk/scratch_big/tli/EntGraph_nick/evaluate')
sys.path.append('/disk/scratch/tli/EntGraph_nick/evaluate')
from graph_encoder import GraphDeducer
from graph import graph
from qaeval_chinese_general_functions import load_data_entries, calc_bsln_prec, type_matched, reconstruct_sent_from_rel, \
	calc_per_entry_score_bert, find_entailment_matches_from_graph, calc_per_entry_score_T5, calc_per_entry_score_ss_ssc, \
	compute_ss_auc
from evaluation.util_chinese import get_auc  # this get_auc is reusable in English setting as well.
from sklearn.metrics import precision_recall_curve as pr_curve_sklearn
from pytorch_lightning.metrics.functional.classification import precision_recall_curve as pr_curve_pt

import os
import time
import json
import torch
import transformers


T5_DEVICE_MAPS = {
	'small': {0: [0, 1, 2, 3, 4, 5]},
	'base': {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
	'large': {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [8, 9, 10, 11, 12, 13, 14, 15], 2: [16, 17, 18, 19, 20, 21, 22, 23]},
	'3b': {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8], 2: [9, 10, 11, 12, 13], 3: [14, 15, 16, 17, 18], 4: [19, 20, 21, 22, 23]},
	# '3b': {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11], 4: [12, 13, 14], 5: [15, 16, 17], 6: [18, 19, 20], 7: [21, 22, 23]},
	# '11b': {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15], 4: [16, 17, 18, 19], 5: [20, 21, 22, 23]},
	'11b': {0: [0, 1], 1: [2, 3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11], 4: [12, 13, 14], 5: [15, 16, 17], 6: [18, 19, 20], 7: [21, 22, 23]}
}


def qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
								   entry_uscores=None, entry_sscores=None, entry_prm_missings=None, entry_hyp_missings=None,
								   gr=None, _backup_gr=None, graph_deducer=None,
								   available_graph_types=None, all_preds_set=None, loaded_data_refs_by_partition=None,
								   loaded_ref_triples_by_partition=None, suppress=False):
	# These following values are set to None unless otherwise specified.
	bert_tokenizer = None
	bert_model = None
	ranker = None
	ss_model = None
	ss_preprocessor = None
	_gr_predstr2preds = None
	_bkupgr_predstr2preds = None

	num_refs_bucket = {30: 0, 60: 0, 90: 0, 1000: 0, 3200: 0, 100000000: 0}

	def add_len_to_bucket(lst, bkt):
		before_sum = sum(num_refs_bucket.values())
		lst_len = len(lst)
		for key in bkt:
			if lst_len <= key:
				bkt[key] += 1
				break
		after_sum = sum(num_refs_bucket.values())
		assert after_sum - before_sum == 1
		return

	if args.eval_method in ['bert1A', 'bert2A', 'bert3A']:
		with torch.no_grad():
			bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_dir)
			bert_model = transformers.AutoModel.from_pretrained(args.bert_dir)
			bert_model.eval()
			args.device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
			bert_model = bert_model.to(args.device)
		if args.eval_method in ['bert1A']:
			sys.path.append("../../DrQA/")
			sys.path.append("/Users/teddy/PycharmProjects/DrQA/")
			from drqa.retriever.tfidf_doc_ranker import TfidfDocRanker
			ranker = TfidfDocRanker(tfidf_path=args.tfidf_path, articleIds_by_partition_path=args.articleIds_dict_path,
									strict=False, lang=args.lang)
		else:
			pass
	elif args.eval_method in ['T51A', 'T53A']:
		with torch.no_grad():
			t5_identifier = '-'.join(['t5', args.T5_size])
			t5_path = os.path.join(args.bert_dir, t5_identifier) if os.path.exists(args.bert_dir) else t5_identifier
			print(f"t5 path: {t5_path}")
			bert_tokenizer = transformers.T5Tokenizer.from_pretrained(t5_path)
			bert_model = transformers.T5ForConditionalGeneration.from_pretrained(t5_path)
			bert_model.eval()
			device_map = T5_DEVICE_MAPS[args.T5_size]
			if torch.cuda.is_available():
				bert_model.parallelize(device_map)
		if args.eval_method in ['T51A']:
			sys.path.append("../../DrQA/")
			sys.path.append("/Users/teddy/PycharmProjects/DrQA/")
			from drqa.retriever.tfidf_doc_ranker import TfidfDocRanker
			ranker = TfidfDocRanker(tfidf_path=args.tfidf_path, articleIds_by_partition_path=args.articleIds_dict_path,
									strict=False, lang=args.lang)
		else:
			pass
	elif args.eval_method in ['ss']:
		sys.path.append('/disk/scratch_big/tli/multilingual-lexical-inference/lm-lexical-inference/')
		sys.path.append('/home/s2063487/multilingual-lexical-inference/lm-lexical-inference/')
		sys.path.append('/Users/teddy/PycharmProjects/multilingual-lexical-inference/lm-lexical-inference')
		from src.data.levy_holt import LevyHoltPattern as ssDiscretePreprocessor
		from src.models.multnat_model import MultNatModel as ssDiscreteModel
		with torch.no_grad():
			ss_model = ssDiscreteModel.load_from_checkpoint(args.bert_dir)
			if args.data_parallel and torch.cuda.is_available():
				print(f"Doing Data Parallel")
				ss_model = torch.nn.DataParallel(ss_model, device_ids=[0, 1, 2, 3])
				args.device = torch.device('cpu')
			else:
				args.device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
				ss_model = ss_model.to(args.device)
			ss_preprocessor = ssDiscretePreprocessor(txt_file=None, pattern_file=None, antipattern_file=None, best_k_patterns=None,
									  pattern_chunk_size=5, training=False, curated_auto=False, is_directional=True)
			ss_model.eval()

	elif args.eval_method in ['sscont']:
		sys.path.append('/disk/scratch_big/tli/multilingual-lexical-inference/conan/')
		sys.path.append('/home/s2063487/multilingual-lexical-inference/conan/')
		sys.path.append('/Users/teddy/PycharmProjects/multilingual-lexical-inference/conan')
		from src.data.levy_holt import LevyHolt as ssContPreprocessor
		from src.models.multnat_model import MultNatModel as ssContModel
		with torch.no_grad():
			ss_model = ssContModel.load_from_checkpoint(args.bert_dir)
			ss_preprocessor = ssContPreprocessor(txt_file=None, num_patterns=args.sscont_num_patts,
												 num_tokens_per_pattern=args.sscont_num_toks_per_patt,
					   							 only_sep=False, use_antipatterns=True, training=False,
												 pattern_chunk_size=5)
			ss_model.eval()
			args.device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
			ss_model = ss_model.to(args.device)
	elif args.eval_method in ['eg']:
		assert gr is not None
		if args.lang == 'en':
			assert all_preds_set is not None or not args.backoff_to_predstr
		else:
			assert not args.backoff_to_predstr

		if args.smooth_p == 'lm' or args.smooth_h == 'lm':
			assert graph_deducer is not None
		if args.smooth_p == 'wn' or args.smooth_h == 'wn':
			_gr_predstr2preds = get_predstr2preds(gr)
			_bkupgr_predstr2preds = get_predstr2preds(_backup_gr)

	elif args.eval_method in ['null']:
		pass
	else:
		raise AssertionError

	dur_loadtriples = 0.0
	this_total_num_matches = 0
	sum_data_refs = 0.0
	sum_data = 0.0
	sum_typed_ents = 0.0
	sum_samestr_backoff_typed_ents = 0.0
	sum_typematches = 0.0
	sum_smoothing = 0.0

	if loaded_data_refs_by_partition is None:
		load_data_refs_flag = False
		store_data_refs_flag = False
	elif len(loaded_data_refs_by_partition) == 0:
		assert isinstance(loaded_data_refs_by_partition, dict)
		load_data_refs_flag = False
		store_data_refs_flag = True
	else:
		assert len(loaded_data_refs_by_partition) == len(date_slices) or args.debug
		load_data_refs_flag = True
		store_data_refs_flag = False

	if loaded_ref_triples_by_partition is None:
		load_triples_flag = False
		store_triples_flag = False
	elif len(loaded_ref_triples_by_partition) == 0:
		assert isinstance(loaded_ref_triples_by_partition, dict)
		load_triples_flag = False
		store_triples_flag = True
	else:
		assert len(loaded_ref_triples_by_partition) == len(date_slices) or args.debug
		load_triples_flag = True
		store_triples_flag = False

	for partition_key in date_slices:
		if args.lang == 'zh':
			debug_partition_key = '07-26_07-28'
		elif args.lang == 'en':
			debug_partition_key = '2008_04-21_04-23'
		else:
			raise AssertionError
		if args.debug and partition_key != debug_partition_key:
			print(f"Processing only partition ``{debug_partition_key}'', skipping current partition!")
			continue

		if not suppress:
			print(f"Processing partition {partition_key}! Loading time so far: {dur_loadtriples} seconds; current num_refs_bucket: {num_refs_bucket}")
		partition_triple_path = os.path.join(args.sliced_triples_dir,
											 args.sliced_triples_base_fn % (args.slicing_method, partition_key))
		partition_triples_in_sents = []

		st_loadtriples = time.time()
		if load_triples_flag:
			partition_triples_in_sents = loaded_ref_triples_by_partition[partition_key]
		else:
			with open(partition_triple_path, 'r', encoding='utf8') as fp:
				for line in fp:
					item = json.loads(line)
					partition_triples_in_sents.append(item)
		if store_triples_flag:
			loaded_ref_triples_by_partition[partition_key] = partition_triples_in_sents
		else:
			pass
		et_loadtriples = time.time()
		dur_loadtriples += (et_loadtriples - st_loadtriples)

		# This dict is only populated and used in bert1 setting! The purpose is to find the corresponding sentence indices
		# after retrieving the most relevant documents by tf-idf.
		cur_partition_docids_to_in_partition_sidxes = {}

		# build up the current-partition-dataset
		cur_partition_data_entries = []
		cur_partition_data_refs = []  # this does not change across different graphs, and can be computed once and loaded each time afterwards!
		cur_partition_global_dids = []
		cur_partition_typematched_flags = []
		for iid, item in enumerate(data_entries):
			upred, subj, obj, tsubj, tobj = parse_rel(item)
			tm_flag = None
			if gr is not None and not type_matched(gr.types, tsubj, tobj):
				tm_flag = False
			else:
				tm_flag = True
				this_total_num_matches += 1

			if item['partition_key'] == partition_key:
				cur_partition_data_entries.append(item)
				cur_partition_data_refs.append([])
				cur_partition_global_dids.append(iid)
				cur_partition_typematched_flags.append(tm_flag)

		# build up entity-pair dict
		ep_to_cur_partition_dids = {}
		for cid, ent in enumerate(cur_partition_data_entries):
			upred, subj, obj, tsubj, tobj = parse_rel(ent)
			if args.lang == 'zh':
				pass
			elif args.lang == 'en':
				# correspondingly, the query_keys for ref_sents need to be simple_lemmatized as well.
				subj = simple_lemmatize(subj)
				obj = simple_lemmatize(obj)
			else:
				raise AssertionError

			ep_fwd = '::'.join([subj, obj])
			ep_rev = '::'.join([obj, subj])
			if ep_fwd not in ep_to_cur_partition_dids:
				ep_to_cur_partition_dids[ep_fwd] = []
			ep_to_cur_partition_dids[ep_fwd].append((cid, True, upred))

			if ep_rev not in ep_to_cur_partition_dids:
				ep_to_cur_partition_dids[ep_rev] = []
			ep_to_cur_partition_dids[ep_rev].append((cid, False, upred))

		if args.eval_method not in ['bert1A', 'T51A']:
			# sort out the related sent and rel ids for each entity pair
			# if ``delete-same-rel-sents'', delete the sentences with the exact same relations.
			# (or maybe put that into a filter for positives, the occurrence of entity pairs do not count the ones with the same predicate.)
			if load_data_refs_flag is True:
				assert len(loaded_data_refs_by_partition[partition_key]) == len(cur_partition_data_refs)
				cur_partition_data_refs = loaded_data_refs_by_partition[partition_key]
			else:
				for sidx, sent_item in enumerate(partition_triples_in_sents):
					if args.debug and sidx > 100000:
						break
					for ridx, r in enumerate(sent_item['rels']):
						# TODO: should we lemmatized arguments for alignment? Yes, this has been done.
						rupred, rsubj, robj, rtsubj, rtobj = parse_rel(r)
						if args.lang == 'zh':
							pass
						elif args.lang == 'en':
							# correspondingly, the query_keys for ref_sents need to be simple_lemmatized as well.
							rsubj = simple_lemmatize(rsubj)
							robj = simple_lemmatize(robj)
						else:
							raise AssertionError

						r_ep = '::'.join([rsubj,
										  robj])  # reference entity pair may be in the same order or reversed order w.r.t. the queried entity pair.
						if r_ep in ep_to_cur_partition_dids:
							for (cur_partition_did, aligned, query_upred) in ep_to_cur_partition_dids[r_ep]:
								assert isinstance(aligned, bool)

								# skip that sentence where the query is found! Also skip those relations that are the same as the query relation.
								# TODO: but leave those sentences that have the exact match relations to the query relation be!
								# TODO: if there are other relations in those sentences, and they are extracted, then these sentences
								# TODO: would still be used as part of context!
								if sidx != cur_partition_data_entries[cur_partition_did]['in_partition_sidx']:
									info_leak_flag = False
									if (not args.keep_same_rel_sents) and query_upred == rupred:
										if args.debug:
											# print(f"Same predicate!")
											pass
										info_leak_flag = True
									elif not args.keep_same_rel_sents:
										if args.lang == 'zh':
											pass
										elif args.lang == 'en':
											_, query_upred_str = rel2concise_str(query_upred, '', '', '', '', lang=args.lang)
											_, rupred_str = rel2concise_str(rupred, '', '', '', '', lang=args.lang)
											if query_upred_str == rupred_str:
												# print(f"upred_str: {query_upred_str}; query_upred: {query_upred} rupred: {rupred};")
												info_leak_flag = True
										else:
											raise AssertionError
									else:
										pass

									if info_leak_flag:
										pass
									else:
										cur_partition_data_refs[cur_partition_did].append((sidx, ridx, aligned))
								else:
									if args.debug:
										# print(f"Same sentence: ref rel: {r}; query rel: {cur_partition_data_entries[cur_partition_did]['r']}")
										pass
			if store_data_refs_flag:
				assert loaded_data_refs_by_partition is not None
				loaded_data_refs_by_partition[partition_key] = cur_partition_data_refs
			else:
				pass
		else:  # if args.eval_method in ['bert1A']
			for sidx, sent_item in enumerate(partition_triples_in_sents):
				# Can't do the early stopping below! Will cause sentences to be unmatched for Bert1 method!
				# if args.debug and sidx > 100000:
				# 	break
				sent_docid = str(sent_item['articleId'])
				if sent_docid not in cur_partition_docids_to_in_partition_sidxes:
					cur_partition_docids_to_in_partition_sidxes[sent_docid] = []
				assert sidx not in cur_partition_docids_to_in_partition_sidxes[sent_docid]
				cur_partition_docids_to_in_partition_sidxes[sent_docid].append(sidx)

		for cid, reflst in enumerate(cur_partition_data_refs):
			sum_data += 1
			sum_data_refs += len(reflst)

		st_calcscore = time.time()
		# calculate the confidence value for each entry
		for cid, ent in enumerate(cur_partition_data_entries):
			if cid % 200 == 1:
				ct_calcscore = time.time()
				dur_calcscore = ct_calcscore - st_calcscore
				print(f"calculating score for data entry {cid} / {len(cur_partition_data_entries)} for current partition;")
				duration_format_print(dur_calcscore, '')

			cur_score = None
			cur_uscore = None  # BackUpAvg
			cur_tsscore = None  # BackUpSmoothing, typed.
			cur_usscore = None
			prm_all_missing_flag = None
			hyp_missing_flag = None

			if args.eval_method == 'bert1A':
				ref_sents = []
				query_sent, query_charmap = reconstruct_sent_from_rel(ent, args.max_spansize, lang=args.lang)
				if isinstance(query_sent, list):
					query_sent = ' '.join(query_sent)
				ref_docids, ref_tfidf_scrs = ranker.closest_docs(query_sent, partition_key=ent['partition_key'], k=args.num_refs_bert1)
				assert len(ref_docids) <= args.num_refs_bert1
				for rdid in ref_docids:
					# print(rdid)
					rsidxes = cur_partition_docids_to_in_partition_sidxes[rdid]
					rsidxes = list(set(rsidxes))
					if len(rsidxes) > 100:
						# print(f"BERT1A: reference article has > 100 sentences: {rsidxes}")
						rsidxes = rsidxes[:100]
					for rsidx in rsidxes:
						if rsidx != ent['in_partition_sidx']:
							ref_sents.append(partition_triples_in_sents[rsidx]['s'])

				add_len_to_bucket(ref_sents, num_refs_bucket)

				cur_score = calc_per_entry_score_bert(ent, ref_rels=None, ref_sents=ref_sents,
													  method=args.eval_method,
													  max_spansize=args.max_spansize, bert_model=bert_model,
													  bert_tokenizer=bert_tokenizer, bert_device=args.device,
													  max_context_size=args.max_context_size, debug=args.debug,
													  batch_size=args.batch_size, lang=args.lang)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
				cur_tsscore = None
				cur_usscore = None
			elif args.eval_method in ['T51A']:
				ref_sents = []
				query_sent, query_charmap = reconstruct_sent_from_rel(ent, args.max_spansize, lang=args.lang)
				if isinstance(query_sent, list):
					query_sent = ' '.join(query_sent)
				ref_docids, ref_tfidf_scrs = ranker.closest_docs(query_sent, partition_key=ent['partition_key'],
																 k=args.num_refs_bert1)
				assert len(ref_docids) <= args.num_refs_bert1
				for rdid in ref_docids:
					# print(rdid)
					rsidxes = cur_partition_docids_to_in_partition_sidxes[rdid]
					for rsidx in rsidxes:
						if rsidx != ent['in_partition_sidx']:
							ref_sents.append(partition_triples_in_sents[rsidx]['s'])

				add_len_to_bucket(ref_sents, num_refs_bucket)
				cur_score = calc_per_entry_score_T5(ent, ref_rels=None, ref_sents=ref_sents, method=args.eval_method,
													max_spansize=args.max_spansize, bert_model=bert_model,
													bert_tokenizer=bert_tokenizer, max_context_size=args.max_context_size,
													debug=args.debug, batch_size=args.batch_size, lang=args.lang)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
				cur_tsscore = None
				cur_usscore = None
			elif args.eval_method in ['bert2A', 'bert3A']:
				ref_rels = []
				ref_sents = []
				# for Bert methods, ``aligned'' var is not used: whether or not the entity pairs are aligned is unimportant for Bert.
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					ref_sents.append(partition_triples_in_sents[sid]['s'])
					ref_rels.append(partition_triples_in_sents[sid]['rels'][rid])

				add_len_to_bucket(ref_rels, num_refs_bucket)
				cur_score = calc_per_entry_score_bert(ent, ref_rels=ref_rels, ref_sents=ref_sents,
													  method=args.eval_method,
													  max_spansize=args.max_spansize, bert_model=bert_model,
													  bert_tokenizer=bert_tokenizer, bert_device=args.device,
													  max_context_size=args.max_context_size, debug=args.debug,
													  batch_size=args.batch_size, lang=args.lang)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
				cur_tsscore = None
				cur_usscore = None
			elif args.eval_method in ['T53A']:
				ref_rels = []
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					ref_rels.append(partition_triples_in_sents[sid]['rels'][rid])

				add_len_to_bucket(ref_rels, num_refs_bucket)
				cur_score = calc_per_entry_score_T5(ent, ref_rels=ref_rels, ref_sents=None, method=args.eval_method,
													max_spansize=args.max_spansize, bert_model=bert_model,
													bert_tokenizer=bert_tokenizer, max_context_size=args.max_context_size,
													debug=args.debug, batch_size=args.batch_size, lang=args.lang)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
				cur_tsscore = None
				cur_usscore = None
			elif args.eval_method in ['ss', 'sscont']:
				ref_rels = []
				# for Bert methods, ``aligned'' var is not used: whether or not the entity pairs are aligned is unimportant for Bert.
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					ref_rels.append(partition_triples_in_sents[sid]['rels'][rid])

				add_len_to_bucket(ref_rels, num_refs_bucket)
				cur_score = calc_per_entry_score_ss_ssc(query_ent=ent, all_ref_rels=ref_rels, max_spansize=args.max_spansize,
														method=args.eval_method, ss_model=ss_model,
														ss_data_preprocer=ss_preprocessor, bert_device=args.device,
														max_context_size=args.max_context_size, debug=args.debug,
														is_wh=False, batch_size=args.batch_size, lang=args.lang,
														data_parallel=args.data_parallel)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
				cur_tsscore = None
				cur_usscore = None
			elif args.eval_method in ['eg']:
				ref_rels = []
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					this_rel = partition_triples_in_sents[sid]['rels'][rid]
					this_rel['aligned'] = aligned
					ref_rels.append(this_rel)

				add_len_to_bucket(ref_rels, num_refs_bucket)
				cur_score, cur_uscore, cur_tsscore, cur_usscore, cur_num_true_entailments, cur_num_samestr_true_entailments, \
				cur_num_smoothing, prm_all_missing_flag, hyp_missing_flag = \
					find_entailment_matches_from_graph(gr, _backup_gr, _gr_predstr2preds, _bkupgr_predstr2preds,
													   graph_deducer, ent, ref_rels, cur_partition_typematched_flags[cid],
													   feat_idx=args.eg_feat_idx, all_predstr_to_upreds=all_preds_set,
													   debug=args.debug, lang=args.lang,
													   threshold_samestr=args.threshold_samestr, smooth_p=args.smooth_p,
													   smooth_h=args.smooth_h, smooth_sim_order=args.smooth_sim_order,
													   smoothing_k=args.smoothing_k, available_graph_types=available_graph_types,
													   num_prems_to_check=args.num_prems_to_check)
				if cur_num_true_entailments is not None:
					sum_typed_ents += cur_num_true_entailments
					sum_typematches += 1
				if cur_num_samestr_true_entailments is not None:
					sum_samestr_backoff_typed_ents += cur_num_samestr_true_entailments
				if cur_num_smoothing is not None:
					sum_smoothing += cur_num_smoothing
			elif args.eval_method in ['null']:
				ref_rels = []
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					this_rel = partition_triples_in_sents[sid]['rels'][rid]
					this_rel['aligned'] = aligned
					ref_rels.append(this_rel)
				add_len_to_bucket(ref_rels, num_refs_bucket)
				cur_score = None
				cur_uscore = None
				cur_tsscore = None
				cur_usscore = None
			else:
				raise AssertionError

			# The condition below means, either the current eval_method is some Bert method, or the current entry matches
			# the type of the current graph
			if cur_partition_typematched_flags[cid] is True:
				assert entry_tscores[cur_partition_global_dids[cid]] is None
				assert entry_processed_flags[cur_partition_global_dids[cid]] is False
				if cur_score is None:
					cur_score = 0.0
				entry_tscores[cur_partition_global_dids[cid]] = cur_score
				entry_processed_flags[cur_partition_global_dids[cid]] = True

				assert entry_sscores[cur_partition_global_dids[cid]] is None
				if cur_tsscore is None:
					cur_tsscore = 0.0
				entry_sscores[cur_partition_global_dids[cid]] = cur_tsscore

				# smoothing only happens when type-match flag is True; therefore, type-match flag is only synced to the
				# list in such cases
				assert entry_prm_missings[cur_partition_global_dids[cid]] is None
				entry_prm_missings[cur_partition_global_dids[cid]] = prm_all_missing_flag
				assert entry_hyp_missings[cur_partition_global_dids[cid]] is None
				entry_hyp_missings[cur_partition_global_dids[cid]] = hyp_missing_flag

			else:
				assert cur_score is None
				assert cur_tsscore is None

			# The condition below means, the eval_method is EG, and some non-zero entailment score has been found between
			# the query rel and some reference rel in this type pair. (the uscore means this is ignoring type, we'll average them later)
			# TODO: double check whether backupAvg indeed means backup to the average of all type pairs where some non-zero
			# TODO: entailment score has been found.
			# ⬆️ it is indeed: the predPairFeats were the sum of all entailment scores where some entailment score other
			# than ``None'' was returned; later on it is divided by the value in predPairSumCoefs, which is the number of
			# such entailment scores as described above.
			# TODO: NOTE! The ``other than None'' includes that cases where all-zeros are returned. These cases mean that
			# TODO: both predicates are found in the graph, but no edges connect between them. The meaning of this is that,
			# TODO: this sub-graph does not think there exists an edge between this pair of predicates, that opinion matters,
			# TODO: so this zero-score should be counted in the denominator, and should not be ignored.
			if cur_uscore is not None and ((not args.ignore_0_for_Avg) or cur_uscore > 0.0000000001):  # a small number, not zero for numerical stability
				entry_uscores[cur_partition_global_dids[cid]].append(cur_uscore)

	print(f"Final number-of-references bucket: ")
	print(num_refs_bucket)
	print(f"")

	if sum_data > 0:
		avg_refs_per_entry = sum_data_refs / sum_data
		print(f"Average number of references per entry: {avg_refs_per_entry}")
	else:
		print(f"Anomaly! sum_data not larger than zero! sum_data: {sum_data}; sum_data_refs: {sum_data_refs}.")
	if sum_typematches > 0:
		avg_typed_ents = sum_typed_ents / sum_typematches
		print(f"Average number of typed entailment edges utilized: {avg_typed_ents}")
		avg_samestr_backoff_typed_ents = sum_samestr_backoff_typed_ents / sum_typematches
		print(f"Average number of typed entailment edges additionally found from same_str backoff: {avg_samestr_backoff_typed_ents}")
	else:
		print("No type match found! avg_typed_ents equals to 0.")

	if sum_typematches > 0:
		ratio_smoothing = sum_smoothing / sum_typematches
		print(f"Total ratio of type matches involving LM smoothing: {ratio_smoothing}")
	else:
		print("No type match found! ratio_smoothing equals to 0.")

	return dur_loadtriples, this_total_num_matches


def qa_eval_boolean_main(args, date_slices):
	data_entries = load_data_entries(args.fpath, posi_only=False)
	dataset_bsln_prec = calc_bsln_prec(data_entries)

	entry_processed_flags = [False for x in range(len(data_entries))]
	entry_tscores = [None for x in range(len(data_entries))]
	entry_uscores = [[] for x in range(len(data_entries))]
	entry_sscores = [None for x in range(len(data_entries))]
	entry_prm_missings = [None for x in range(len(data_entries))]  # these are missings flags before smoothing
	entry_hyp_missings = [None for x in range(len(data_entries))]  # these are missings flags before smoothing

	all_tps = []  # all_type_pairs

	total_dur_loadtriples = 0.0

	# There are two loops: one for all entailment sub-graphs, the other for all partitions.
	# Both are too large to store all in memory at once, and entGraphs take longer to load.
	# So in the outer loop, iterate over all type-pairs; for each type pair, retrieve results from the corresponding subgraphs

	if args.eval_method == 'eg':
		if args and args.eg_feat_idx is not None:
			graph.Graph.featIdx = args.eg_feat_idx

		files = os.listdir(args.eg_dir)
		files.sort()

		if args.backoff_to_predstr is True:
			if args.lang == 'en':
				all_preds_set = read_all_preds_set(args.all_preds_set_path)
				print(f"all_preds_set contains predicates in {len(all_preds_set)} unique ORDERED type pairs;")
				print(f"These types are: ")
				print(all_preds_set.keys())
			elif args.lang == 'zh':
				raise AssertionError
			else:
				raise AssertionError
		else:
			all_preds_set = None

		partition_chunk_size = 228
		# partition_chunk_size = 1500
		# partition_chunk_size = 342
		partition_chunks = []
		partition_ite = 0
		while partition_ite < len(date_slices):
			partition_chunks.append(date_slices[partition_ite:partition_ite+partition_chunk_size])
			partition_ite += partition_chunk_size
		assert sum([len(x) for x in partition_chunks]) == len(date_slices)
		print(f"EG Eval: date_slices is split into {len(partition_chunks)} chunks!")

		# TODO: list files, get available_graph_types
		available_graph_types = set()
		_files = os.listdir(args.lm_graph_embs_dir)
		for f in _files:
			if f.endswith('.pkl'):
				gr_t = f.rstrip('.pkl')
				t1, t2 = gr_t.split('#')
				gr_t_rev = '#'.join([t2, t1])
				available_graph_types.add(gr_t)
				available_graph_types.add(gr_t_rev)

		graph_deducer = GraphDeducer("bert", args.lm_graph_embs_dir, valency=2)
		print(f"GraphDeducer initialized!")

		_backup_gr_f = 'thing#thing' + args.eg_suff
		_backup_gpath = os.path.join(args.eg_dir, _backup_gr_f)
		_backup_gr = graph.Graph(gpath=_backup_gpath, args=args)
		_backup_gr.set_Ws()

		for dsc_idx, date_slice_chunk in enumerate(partition_chunks):
			# if dsc_idx < 2:
			# 	print(f"Debugging! Skipping dsc_idx {dsc_idx}!")
			# 	continue

			print(f"Processing date slice chunk idx {dsc_idx} / {len(partition_chunks)};")

			num_type_pairs_processed = 0
			num_type_pairs_processed_reported_flag = False

			loaded_data_refs_by_partition = None if args.no_ref_cache else {}
			loaded_ref_triples_by_partition = None if args.no_triple_cache else {}

			for f in files:
				# if f < 'thing#visual_art':
				# 	print(f"Debugging! Skipping graph {f} before thing#visual_art!")
				# 	continue
				print(f"Currently processing graph {f};")
				if num_type_pairs_processed % 50 == 1 and not num_type_pairs_processed_reported_flag:
					print(f"num processed type pairs: {num_type_pairs_processed}")
					num_type_pairs_processed_reported_flag = True
				if not f.endswith(args.eg_suff):
					continue
				f_grt = f[:-len(args.eg_suff)]
				if args.debug and f_grt != 'thing#person':
				# if f_grt != 'thing#person':
					print(f"Debugging! Only evaluating with person#product graph; current graph type: {f_grt};")
					continue
				gpath = os.path.join(args.eg_dir, f)
				if os.path.getsize(gpath) < args.min_graphsize:
					continue
				gr = graph.Graph(gpath=gpath, args=args)
				gr.set_Ws()
				all_tps.append(gr.types)

				cur_dur_loadtriples, this_num_matches = qa_eval_boolean_all_partitions(args, date_slice_chunk, data_entries, entry_processed_flags, entry_tscores,
											   entry_uscores=entry_uscores, entry_sscores=entry_sscores,
											   entry_prm_missings=entry_prm_missings, entry_hyp_missings=entry_hyp_missings,
											   gr=gr, _backup_gr=_backup_gr,
											   graph_deducer=graph_deducer, available_graph_types=available_graph_types,
											   all_preds_set=all_preds_set,
											   loaded_data_refs_by_partition=loaded_data_refs_by_partition,
											   loaded_ref_triples_by_partition=loaded_ref_triples_by_partition, suppress=True)
				total_dur_loadtriples += cur_dur_loadtriples
				num_type_pairs_processed += 1
				num_type_pairs_processed_reported_flag = False
				this_percent_matches = '%.2f' % (100 * this_num_matches / len(data_entries))
				print(f"Finished processing for graph of types: {gr.types[0]}#{gr.types[1]}; num of entries matched: {this_num_matches} -> {this_percent_matches} percents of all entries.")
	elif args.eval_method in ['bert1A', 'bert2A', 'bert3A', 'T51A', 'T53A', 'ss', 'sscont', 'null']:
		total_dur_loadtriples, _ = qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
									   entry_uscores=None, entry_sscores=None, entry_prm_missings=None,
									   entry_hyp_missings=None, gr=None, _backup_gr=None,
									   graph_deducer=None, available_graph_types=None,
									   all_preds_set=None)
	else:
		raise AssertionError

	duration_format_print(total_dur_loadtriples, f"Total duration for loading triples")

	assert len(entry_processed_flags) == len(entry_tscores)
	unmatched_types = set()

	for eidx, (fl, sc) in enumerate(zip(entry_processed_flags, entry_tscores)):
		if args.eval_method in ['eg']:
			entry_rel = data_entries[eidx]
			upred, subj, obj, tsubj, tobj = parse_rel(entry_rel)
			matched_flag = False
			for tp in all_tps:
				if type_matched(tp, tsubj, tobj):
					matched_flag = True
			if matched_flag is False:
				if ('::'.join([tsubj, tobj]) not in unmatched_types) and (
						'::'.join([tobj, tsubj]) not in unmatched_types):
					unmatched_types.add('::'.join([tsubj, tobj]))
				continue

		assert fl is True or args.debug
		assert sc is not None or args.debug

	if args.eval_method in ['eg']:
		print('unmatched types: ')
		print(unmatched_types)

	entry_avg_uscores = []
	for eidx, cur_uscores in enumerate(entry_uscores):
		avg_uscr = sum(cur_uscores) / float(len(cur_uscores)) if len(cur_uscores) > 0 else 0.0
		entry_avg_uscores.append(avg_uscr)
	assert len(entry_tscores) == len(entry_avg_uscores) == len(entry_sscores)

	# if args.store_skip_idxes:
	# 	with open(args.skip_idxes_fn, 'w', encoding='utf8') as fp:
	# 		skip_idxes = []
	# 		for eidx, flg in enumerate(entry_processed_flags):
	# 			if not flg:
	# 				skip_idxes.append(eidx)
	# 		json.dump(skip_idxes, fp, ensure_ascii=False)
	# else:
	# 	with open(args.skip_idxes_fn, 'r', encoding='utf8') as fp:
	# 		skip_idxes = json.load(fp)
	# 	for si in skip_idxes:
	# 		assert si < len(data_entries)
	# 	for eidx, flg in enumerate(entry_processed_flags):
	# 		assert eidx in skip_idxes or flg

	# this ``skipping those data entries whose type-pairs unmatched by any sub-graph'' thing, it should not be
	# necessary with backupAvg, and should not be reasonable without backupAvg. It kind of biases the evaluation.

	prm_missing_cnt = 0
	hyp_missing_cnt = 0
	for (prm_missing_flg, hyp_missing_flg) in zip(entry_prm_missings, entry_hyp_missings):
		if args.eval_method in ['eg']:
			if not (prm_missing_flg in [True, False] or args.debug):
				print(f"prm missing flag anomaly!")
			if not (hyp_missing_flg in [True, False] or args.debug):
				print(f"hyp missing flag anomaly!")
		if prm_missing_flg is True:
			prm_missing_cnt += 1
		else:  # this ``else'' can be either False or None
			pass
		if hyp_missing_flg is True:
			hyp_missing_cnt += 1
		else:  # this ``else'' can be either False or None
			pass

	final_scores = []  # this is typed score if not backupAvg, and back-up-ed score if backupAvg
	final_labels = []
	tscr_in_final_cnt = 0
	uscr_in_final_cnt = 0
	sscr_in_final_cnt = 0
	null_in_final_cnt = 0

	for eidx, (tscr, uscr, sscr, ent) in enumerate(zip(entry_tscores, entry_avg_uscores, entry_sscores, data_entries)):
		no_smooth_final_scr = None
		no_smooth_source = None

		if tscr is not None and tscr > 0:
			no_smooth_final_scr = tscr
			no_smooth_source = 't'
		elif args.backupAvg and uscr is not None and uscr > 0:
			no_smooth_final_scr = uscr
			no_smooth_source = 'u'
		else:
			pass

		if sscr is not None and sscr > 0 and (no_smooth_final_scr is None or sscr > no_smooth_final_scr):
			final_scores.append(sscr)
			sscr_in_final_cnt += 1
		else:
			if no_smooth_final_scr is not None and no_smooth_final_scr > 0:
				final_scores.append(no_smooth_final_scr)
				if no_smooth_source == 't':
					tscr_in_final_cnt += 1
				elif no_smooth_source == 'u':
					uscr_in_final_cnt += 1
				else:
					raise AssertionError
			else:
				final_scores.append(0.0)
				null_in_final_cnt += 1

		if bool(ent['label']) is True:
			final_labels.append(1)

		elif bool(ent['label']) is False:
			final_labels.append(0)
		else:
			raise AssertionError
	assert len(final_labels) == len(final_scores) and len(final_labels) == len(data_entries)
	final_scores = [float(x.cpu()) if isinstance(x, torch.Tensor) else x for x in final_scores]
	print(f"final labels: ")
	print(final_labels)
	print(f"final scores: ")
	print(final_scores)
	skl_prec, skl_rec, skl_thres = pr_curve_sklearn(final_labels, final_scores)
	assert len(skl_prec) == len(skl_rec) and len(skl_prec) == len(skl_thres) + 1
	# perhaps report the AUC_BASELINE and the normalized AUC as well!
	skl_auc_value = get_auc(skl_prec[1:], skl_rec[1:])
	print(f"Hosseini Area under curve: {skl_auc_value};")
	print(f"Final scores from tscr: {tscr_in_final_cnt}; from uscr: {uscr_in_final_cnt}; from sscr: {sscr_in_final_cnt}; "
		  f"empty: {null_in_final_cnt};")

	print(f"Premise missing cnt: {prm_missing_cnt} / {len(entry_prm_missings)};")
	print(f"Hypothesis missing cnt: {hyp_missing_cnt} / {len(entry_hyp_missings)}")

	try:
		final_labels_pt = torch.tensor(final_labels)
		final_scores_pt = torch.tensor(final_scores)
		pt_prec, pt_rec, pt_thres = pr_curve_pt(final_scores_pt, final_labels_pt)
		ss_bsln_auc = compute_ss_auc(
			pt_prec, pt_rec,
			filter_threshold=dataset_bsln_prec
		)
		ss_50_auc = compute_ss_auc(
			pt_prec, pt_rec,
			filter_threshold=0.5
		)

		ss_rel_prec = torch.tensor([max(p - dataset_bsln_prec, 0) for p in pt_prec], dtype=torch.float)
		ss_rel_rec = torch.tensor([r for r in pt_rec], dtype=torch.float)
		ss_auc_norm = compute_ss_auc(
			ss_rel_prec, ss_rel_rec,
			filter_threshold=0.0
		)
		ss_auc_norm /= (1 - dataset_bsln_prec)
		print(f"S&S 50 AUC: {ss_50_auc};")
		print(f"S&S bsln AUC: {ss_bsln_auc};")
		print(f"S&S AUC NORM: {ss_auc_norm};")
		print("")
		print("")
		# print(f"p\tr\tt")
		# for p, r, t in zip(pt_prec, pt_rec, pt_thres):
		# 	print(f"{p}\t{r}\t{t}")

	except Exception as e:
		print(f"Exception when calculating S&S style AUC!")
		print(e)
		ss_50_auc = None
		ss_bsln_auc = None
		ss_auc_norm = None

	if args.eval_method in ['bert1A', 'bert2A', 'bert3A']:
		bert_names = args.bert_dir.split('/')
		bert_names = [x for x in bert_names if len(x) > 0]
		bert_lastname = bert_names[-1]
		method_ident_str = args.eval_method + '_' + bert_lastname
	elif args.eval_method in ['ss', 'sscont']:
		model_name_lst = args.bert_dir.split('/')
		print(f"model_name_lst: {model_name_lst}")
		assert len(model_name_lst) >= 3
		model_name = '_'.join(model_name_lst[-3:-1])
		method_ident_str = '_'.join([args.eval_method, model_name])
	elif args.eval_method in ['eg']:
		method_ident_str = '_'.join([args.eval_method, os.path.split(args.eg_name)[-1], args.eg_suff])
		if args.backoff_to_predstr:
			method_ident_str += f'_backoff2predstr'
		if args.smooth_p is not None and args.smooth_h is not None:
			method_ident_str += f"_smoothboth_p%s_h%s_%.1f" % (args.smooth_p, args.smooth_h, args.smooth_sim_order)
		elif args.smooth_p:
			method_ident_str += f"_smoothp%s_%.1f" % (args.smooth_p, args.smooth_sim_order)
		elif args.smooth_h:
			method_ident_str += f"_smoothh%s_%.1f" % (args.smooth_h, args.smooth_sim_order)
		else:
			method_ident_str += f"_nosmooth"
		if args.num_prems_to_check is not None:
			method_ident_str += f'_numprem{args.num_prems_to_check}'
	elif args.eval_method in ['T51A', 'T53A']:
		method_ident_str = '_'.join([args.eval_method, args.T5_size])
	elif args.eval_method in ['null']:
		method_ident_str = args.eval_method
	else:
		raise AssertionError
	with open(args.boolean_predictions_path % method_ident_str, 'w', encoding='utf8') as fp:
		for t, u, smt, s, l in zip(entry_tscores, entry_avg_uscores, entry_sscores, final_scores, final_labels):
			fp.write(f"{t}\t{u}\t{smt}\t{s}\t{l}\n")

	with open(args.pr_rec_path % method_ident_str, 'w', encoding='utf8') as fp:
		fp.write(f"Hosseini auc: {skl_auc_value}\n")
		fp.write(f"S&S 50 AUC: {ss_50_auc}")
		fp.write(f"S&S bsln AUC: {ss_bsln_auc}")
		fp.write(f"S&S AUC NORM: {ss_auc_norm}")
		for p, r, t in zip(skl_prec[1:], skl_rec[1:], skl_thres):
			fp.write(f"{p}\t{r}\t{t}\n")

	print(f"Finished!")
