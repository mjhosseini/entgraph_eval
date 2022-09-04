from qaeval_utils import DateManager, parse_rel, calc_simscore, duration_format_print
import sys
sys.path.append("..")
sys.path.append("../../DrQA/")
sys.path.append("/Users/teddy/PycharmProjects/DrQA/")

from graph import graph
from qaeval_chinese_general_functions import load_data_entries, type_matched, type_contains, reconstruct_sent_from_rel, \
	calc_per_entry_score_bert, in_context_prediction_bert, find_answers_from_graph
import os
import json
import torch
import transformers
import time
import copy
import statistics
import psutil
from drqa.retriever.tfidf_doc_ranker import TfidfDocRanker

import evaluation.util_chinese
from sklearn.metrics import precision_recall_curve


def qa_eval_wh_all_partitions(args, date_slices, data_entries, entry_restricted_tscores, entry_free_tscores=None,
							  entry_uscores=None, gr=None, loaded_data_refs_by_partition=None,
							  loaded_ref_triples_by_partition=None, suppress=False):
	print(f"Starting qa_eval_wh_all_partitions!")
	if args.eval_method in ['bert1A', 'bert1B', 'bert2A', 'bert2B', 'bert3A', 'bert3B']:
		with torch.no_grad():
			if args.eval_method in ['bert1A', 'bert2A', 'bert3A']:
				bert_tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_dir)
				bert_model = transformers.BertModel.from_pretrained(args.bert_dir)
				args.device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
				bert_model = bert_model.to(args.device)
			elif args.eval_method in ['bert1B', 'bert2B', 'bert3B']:
				bert_tokenizer = transformers.T5Tokenizer.from_pretrained(args.mt5_dir)
				bert_model = transformers.MT5ForConditionalGeneration.from_pretrained(args.mt5_dir)
				if torch.cuda.is_available():
					device_map = {0: [0],
								  1: [1, 2, 3, 4],
								  2: [5, 6, 7]}
					bert_model.parallelize(device_map)
				# in this case, args.device is the first device, the one hosting the embeddings
				args.device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
			else:
				raise AssertionError
			bert_model.eval()
		if args.eval_method in ['bert1A', 'bert1B']:
			ranker = TfidfDocRanker(tfidf_path=args.tfidf_path, articleIds_by_partition_path=args.articleIds_dict_path,
									strict=False)
		else:
			ranker = None
	elif args.eval_method in ['eg']:
		bert_tokenizer = None
		bert_model = None
		ranker = None
		assert gr is not None
	else:
		raise AssertionError

	dur_loadtriples = 0.0
	this_total_num_matches = 0
	sum_data_refs = 0.0
	sum_data = 0.0
	sum_typed_ents = 0.0
	sum_typematches = 0.0

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
		if args.debug and partition_key != '07-26_07-28':
			print(f"Processing only partition ``07-26_07-28'', skipping current partition!")
			continue
		if not suppress:
			print(f"Processing partition {partition_key}! Loading time so far: {dur_loadtriples} seconds")

		partition_triple_path = os.path.join(args.sliced_triples_dir,
											 args.sliced_triples_base_fn % (args.slicing_method, partition_key))
		partition_triples_in_sents = []

		st_loadtriples = time.time()
		if load_triples_flag:
			print("loading triples!")
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

		cur_partition_docids_to_in_partition_sidxes = {}  # This dict is only populated and used in bert1 setting!

		# build up the current-partition-dataset
		cur_partition_data_entries = []
		cur_partition_data_refs = []  # this does not change across different graphs, and can be computed once and loaded each time afterwards!
		cur_partition_global_dids = []
		cur_partition_typematched_flags = []  # this is type-match in the `restricted' sense
		cur_partition_partial_typematched_flags = []  # this is type-match in the `free' sense
		for iid, item in enumerate(data_entries):
			upred, subj, obj, tsubj, tobj = parse_rel(item)
			tm_flag = None
			partial_tm_flag = None
			if gr is not None and not type_matched(gr.types, tsubj, tobj):
				tm_flag = False
				if type_contains(gr.types, item['index_type']):
					partial_tm_flag = True
				else:
					partial_tm_flag = False
			else:
				tm_flag = True
				partial_tm_flag = True
				this_total_num_matches += 1
			assert tm_flag is not None and partial_tm_flag is not None

			if item['partition_key'] == partition_key:
				cur_partition_data_entries.append(item)
				cur_partition_data_refs.append([])
				cur_partition_global_dids.append(iid)
				cur_partition_typematched_flags.append(tm_flag)
				cur_partition_partial_typematched_flags.append(partial_tm_flag)

		# build up entity-pair dict
		indexarg_to_cur_partition_dids = {}
		for cid, ent in enumerate(cur_partition_data_entries):
			upred, subj, obj, tsubj, tobj = parse_rel(ent)
			# if this flag is set to true, then the type of index args in context triples must be the same as the query,
			# although not necessarily the same as the current graph (if the evaluation is for typed entailment graphs)
			if args.assert_indexarg_type:
				indexarg = '::'.join([ent['index_arg'], ent['index_type']])
			else:
				indexarg = ent['index_arg']
			if indexarg not in indexarg_to_cur_partition_dids:
				indexarg_to_cur_partition_dids[indexarg] = []
			indexarg_to_cur_partition_dids[indexarg].append((cid, ent['index_position'], upred))

		if args.eval_method not in ['bert1A', 'bert1B']:
			if load_data_refs_flag is True:
				print("loading data refs!")
				assert len(loaded_data_refs_by_partition[partition_key]) == len(cur_partition_data_refs)
				cur_partition_data_refs = loaded_data_refs_by_partition[partition_key]
			else:
				for sidx, sent_item in enumerate(partition_triples_in_sents):
					if args.debug and sidx > 100000:
						break
					for ridx, r in enumerate(sent_item['rels']):
						rupred, rsubj, robj, rtsubj, rtobj = parse_rel(r)
						if args.assert_indexarg_type:
							r_indexsubj = '::'.join([rsubj, rtsubj])
							r_indexobj = '::'.join([robj, rtobj])
						else:
							r_indexsubj = rsubj
							r_indexobj = robj

						if r_indexsubj in indexarg_to_cur_partition_dids or r_indexobj in indexarg_to_cur_partition_dids:
							if r_indexsubj in indexarg_to_cur_partition_dids:
								rindexarg_pos = 'subj'
								r_indexarg = r_indexsubj
							# CHANGES: the condition above and the condition below are not really mutually exclusive!
							if r_indexobj in indexarg_to_cur_partition_dids:
								rindexarg_pos = 'obj'
								r_indexarg = r_indexobj

							for (cur_partition_did, qindexarg_pos, query_upred) in indexarg_to_cur_partition_dids[r_indexarg]:
								assert qindexarg_pos in ['subj', 'obj']
								if rindexarg_pos == qindexarg_pos:
									aligned = True
								else:
									aligned = False

								# skip that sentence where the query is found! Also skip those relations that are the same as the query relation.
								# TODO: but leave those sentences that have the exact match relations to the query relation be!
								# TODO: if there are other relations in those sentences, and they are extracted, then these sentences
								# TODO: would still be used as part of context!
								if sidx != cur_partition_data_entries[cur_partition_did]['in_partition_sidx']:  # if the context sentence is not the same as the matched query
									if (not args.keep_same_rel_sents) and query_upred == rupred:
										if args.debug:
											pass
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
		else:
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
			if cid % 1000 == 1:
				ct_calcscore = time.time()
				dur_calcscore = ct_calcscore - st_calcscore
				print(f"calculating score for data entry {cid} / {len(cur_partition_data_entries)} for current partition;")
				print(f"percentage of used memory: {psutil.virtual_memory().percent}")
				duration_format_print(dur_calcscore, '')

			cur_score = None
			if args.eval_method in ['bert1A', 'bert1B']:
				ref_sents = []
				query_sent, query_charmap = reconstruct_sent_from_rel(ent, args.max_spansize, mask_answer_flag=True)
				if isinstance(query_sent, list):
					query_sent = ' '.join(query_sent)
				ref_docids, ref_tfidf_scrs = ranker.closest_docs(query_sent, partition_key=ent['partition_key'], k=args.num_refs_bert1)
				assert len(ref_docids) <= args.num_refs_bert1
				# the docids are globally unique, irrespective of the partition key
				# the returned docids should be guaranteed to have been seen in the partition.
				for rdid in ref_docids:
					# print(rdid)
					rsidxes = cur_partition_docids_to_in_partition_sidxes[rdid]
					for rsidx in rsidxes:
						if rsidx != ent['in_partition_sidx']:
							ref_sents.append(partition_triples_in_sents[rsidx]['s'])
				if args.eval_method in ['bert1A']:
					cur_rtscore = calc_per_entry_score_bert(ent, ref_rels=None, ref_sents=ref_sents,
															method=args.eval_method, max_spansize=args.max_spansize,
															bert_model=bert_model, bert_tokenizer=bert_tokenizer,
															bert_device=args.device, max_context_size=args.max_context_size,
															debug=args.debug, is_wh=True, batch_size=args.batch_size)
				elif args.eval_method in ['bert1B']:
					cur_rtscore = in_context_prediction_bert(ent, ref_rels=None, ref_sents=ref_sents, method=args.eval_method,
															 max_spansize=args.max_spansize, bert_model=bert_model,
															 bert_tokenizer=bert_tokenizer, bert_device=args.device,
															 max_seqlength=args.max_t5_seq_length, debug=args.debug,
															 is_wh=True)
				else:
					raise AssertionError
				assert cur_partition_typematched_flags[cid] is True
				cur_ftscore = None
				cur_uscore = None
			elif args.eval_method in ['bert2A', 'bert2B', 'bert3A', 'bert3B']:
				ref_rels = []
				ref_sents = []
				# for Bert methods, ``aligned'' var is not used: whether or not the entity pairs are aligned is unimportant for Bert.
				# the answer types are unimportant as well.
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					ref_sents.append(partition_triples_in_sents[sid]['s'])
					ref_rels.append(partition_triples_in_sents[sid]['rels'][rid])
				if args.eval_method in ['bert2A', 'bert3A']:
					cur_rtscore = calc_per_entry_score_bert(ent, ref_rels=ref_rels, ref_sents=ref_sents,
															method=args.eval_method, max_spansize=args.max_spansize,
															bert_model=bert_model, bert_tokenizer=bert_tokenizer,
															bert_device=args.device, max_context_size=args.max_context_size,
															debug=args.debug, is_wh=True, batch_size=args.batch_size)
				elif args.eval_method in ['bert2B', 'bert3B']:
					cur_rtscore = in_context_prediction_bert(ent, ref_rels=ref_rels, ref_sents=ref_sents,
															 method=args.eval_method, max_spansize=args.max_spansize,
															 bert_model=bert_model, bert_tokenizer=bert_tokenizer,
															 bert_device=args.device, max_seqlength=args.max_t5_seq_length,
															 debug=args.debug, is_wh=True)
				else:
					raise AssertionError
				assert cur_partition_typematched_flags[cid] is True
				cur_ftscore = None
				cur_uscore = None

			elif args.eval_method in ['eg']:
				ref_rels = []
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					this_rel = partition_triples_in_sents[sid]['rels'][rid]
					this_rel['aligned'] = aligned
					ref_rels.append(this_rel)
				cur_rtscore, cur_ftscore, cur_uscore, \
				cur_num_true_entailments = find_answers_from_graph(gr, ent, ref_rels, cur_partition_typematched_flags[cid],
																   cur_partition_partial_typematched_flags[cid],
																  feat_idx=args.eg_feat_idx, debug=args.debug)

				if cur_num_true_entailments is not None:
					sum_typed_ents += cur_num_true_entailments
					sum_typematches += 1
			else:
				raise AssertionError

			cur_entry_global_id = cur_partition_global_dids[cid]

			if cur_partition_typematched_flags[cid] is True:
				assert entry_restricted_tscores[cur_entry_global_id] is None
				if cur_rtscore is None:
					cur_rtscore = 0.0
				entry_restricted_tscores[cur_entry_global_id] = cur_rtscore
			else:
				assert cur_rtscore is None

			if cur_ftscore is not None:
				assert cur_partition_partial_typematched_flags[cid] is True
				for answer_key in cur_ftscore:
					if (not args.ignore_0_for_Avg) or cur_ftscore[answer_key] > 0.0000000001:
						if answer_key not in entry_free_tscores[cur_entry_global_id]:
							entry_free_tscores[cur_entry_global_id][answer_key] = []
						entry_free_tscores[cur_entry_global_id][answer_key].append(cur_ftscore[answer_key])

			if cur_uscore is not None:  # a small number, not zero for numerical stability
				for answer_key in cur_uscore:
					if (not args.ignore_0_for_Avg) or cur_uscore[answer_key] > 0.0000000001:
						if answer_key not in entry_uscores[cur_entry_global_id]:
							entry_uscores[cur_entry_global_id][answer_key] = []
						entry_uscores[cur_entry_global_id][answer_key].append(cur_uscore[answer_key])

	if sum_data > 0:
		avg_refs_per_entry = sum_data_refs / sum_data
		print(f"Average number of references per entry: {avg_refs_per_entry}")
	else:
		print(f"Anomaly! sum_data not larger than zero! sum_data: {sum_data}; sum_data_refs: {sum_data_refs}.")
	if sum_typematches > 0:
		avg_typed_ents = sum_typed_ents / sum_typematches
		print(f"Average number of typed entailment edges utilized: {avg_typed_ents}")
	else:
		print("No type match found! avg_typed_ents equals to 0.")

	return dur_loadtriples, this_total_num_matches


def wh_max_of_weighted(args, rtscr, ftscr, uscr):
	final_scr = {}
	if rtscr is not None:
		for ans in rtscr:
			if ans not in final_scr:
				final_scr[ans] = args.rtscr_w*rtscr[ans]
			else:
				final_scr[ans] = max(final_scr[ans], args.rtscr_w*rtscr[ans])
	if ftscr is not None and not args.ignore_ftscr:
		for ans in ftscr:
			if ans not in final_scr:
				final_scr[ans] = args.ftscr_w*ftscr[ans]
			else:
				if not args.ftscr_backupOnly:
					final_scr[ans] = max(final_scr[ans], args.ftscr_w*ftscr[ans])
	if uscr is not None and not args.ignore_uscr:
		for ans in uscr:
			if ans not in final_scr:
				final_scr[ans] = args.uscr_w*uscr[ans]
			else:
				if not args.uscr_backupOnly:
					final_scr[ans] = max(final_scr[ans], args.uscr_w*uscr[ans])
	# add an 1e-10 background weight to give a slight privilige to predicates found as nodes against predicates not found.
	for ans in final_scr:
		final_scr[ans] += 0.0000000001
	return final_scr


def wh_final_evaluation(args, data_entries, entry_tscores, entry_avg_ftscores=None, entry_avg_uscores=None,
						write_individual_preds=False):
	final_scores = []  # this is typed score if not backupAvg, and back-up-ed score if backupAvg
	final_labels = []

	if entry_avg_ftscores is None:
		entry_avg_ftscores = [None for x in range(len(entry_tscores))]
	if entry_avg_uscores is None:
		entry_avg_uscores = [None for x in range(len(entry_tscores))]

	for eidx, (rtscr, ftscr, uscr, ent) in enumerate(zip(entry_tscores, entry_avg_ftscores, entry_avg_uscores, data_entries)):

		cur_final_scr = wh_max_of_weighted(args, rtscr, ftscr, uscr)
		final_scores.append(cur_final_scr)

		final_labels.append(ent['answer'])

	assert len(final_labels) == len(final_scores) and len(final_labels) == len(data_entries)

	hit_at_X = {1: 0.0, 3: 0.0, 10: 0.0, 30: 0.0, 100: 0.0}
	mrr = 0.0

	for eidx, (cur_fscr, cur_label) in enumerate(zip(final_scores, final_labels)):
		sorted_fscr = {k: v for k, v in sorted(cur_fscr.items(), key=lambda item: item[1], reverse=True)}
		sorted_predictions = list(sorted_fscr.keys())
		print(f"Length of sorted predictions: {len(sorted_predictions)}")
		for X in hit_at_X:
			if cur_label in sorted_predictions[:X]:
				hit_at_X[X] += 1
		try:
			cur_rank = sorted_predictions.index(cur_label)
			mrr += 1 / (float(cur_rank)+1)
		except ValueError as e:
			print(f"eidx: {eidx}; cur_label: {cur_label}; cur_fscr: ")
			print(cur_fscr)

	assert len(final_scores) > 0
	for X in hit_at_X:
		hit_at_X[X] /= float(len(final_scores))
		print(f"Hit @ {X}: %.2f percents;" % (100 * hit_at_X[X]))
	mrr /= float(len(final_scores))
	print(f"Mean reciprocal rank: %.4f" % mrr)

	if args.eval_method in ['bert1A', 'bert2A', 'bert3A', 'bert1B', 'bert2B', 'bert3B']:
		method_ident_str = args.eval_method
	elif args.eval_method in ['eg']:
		method_ident_str = '_'.join([args.eval_method, os.path.split(args.eg_name)[-1], args.eg_suff])
	else:
		raise AssertionError

	with open(args.wh_predictions_path % (method_ident_str, args.wh_label), 'w', encoding='utf8') as fp:
		for s, l in zip(final_scores, final_labels):
			out_item = {'answer': l, 'scores': s}
			out_line = json.dumps(out_item, ensure_ascii=False)
			fp.write(out_line+'\n')

	with open(args.wh_results_path % (method_ident_str, args.wh_label), 'w', encoding='utf8') as fp:
		fp.write(f"MRR: {mrr}\n")
		for X in hit_at_X:
			fp.write(f"Hit @ {X}: {hit_at_X[X]}\n")

	if write_individual_preds is True:
		with open(args.wh_predictions_path % (method_ident_str+'_individuals', args.wh_label), 'w', encoding='utf8') as fp:
			for rtscr, ftscr, uscr, l in zip(entry_tscores, entry_avg_ftscores, entry_avg_uscores, final_labels):
				out_item = {'label': l, 'rtscr': rtscr, 'ftscr': ftscr, 'uscr': uscr}
				out_line = json.dumps(out_item, ensure_ascii=False)
				fp.write(out_line+'\n')


def wh_pr_rec_evaluation(args):

	def add_results_to_entry(ent, res):
		assert isinstance(ent, dict)
		assert ent['answer'] == res['label']
		ent['predictions'] = copy.deepcopy(res)
		final_scr = wh_max_of_weighted(args, res['rtscr'], res['ftscr'], res['uscr'])
		ent['answer_confidence'] = final_scr[ent['answer']] if ent['answer'] in final_scr else 0.0

	assert args.wh_label == 'both'
	posi_data_entries = load_data_entries(args.wh_fpath, posi_only=True)
	negi_data_entries = load_data_entries(args.negi_fpath, posi_only=True)

	if args.eval_method in ['bert1A', 'bert2A', 'bert3A', 'bert1B', 'bert2B', 'bert3B']:
		method_ident_str = args.eval_method
	elif args.eval_method in ['eg']:
		method_ident_str = '_'.join([args.eval_method, os.path.split(args.eg_name)[-1], args.eg_suff])
	else:
		raise AssertionError

	posi_results = []
	negi_results = []
	with open(args.wh_predictions_path % (method_ident_str+'_individuals', 'positive'), 'r', encoding='utf8') as fp:
		for line in fp:
			item = json.loads(line)
			posi_results.append(item)
	with open(args.wh_predictions_path % (method_ident_str+'_individuals', 'negative'), 'r', encoding='utf8') as fp:
		for line in fp:
			item = json.loads(line)
			negi_results.append(item)

	assert len(posi_data_entries) == len(posi_results)
	assert len(negi_data_entries) == len(negi_results)

	for i in range(len(posi_data_entries)):
		add_results_to_entry(posi_data_entries[i], posi_results[i])
	for i in range(len(negi_data_entries)):
		add_results_to_entry(negi_data_entries[i], negi_results[i])

	total_labels = [1 for x in posi_data_entries] + [0 for x in negi_data_entries]
	total_answer_confidence = [ent['answer_confidence'] for ent in posi_data_entries] + [ent['answer_confidence'] for ent in negi_data_entries]

	prec, rec, thres = precision_recall_curve(total_labels, total_answer_confidence)
	assert len(prec) == len(rec) and len(prec) == len(thres) + 1
	auc_value = evaluation.util_chinese.get_auc(prec[1:], rec[1:])
	print(f"Area under curve: {auc_value};")

	posi_entries_by_sidx = {}
	for ent in posi_data_entries:
		if ent['in_partition_sidx'] not in posi_entries_by_sidx:
			posi_entries_by_sidx[ent['in_partition_sidx']] = []
		posi_entries_by_sidx[ent['in_partition_sidx']].append(ent)

	posi_negi_diffs = []  # this should finally be of the same length as the negative samples
	posi_abs_confidences = []  # this should also be of the same length as the negative samples.
	negi_abs_confidences = []  # this should also be of the same length as the negative samples.

	for negi_ent in negi_data_entries:
		posi_ent = None
		for pot_posi_ent in posi_entries_by_sidx[negi_ent['in_partition_sidx']]:
			if pot_posi_ent['partition_key'] == negi_ent['partition_key']:
				assert posi_ent is None
				posi_ent = pot_posi_ent
		assert posi_ent is not None

		cur_posi_negi_diff = posi_ent['answer_confidence'] - negi_ent['answer_confidence']
		cur_posi_abs_confidence = posi_ent['answer_confidence']
		cur_negi_abs_confidence = negi_ent['answer_confidence']

		posi_negi_diffs.append(cur_posi_negi_diff)
		posi_abs_confidences.append(cur_posi_abs_confidence)
		negi_abs_confidences.append(cur_negi_abs_confidence)

	mean_posi_negi_diff = statistics.mean(posi_negi_diffs)
	mean_posi_abs_confidence = statistics.mean(posi_abs_confidences)
	mean_negi_abs_confidence = statistics.mean(negi_abs_confidences)

	print(f"mean posi-negi difference: {mean_posi_negi_diff};")
	print(f"mean diff/posi: {mean_posi_negi_diff / mean_posi_abs_confidence}")
	print(f"mean diff/negi: {mean_posi_negi_diff / mean_negi_abs_confidence}")


def qa_eval_wh_main(args, date_slices):
	if args.wh_label == 'positive':
		data_entries = load_data_entries(args.wh_fpath, posi_only=True)
	elif args.wh_label == 'negative':
		data_entries = load_data_entries(args.negi_fpath, negi_only=True)
	else:
		raise AssertionError

	do_write_individual_preds = (not args.no_write_individual_preds)
	if do_write_individual_preds:
		print(f"Will write individual prediction scores!")
	else:
		print(f"Will NOT write individual prediction scores!")

	assert len(data_entries) > 0
	entry_restricted_tscores = [None for x in range(len(data_entries))]  # tscores where the type of the blank is restricted;
	entry_free_tscores = [{} for x in range(len(data_entries))]  # tscores where the type of the blank is free;
	# formal equation for uscore: avg_{type}(max_{p \in V(type)}(EntScore_{type}(p->q)))
	entry_uscores = [{} for x in range(len(data_entries))]  # uscores with no restrictions on type pairs at all, calculated as in boolean setting.

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
		num_type_pairs_processed = 0
		num_type_pairs_processed_reported_flag = False

		loaded_data_refs_by_partition = None if args.no_ref_cache else {}
		loaded_ref_triples_by_partition = None if args.no_triple_cache else {}

		for f in files:
			if num_type_pairs_processed % 50 == 1 and not num_type_pairs_processed_reported_flag:
				print(f"num processed type pairs: {num_type_pairs_processed}")
				num_type_pairs_processed_reported_flag = True
			if not f.endswith(args.eg_suff):
				continue
			gpath = os.path.join(args.eg_dir, f)
			if os.path.getsize(gpath) < args.min_graphsize:
				continue
			gr = graph.Graph(gpath=gpath, args=args)
			gr.set_Ws()
			all_tps.append(gr.types)

			cur_dur_load_triples, this_num_matches = qa_eval_wh_all_partitions(args, date_slices, data_entries, entry_restricted_tscores,
																			   entry_free_tscores=entry_free_tscores,
																			   entry_uscores=entry_uscores, gr=gr,
																			   loaded_data_refs_by_partition=loaded_data_refs_by_partition,
																			   loaded_ref_triples_by_partition=loaded_ref_triples_by_partition,
																			   suppress=True)
			total_dur_loadtriples += cur_dur_load_triples
			num_type_pairs_processed += 1
			num_type_pairs_processed_reported_flag = False
			this_percent_matches = '%.2f' % (100 * this_num_matches / len(data_entries))
			print(
				f"Finished processing for graph of types: {gr.types[0]}#{gr.types[1]}; num of entries matched: {this_num_matches} -> {this_percent_matches} percents of all entries.")

	elif args.eval_method in ['bert1A', 'bert2A', 'bert3A', 'bert1B', 'bert2B', 'bert3B']:
		total_dur_loadtriples, _ = qa_eval_wh_all_partitions(args, date_slices, data_entries, entry_restricted_tscores,
															 entry_free_tscores=None, entry_uscores=None, gr=None)
	else:
		raise AssertionError

	duration_format_print(total_dur_loadtriples, f"Total duration for loading triples")

	unmatched_types = set()

	for eidx, sc in enumerate(entry_restricted_tscores):
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

		assert sc is not None

	if args.eval_method in ['eg']:
		print('unmatched types: ')
		print(unmatched_types)

	entry_avg_uscores = []
	entry_avg_free_tscores = []
	for eidx, (cur_uscores, cur_free_tscores) in enumerate(zip(entry_uscores, entry_free_tscores)):
		avg_uscr = {}
		avg_ftscr = {}
		for ans in cur_uscores:
			avg_uscr[ans] = sum(cur_uscores[ans]) / float(len(cur_uscores[ans])) if len(cur_uscores[ans]) > 0 else 0.0
		for ans in cur_free_tscores:
			avg_ftscr[ans] = sum(cur_free_tscores[ans]) / float(len(cur_free_tscores[ans])) if len(cur_free_tscores[ans]) > 0 else 0.0
		entry_avg_uscores.append(avg_uscr)
		entry_avg_free_tscores.append(avg_ftscr)
	assert len(entry_restricted_tscores) == len(entry_avg_uscores)
	assert len(entry_restricted_tscores) == len(entry_avg_free_tscores)

	wh_final_evaluation(args, data_entries, entry_restricted_tscores, entry_avg_free_tscores, entry_avg_uscores,
						write_individual_preds=do_write_individual_preds)
