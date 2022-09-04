import json
import sys
sys.path.append("..")
sys.path.append("../../DrQA/")
sys.path.append("/Users/teddy/PycharmProjects/DrQA/")

import numpy as np
from graph import graph
from lemma_baseline import qa_utils_chinese
import evaluation.util_chinese
from lemma_baseline import chinese_baselines
import os
import argparse
from qaeval_utils import DateManager, parse_rel, calc_simscore, duration_format_print
from sklearn.metrics import precision_recall_curve
import torch
import transformers
import psutil
import copy
import time
from drqa.retriever.tfidf_doc_ranker import TfidfDocRanker
import math


def load_data_entries(fpath, posi_only=False):
	data_entries = []
	with open(fpath, 'r', encoding='utf8') as fp:
		for line in fp:
			item = json.loads(line)
			assert item['label'] is not None and isinstance(item['label'], bool)
			# if the posi_only flag is set to True, then don't load the negatives! (This is reserved for wh-question answering (objects))
			if item['label'] is not True and posi_only:
				continue
			data_entries.append(item)
	process = psutil.Process(os.getpid())
	print(f"Current memory usage in bytes: {process.memory_info().rss}")  # in bytes
	return data_entries


def find_all_matches_in_string(string, pattern):
	id_list = []
	offset = 0
	while True:
		cur_id = string.find(pattern, offset)
		if cur_id < 0:
			break
		id_list.append(cur_id)
		offset = cur_id + len(pattern)
	return id_list


# From a given sentence, fetch the most relevant span to a given rel; if the rel is not given (None), just truncate the
# sentence to its first max_spansize tokens.
def fetch_span_by_rel(sent, rel, max_spansize):
	if len(sent) <= max_spansize:
		return sent
	if rel is None:
		return sent[:max_spansize]

	upred, subj, obj, tsubj, tobj = parse_rel(rel)
	subj_ids = find_all_matches_in_string(sent, subj)
	obj_ids = find_all_matches_in_string(sent, obj)

	if len(subj_ids) == 0 or len(obj_ids) == 0:
		return sent[:max_spansize]

	selected_sid = None
	selected_oid = None
	min_dist = len(sent)

	for sid in subj_ids:
		for oid in obj_ids:
			if abs(sid - oid) < min_dist:
				selected_sid = sid
				selected_oid = oid
				min_dist = abs(sid - oid)
	mid = int(selected_sid + selected_oid) // 2
	if mid < max_spansize // 2:
		return sent[:max_spansize]
	if mid > len(sent) - (max_spansize // 2):
		assert len(sent) - max_spansize > 0
		return sent[len(sent) - max_spansize:]

	assert mid - (max_spansize // 2) >= 0 and mid + (max_spansize // 2) <= len(sent)
	return sent[(mid - (max_spansize // 2)):(mid + (max_spansize // 2))]


def reconstruct_sent_from_rel(rel, max_spansize):
	upred, subj, obj, tsubj, tobj = parse_rel(rel)
	assert upred[0] == '(' and upred[-1] == ')'
	upred_surface_form = upred[1:-1]
	upred_surface_form = upred_surface_form.split('.1,')
	assert len(upred_surface_form) == 2
	upred_surface_form = upred_surface_form[0]
	upred_xidxs = find_all_matches_in_string(upred_surface_form, '·X·')

	if len(upred_xidxs) == 0:
		upred_surface_form = upred_surface_form.replace('·', '')
		reconstructed_sent = subj + upred_surface_form + obj
	elif len(upred_xidxs) == 1:
		upred_surface_form = upred_surface_form.replace('·X·', obj)
		upred_surface_form = upred_surface_form.replace('·', '')
		reconstructed_sent = subj + upred_surface_form  # we have stuck the object back into the predicate!
	else:
		raise AssertionError
	if len(reconstructed_sent) > max_spansize:
		reconstructed_sent = reconstructed_sent[:max_spansize]
	return reconstructed_sent


def type_matched(types_lst_pointer, tsubj, tobj):
	types_lst = copy.deepcopy(types_lst_pointer)
	assert len(types_lst) == 2
	if types_lst[0][-2:] == '_1':
		assert types_lst[1][-2:] == '_2'
		types_lst[0] = types_lst[0][:-2]
		types_lst[1] = types_lst[1][:-2]
	if types_lst[0][-2:] == '_2':
		assert types_lst[1][-2:] == '_1'
		types_lst[0] = types_lst[0][:-2]
		types_lst[1] = types_lst[1][:-2]

	if tsubj[-2:] == '_1':
		assert tobj[-2:] == '_2'
		tsubj = tsubj[:-2]
		tobj = tobj[:-2]
	if tsubj[-2:] == '_2':
		assert tobj[-2:] == '_1'
		tsubj = tsubj[:-2]
		tobj = tobj[:-2]

	if tsubj == types_lst[0] and tobj == types_lst[1]:
		return True
	elif tsubj == types_lst[1] and tobj == types_lst[0]:
		return True
	else:
		return False


def calc_per_entry_score_bert(query_ent, ref_rels, ref_sents, method, max_spansize, bert_model, bert_tokenizer,
							  bert_device, debug=False):
	assert method in ['bert1', 'bert2', 'bert3']
	# the data entry can be positive or negative, so that must be scenario 3; but the references here can be scenario 2
	query_sent = reconstruct_sent_from_rel(query_ent, max_spansize)
	query_toks = bert_tokenizer([query_sent], padding=True)
	query_toks = query_toks.convert_to_tensors('pt')
	query_toks = query_toks.to(bert_device)
	query_outputs = bert_model(**query_toks)
	query_vecs = query_outputs.last_hidden_state
	assert query_vecs.shape[0] == 1
	query_vecs = query_vecs[:, 0, :].cpu().detach().numpy()

	ref_emb_inputstrs = []
	ref_emb_outputvecs = []
	if method in ['bert2', 'bert3']:
		assert len(ref_rels) == len(ref_sents)
		for rrel, rsent in zip(ref_rels, ref_sents):
			if method == 'bert2':
				ref_emb_inputstrs.append(fetch_span_by_rel(rsent, rrel, max_spansize=max_spansize))
			elif method == 'bert3':
				ref_emb_inputstrs.append(reconstruct_sent_from_rel(rrel, max_spansize))
			else:
				raise AssertionError
	else:
		for rsent in ref_sents:
			if method == 'bert1':
				ref_emb_inputstrs.append(fetch_span_by_rel(rsent, None, max_spansize=max_spansize))
			else:
				raise AssertionError

	ref_emb_chunks = []
	chunk_size = 32
	offset = 0
	while offset < len(ref_emb_inputstrs):
		ref_emb_chunks.append((offset, min(offset + chunk_size, len(ref_emb_inputstrs))))
		offset += chunk_size
	for chunk in ref_emb_chunks:
		if chunk[1] == chunk[0]:  # do not attempt to send empty input into the model!
			continue
		ref_emb_inputtoks = bert_tokenizer(ref_emb_inputstrs[chunk[0]:chunk[1]], padding=True)
		ref_emb_inputtoks = ref_emb_inputtoks.convert_to_tensors('pt')
		ref_emb_inputtoks = ref_emb_inputtoks.to(bert_device)
		ref_encoder_outputs = bert_model(**ref_emb_inputtoks)
		ref_encoder_outputs = ref_encoder_outputs.last_hidden_state
		if debug:
			print(ref_encoder_outputs.shape)
		ref_encoder_outputs = ref_encoder_outputs[:, 0, :].cpu().detach().numpy()
		for bidx in range(ref_encoder_outputs.shape[0]):
			ref_emb_outputvecs.append(ref_encoder_outputs[bidx, :])
	assert len(ref_emb_outputvecs) == len(ref_emb_inputstrs)
	ref_emb_outputvecs = np.array(ref_emb_outputvecs)
	if len(ref_emb_inputstrs) > 0:
		cur_sims = calc_simscore(query_vecs, ref_emb_outputvecs)
		assert len(cur_sims.shape) == 2 and cur_sims.shape[0] == 1
		cur_max_sim = np.amax(cur_sims)
		cur_argmax_sim = np.argmax(cur_sims)
		if debug:
			print(f"cur sims shape: {cur_sims.shape}")
			print(f"query rel: {query_ent['r']}")
			if ref_rels is not None:
				print(f"best ref rel: {ref_rels[cur_argmax_sim]['r']}")
			else:
				print(f"Best ref sent: {ref_sents[cur_argmax_sim]}")
	else:
		cur_max_sim = 0.0
		if debug:
			print(f"No relevant relations found!")
	return cur_max_sim


def find_entailment_matches_from_graph(_graph, ent, ref_rels, typematch_flag, feat_idx, debug=False):
	# find entailment matches for one entry from one graph
	maximum_tscore = None
	maximum_uscore = None
	max_tscore_ref = None
	max_uscore_ref = None
	q_upred, q_subj, q_obj, q_tsubj, q_tobj = parse_rel(ent)
	assert '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
	if q_tsubj == q_tobj:
		q_tsubj = q_tsubj + '_1'
		q_tobj = q_tobj + '_2'
	q_tpred_querytype = '#'.join([q_upred, q_tsubj, q_tobj])
	assert len(_graph.types) == 2
	q_tpred_graphtype_fwd = '#'.join([q_upred, _graph.types[0], _graph.types[1]])
	q_tpred_graphtype_rev = '#'.join([q_upred, _graph.types[1], _graph.types[0]])

	num_true_entailments = 0.0

	for rrel in ref_rels:
		r_upred, r_subj, r_obj, r_tsubj, r_tobj = parse_rel(rrel)
		assert '_1' not in r_tsubj and '_2' not in r_tsubj and '_1' not in r_tobj and '_2' not in r_tobj
		if r_tsubj == r_tobj:
			# the assertion below seems deprecated, the ref argument types are allowed to be different from the query
			# argument types, but in these cases the ref argument types would be ignored.
			# assert q_tsubj[:-2] == q_tobj[:-2] and '_1' in q_tsubj and '_2' in q_tobj
			if rrel['aligned'] is True:
				r_tsubj = r_tsubj + '_1'
				r_tobj = r_tobj + '_2'
			elif rrel['aligned'] is False:
				r_tsubj = r_tsubj + '_2'
				r_tobj = r_tobj + '_1'
			else:
				raise AssertionError
		else:
			# assert q_tsubj != q_tobj and '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
			pass

		if rrel['aligned'] is True:
			r_tpred_querytype = '#'.join([r_upred, q_tsubj, q_tobj])
			r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
			r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
		elif rrel['aligned'] is False:
			r_tpred_querytype = '#'.join([r_upred, q_tobj, q_tsubj])
			r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
			r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
		else:
			raise AssertionError

		# print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
		# TODO: Attention! Check this feat_idx matter!
		effective_feat_idx = feat_idx + 0
		if typematch_flag is True:
			assert (q_tsubj == _graph.types[0] and q_tobj == _graph.types[1]) or \
				   (q_tsubj == _graph.types[1] and q_tobj == _graph.types[0])

			cur_tscores = _graph.get_features(r_tpred_querytype, q_tpred_querytype)
			if cur_tscores is not None:
				# print(f"cur tscores length: {len(cur_tscores)}")
				# print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
				if cur_tscores[0] > 0 and cur_tscores[0] < 0.99:
					num_true_entailments += 1
				curfeat_cur_tscore = cur_tscores[effective_feat_idx]
				if maximum_tscore is None or curfeat_cur_tscore > maximum_tscore:
					max_tscore_ref = r_tpred_querytype
					maximum_tscore = curfeat_cur_tscore
		else:
			num_true_entailments = None

		cur_uscores_fwd = _graph.get_features(r_tpred_graphtype_fwd, q_tpred_graphtype_fwd)
		cur_uscores_rev = _graph.get_features(r_tpred_graphtype_rev, q_tpred_graphtype_rev)

		if cur_uscores_fwd is not None:
			if cur_uscores_fwd[1] > 0 and cur_uscores_fwd[1] < 0.99:
				# print(cur_uscores_fwd)
				# [0.43617837 0.2739726  0.16573886 0.18112025 0.13438165 0.09970408, 0.5        0.5        1.         0.5        1.         1.        ]
				# print("!")
				pass
			curfeat_cur_uscore_fwd = cur_uscores_fwd[effective_feat_idx]
			if maximum_uscore is None or curfeat_cur_uscore_fwd > maximum_uscore:
				max_uscore_ref = r_tpred_graphtype_fwd
				maximum_uscore = curfeat_cur_uscore_fwd
		if cur_uscores_rev is not None:
			curfeat_cur_uscore_rev = cur_uscores_rev[effective_feat_idx]
			if maximum_uscore is None or curfeat_cur_uscore_rev > maximum_uscore:
				max_uscore_ref = r_tpred_graphtype_rev
				maximum_uscore = curfeat_cur_uscore_rev

	if debug and maximum_tscore is not None:
		print(f"query: {q_tpred_querytype}; max_tscore_ref: {max_tscore_ref}; max_uscore_ref: {max_uscore_ref}")

	return maximum_tscore, maximum_uscore, num_true_entailments


def qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
								   entry_uscores=None, gr=None, loaded_data_refs_by_partition=None,
								   loaded_ref_triples_by_partition=None, suppress=False):
	if args.eval_method in ['bert1', 'bert2', 'bert3']:
		with torch.no_grad():
			bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')
			bert_model = transformers.BertModel.from_pretrained('bert-base-chinese')
			bert_model.eval()
			args.device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
			bert_model = bert_model.to(args.device)
		if args.eval_method in ['bert1']:
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
			if not suppress:
				print(f"Processing only partition ``07-26_07-28'', skipping current partition!")
			continue

		if not suppress:
			print(f"Processing partition {partition_key}! Loading time so far: {dur_loadtriples} seconds")
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

		cur_partition_docids_to_in_partition_sidxes = {}  # This dict is only populated and used in bert1 setting!

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
			ep_fwd = '::'.join([subj, obj])
			ep_rev = '::'.join([obj, subj])
			if ep_fwd not in ep_to_cur_partition_dids:
				ep_to_cur_partition_dids[ep_fwd] = []
			ep_to_cur_partition_dids[ep_fwd].append((cid, True, upred))

			if ep_rev not in ep_to_cur_partition_dids:
				ep_to_cur_partition_dids[ep_rev] = []
			ep_to_cur_partition_dids[ep_rev].append((cid, False, upred))

		if args.eval_method not in ['bert1']:
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
						rupred, rsubj, robj, rtsubj, rtobj = parse_rel(r)
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
									if (not args.keep_same_rel_sents) and query_upred == rupred:
										if args.debug:
											# print(f"Same predicate!")
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
		else:  # if args.eval_method in ['bert1']
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
			if cid % 2000 == 1:
				ct_calcscore = time.time()
				dur_calcscore = ct_calcscore - st_calcscore
				print(f"calculating score for data entry {cid} / {len(cur_partition_data_entries)} for current partition;")
				duration_format_print(dur_calcscore, '')

			cur_score = None
			if args.eval_method == 'bert1':
				ref_sents = []
				query_sent = reconstruct_sent_from_rel(ent, args.max_spansize)
				ref_docids, ref_tfidf_scrs = ranker.closest_docs(query_sent, partition_key=ent['partition_key'], k=args.num_refs_bert1)
				assert len(ref_docids) <= args.num_refs_bert1
				for rdid in ref_docids:
					# print(rdid)
					rsidxes = cur_partition_docids_to_in_partition_sidxes[rdid]
					for rsidx in rsidxes:
						ref_sents.append(partition_triples_in_sents[rsidx]['s'])

				cur_score = calc_per_entry_score_bert(ent, ref_rels=None, ref_sents=ref_sents,
													  method=args.eval_method,
													  max_spansize=args.max_spansize, bert_model=bert_model,
													  bert_tokenizer=bert_tokenizer, bert_device=args.device,
													  debug=args.debug)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
			elif args.eval_method in ['bert2', 'bert3']:
				ref_rels = []
				ref_sents = []
				# for Bert methods, ``aligned'' var is not used: whether or not the entity pairs are aligned is unimportant for Bert.
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					ref_sents.append(partition_triples_in_sents[sid]['s'])
					ref_rels.append(partition_triples_in_sents[sid]['rels'][rid])
				cur_score = calc_per_entry_score_bert(ent, ref_rels=ref_rels, ref_sents=ref_sents,
													  method=args.eval_method,
													  max_spansize=args.max_spansize, bert_model=bert_model,
													  bert_tokenizer=bert_tokenizer, bert_device=args.device,
													  debug=args.debug)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
			elif args.eval_method in ['eg']:
				ref_rels = []
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					this_rel = partition_triples_in_sents[sid]['rels'][rid]
					this_rel['aligned'] = aligned
					ref_rels.append(this_rel)
				cur_score, cur_uscore, cur_num_true_entailments = find_entailment_matches_from_graph(gr, ent, ref_rels,
																		   cur_partition_typematched_flags[cid],
																		   feat_idx=args.eg_feat_idx, debug=args.debug)
				if cur_num_true_entailments is not None:
					sum_typed_ents += cur_num_true_entailments
					sum_typematches += 1
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
			else:
				assert cur_score is None

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


def qa_eval_boolean_bert1(args, data_entries, entry_processed_flags, entry_tscores):
	pass


def qa_eval_boolean_main(args, date_slices):
	data_entries = load_data_entries(args.fpath, posi_only=False)

	entry_processed_flags = [False for x in range(len(data_entries))]
	entry_tscores = [None for x in range(len(data_entries))]
	entry_uscores = [[] for x in range(len(data_entries))]

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

		loaded_data_refs_by_partition = {}
		loaded_ref_triples_by_partition = {}

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

			cur_dur_loadtriples, this_num_matches = qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
										   entry_uscores=entry_uscores, gr=gr, loaded_data_refs_by_partition=loaded_data_refs_by_partition,
										   loaded_ref_triples_by_partition=loaded_ref_triples_by_partition, suppress=True)
			total_dur_loadtriples += cur_dur_loadtriples
			num_type_pairs_processed += 1
			num_type_pairs_processed_reported_flag = False
			this_percent_matches = '%.2f' % (100 * this_num_matches / len(data_entries))
			print(f"Finished processing for graph of types: {gr.types[0]}#{gr.types[1]}; num of entries matched: {this_num_matches} -> {this_percent_matches} percents of all entries.")
	elif args.eval_method in ['bert1', 'bert2', 'bert3']:
		total_dur_loadtriples, _ = qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
									   entry_uscores=None, gr=None)
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

		assert fl is True
		assert sc is not None

	if args.eval_method in ['eg']:
		print('unmatched types: ')
		print(unmatched_types)

	entry_avg_uscores = []
	for eidx, cur_uscores in enumerate(entry_uscores):
		avg_uscr = sum(cur_uscores) / float(len(cur_uscores)) if len(cur_uscores) > 0 else 0.0
		entry_avg_uscores.append(avg_uscr)
	assert len(entry_tscores) == len(entry_avg_uscores)

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
	final_scores = []  # this is typed score if not backupAvg, and back-up-ed score if backupAvg
	final_labels = []
	for eidx, (tscr, uscr, ent) in enumerate(zip(entry_tscores, entry_avg_uscores, data_entries)):

		if tscr is not None and tscr > 0:
			final_scores.append(tscr)
		elif args.backupAvg and uscr is not None and uscr > 0:
			final_scores.append(uscr)
		else:
			final_scores.append(0.)
		if bool(ent['label']) is True:
			final_labels.append(1)
		elif bool(ent['label']) is False:
			final_labels.append(0)
		else:
			raise AssertionError
	assert len(final_labels) == len(final_scores) and len(final_labels) == len(data_entries)

	prec, rec, thres = precision_recall_curve(final_labels, final_scores)
	assert len(prec) == len(rec) and len(prec) == len(thres) + 1
	auc_value = evaluation.util_chinese.get_auc(prec[1:], rec[1:])
	print(f"Area under curve: {auc_value};")

	if args.eval_method in ['bert1', 'bert2', 'bert3']:
		method_ident_str = args.eval_method
	elif args.eval_method in ['eg']:
		method_ident_str = '_'.join([args.eval_method, os.path.split(args.eg_name)[-1], args.eg_suff])
	else:
		raise AssertionError
	with open(args.predictions_path % method_ident_str, 'w', encoding='utf8') as fp:
		for t, u, s, l in zip(entry_tscores, entry_avg_uscores, final_scores, final_labels):
			fp.write(f"{t}\t{u}\t{s}\t{l}\n")

	with open(args.pr_rec_path % method_ident_str, 'w', encoding='utf8') as fp:
		fp.write(f"auc: {auc_value}\n")
		for p, r, t in zip(prec[1:], rec[1:], thres):
			fp.write(f"{p}\t{r}\t{t}\n")

	print(f"Finished!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval_set', type=str, default='dev')
	parser.add_argument('--version', type=str, default='15_30_triple_doc_disjoint_1400000_2_lexic_wordnet')
	parser.add_argument('--fpath_base', type=str,
						default='../../QAEval/clue_final_samples_%s_%s.json')
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
	parser.add_argument('--result_dir', type=str, default='../gfiles/qaeval_results/%s_%s/')
	parser.add_argument('--pr_rec_fn', type=str, default='%s_prt_vals.tsv')
	parser.add_argument('--predictions_fn', type=str, default='%s_predictions.txt')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--backupAvg', action='store_true')
	parser.add_argument('--ignore_0_for_Avg', action='store_true',
						help='whether or not to ignore the zero entailment scores for averages, or to take them in in the denominator.')
	parser.add_argument('--keep_same_rel_sents', action='store_true')
	parser.add_argument('--device_name', type=str, default='cpu')
	parser.add_argument('--min_graphsize', type=int, default=20480)

	# flags below are for the TF-IDF ranker.
	parser.add_argument('--tfidf_path', type=str,
						default='/Users/teddy/PycharmProjects/DrQA/scripts/retriever/clue_doc_db-tfidf-ngram=2-hash=16777216-tokenizer=spacy-chinese.npz')
	parser.add_argument('--articleIds_dict_path', type=str,
						default='/Users/teddy/PycharmProjects/DrQA/scripts/retriever/articleIds_by_partition.json')
	parser.add_argument('--num_refs_bert1', type=int, default=5,
						help='the number of reference articles to retrieve with TF-IDF for each query sentence.')

	# flags below are put here for the graph initializer, but generally they should not be changed.
	parser.add_argument('--saveMemory', action='store_true')
	parser.add_argument('--threshold', type=int, default=None)
	parser.add_argument('--maxRank', type=int, default=None)

	args = parser.parse_args()
	args.CCG = True
	assert args.eval_set in ['dev', 'test']
	assert args.slicing_method in ['disjoint', 'sliding']
	assert args.eval_mode in ['boolean', 'wh']
	assert args.eval_method in ['bert1', 'bert2', 'bert3', 'eg']

	args.fpath = args.fpath_base % (args.version, args.eval_set)
	args.eg_dir = os.path.join(args.eg_root, args.eg_name)
	# args.skip_idxes_fn = args.skip_idxes_fn % (args.version, args.eval_set)
	args.result_dir = args.result_dir % (args.version, args.eval_set)
	if not os.path.exists(args.result_dir):
		os.mkdir(args.result_dir)
	args.pr_rec_path = os.path.join(args.result_dir, args.pr_rec_fn)
	args.predictions_path = os.path.join(args.result_dir, args.predictions_fn)
	datemngr = DateManager()
	if args.slicing_method == 'disjoint':
		date_slices, _ = datemngr.setup_dateslices(args.time_interval)
	elif args.slicing_method == 'sliding':
		date_slices, _ = datemngr.setup_dates(args.time_interval)
	else:
		raise AssertionError

	print(args)

	if args.eval_mode in ['wh']:
		raise NotImplementedError
	elif args.eval_mode in ['boolean']:
		qa_eval_boolean_main(args, date_slices)

	print(f"Finished.")
