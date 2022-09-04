import xml.etree.ElementTree as ET
import copy
import random
import json
import sys
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


# TODO: this file is copied from NE_Pipeline.

# predicates whose tokens are all among these tokens are considered vague predicates and skipped.
vague_predtoks = ['是', '还是', '也是', '都是', '有', '的', 'X']


def check_vague(pred_bow):
	is_vague = True
	for pred_tok in pred_bow:
		if pred_tok not in vague_predtoks:
			is_vague = False
	return is_vague


def parse_time(time_str):
	date, time = time_str.rstrip('\n').split(' ')
	month, day = date.split('-')
	hr, mnt = time.split(':')
	return {'month': month, 'day': day, 'hour': hr, 'minute': mnt}


class DateManager:
	def __init__(self, lang):
		self.month2numdays = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31,
							  11: 30, 12: 31}
		self.monthstr = {x: f"0{x}" for x in range(1, 10)}
		for x in range(10, 13):
			self.monthstr[x] = str(x)
		self.daysstr = {x: f"0{x}" for x in range(1, 10)}
		for x in range(10, 32):
			self.daysstr[x] = str(x)

		self.lang = lang
		if lang == 'zh':
			self.years_list = ['']
		elif lang == 'en':
			self.years_list = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
		else:
			raise AssertionError

	def date2str(self, cur_date, time_interval=None):  # each cur_date: (month, day, year)
		if time_interval is None:
			return f"{self.monthstr[cur_date[0]]}-{self.daysstr[cur_date[1]]}"
		else:
			return f"{self.monthstr[cur_date[0]]}-{self.daysstr[cur_date[1]]}_{time_interval}"

	# returns disjoint date slices spanning the whole year, each worth ``interval_length'' number of days
	# (except the last one, which ends at December 31st regardless of its size)
	def setup_dateslices(self, interval_length):
		all_slices = {}

		def all_dates(start_date, end_date, year):
			res = []
			if start_date[0] == end_date[0]:
				cur_day = start_date[1]
				while cur_day <= end_date[1]:
					res.append((start_date[0], cur_day, year))
					cur_day += 1
			else:
				assert start_date[0] + 1 == end_date[0]
				assert start_date[1] <= self.month2numdays[start_date[0]]
				cur_day = start_date[1]
				while cur_day <= self.month2numdays[start_date[0]]:
					res.append((start_date[0], cur_day, year))
					cur_day += 1
				cur_day = 1
				assert end_date[1] <= self.month2numdays[end_date[0]]
				while cur_day <= end_date[1]:
					res.append((end_date[0], cur_day, year))
					cur_day += 1
			return res

		for year in self.years_list:
			start_date = (1, 1)
			while start_date[0] <= 12:
				last_month = start_date[0]
				last_day = start_date[1] + (interval_length - 1)
				if last_day > self.month2numdays[start_date[0]]:
					if last_month == 12:
						last_day = 31
					else:
						last_month += 1
						last_day -= self.month2numdays[start_date[0]]
				next_start_month = last_month
				next_start_day = last_day + 1
				if next_start_day > self.month2numdays[last_month]:
					assert next_start_day - self.month2numdays[last_month] == 1
					next_start_month += 1
					next_start_day = 1
				end_date = (last_month, last_day)
				if len(year) > 0:
					cur_slice_name = f"{year}_{self.date2str(start_date)}_{self.date2str(end_date)}"
				else:
					assert self.lang == 'zh'
					cur_slice_name = f"{self.date2str(start_date)}_{self.date2str(end_date)}"
				all_slices[cur_slice_name] = all_dates(start_date, end_date, year)
				start_date = (next_start_month, next_start_day)
		if self.lang == 'zh':
			pass
		elif self.lang == 'en':
			all_slices[f"0000_01-01_01-03"] = [(1, 1, '0000'), (1, 2, '0000'), (1, 3, '0000')]  # a special date for those entries whose date info is corrupted.
		else:
			raise AssertionError

		date2slice = {}
		for slc in all_slices:
			for d in all_slices[slc]:
				d = f"{d[2]}-{self.monthstr[d[0]]}-{self.daysstr[d[1]]}"
				assert d not in date2slice
				date2slice[d] = slc
		slice_keys = list(all_slices.keys())

		return slice_keys, date2slice

	def setup_dates(self, time_interval):
		all_dates = []
		for month in self.month2numdays:
			for day in range(1, self.month2numdays[month] + 1):
				date_str = self.date2str((month, day), time_interval)
				all_dates.append(date_str)
		return all_dates, None

	def get_next_date(self, cur_date_str):
		cur_date_str, interval = cur_date_str.split('_')
		interval = int(interval)
		month, day = cur_date_str.split('-')
		month = int(month)
		day = int(day)
		next_month = month
		next_day = day + 1
		if next_day <= self.month2numdays[month]:
			pass
		elif next_day - 1 == self.month2numdays[month]:
			if month + 1 not in self.month2numdays:
				assert month == 12
				return None
			else:
				next_month += 1
				next_day = 1
		else:
			raise AssertionError
		next_date_str = self.date2str((next_month, next_day), interval)
		return next_date_str

	def get_prev_date(self, cur_date_str):
		cur_date_str, interval = cur_date_str.split('_')
		interval = int(interval)
		month, day = cur_date_str.split('-')
		month = int(month)
		day = int(day)
		prev_month = month
		prev_day = day - 1
		if prev_day > 0:
			pass
		elif prev_day == 0:
			if month == 1:
				return None
			else:
				assert month > 1
				prev_month -= 1
				prev_day = self.month2numdays[prev_month]
		prev_date_str = self.date2str((prev_month, prev_day), interval)
		return prev_date_str


def parse_rel(rel):
	rel = rel["r"]
	assert rel[0] == '(' and rel[-1] == ')'
	rel = rel[1:-1]
	rel_list = rel.split('::')
	assert len(rel_list) == 8
	upred = rel_list[0]
	subj = rel_list[1]
	obj = rel_list[2]
	tsubj = rel_list[6]
	tobj = rel_list[7]
	return upred, subj, obj, tsubj, tobj


def assemble_rel(new_upred, old_rel):
	old_rel = old_rel["r"]
	assert old_rel[0] == '(' and old_rel[-1] == ')'
	old_rel = old_rel[1:-1]
	old_rel_list = old_rel.split('::')
	new_rel_list = copy.copy(old_rel_list)
	new_rel_list[0] = new_upred
	new_rel = '::'.join(new_rel_list)
	new_rel = '(' + new_rel + ')'
	return new_rel


def rel2normalform(old_rel):
	old_rel = old_rel["r"]
	assert old_rel[0] == '(' and old_rel[-1] == ')'
	old_rel = old_rel[1:-1]
	old_rel_list = old_rel.split('::')
	new_rel_list = copy.copy(old_rel_list)
	new_rel_list[5] = '0'
	new_rel = '::'.join(new_rel_list)
	new_rel = '(' + new_rel + ')'
	return new_rel


def split_str_multipat(string, patterns):
	lst = [string]
	new_lst = []
	for pat in patterns:
		for ele in lst:
			new_lst += ele.split(pat)
		lst = copy.deepcopy(new_lst)
		new_lst = []
	return lst


def simple_lemmatize(string):
	return string.lower().replace('_', ' ')


# use cases: 1) in select_positives: prefix doesn't matter; 2) in generate_negatives: prefix matters; 3) qaeval_utils: prefix matters.
def upred2bow(upred, lang):
	splitpred_toks = ['.', '-', '_']

	orig_upred = copy.copy(upred)
	if lang == 'zh':
		assert upred[0] == '(' and upred[-1] == ')'
		upred = upred[1:-1]
		upred_1, upred_2 = upred.split('.1,')
		assert upred_2[-2:] == '.2'
		upred_2 = upred_2[:-2]
		assert upred_1 == upred_2
		upred_list = upred_1.split('·')
	elif lang == 'en':
		try:
			upred_prefix, upred = upred.split('(')
		except ValueError as e:
			print(f"prefix ( not found: {e}; orig upred: {orig_upred}")
			upred_prefix = ''
		try:
			upred, upred_suffix = upred.split(')')
		except ValueError as e:
			print(f"suffix ) not found: {e}; orig upred: {orig_upred}")
			upred_suffix = ''
		assert len(upred_suffix) == 0
		upred = [upred]
		for splt_tok in ['.1,', '.2,', '.3,', '.4,']:
			new_upred = []
			for upt in upred:
				new_upred += upt.split(splt_tok)
			upred = copy.deepcopy(new_upred)

		if len(upred) == 1:
			upred_str = upred[0].lstrip('1,').lstrip('2,').lstrip('3,').lstrip('4,').rstrip(',1').rstrip(',2') \
				.rstrip(',3').rstrip(',4').rstrip('.1').rstrip('.2').rstrip('.3').rstrip('.4')
			upred_list = split_str_multipat(upred_str, splitpred_toks)
			upred_list = [x for x in upred_list if len(x) > 0]
		elif len(upred) == 2:
			upred_str = upred[1].rstrip('.1').rstrip('.2').rstrip('.3').rstrip('.4')
			upred_list = split_str_multipat(upred_str, splitpred_toks)
		else:
			upred_list = []
			for upt in upred:
				upt = upt.rstrip('.1').rstrip('.2').rstrip('.3').rstrip('.4')
				upt_list = split_str_multipat(upt, splitpred_toks)
				upt_addin_list = []
				for tok in upt_list:
					if tok not in upred_list:
						upt_addin_list.append(tok)
				upred_list += upt_addin_list
		upred_list = [x.lower() for x in upred_list]

		if len(upred_prefix) < 2:
			if upred_prefix not in ['', '_']:
				print(f"prefix anomaly: {upred_prefix};", file=sys.stderr)
			pass
		elif upred_prefix == 'NEG__':
			if len(upred_list) > 0 and upred_list[0] in ['is', 'are', 'would', 'should', 'must', 'have']:
				upred_list = [upred_list[0], 'not'] + upred_list[1:]
			else:
				upred_list = ['do', 'not'] + upred_list
		else:
			assert upred_prefix[-2:] == '__'
			upred_prefix = split_str_multipat(upred_prefix[:-2], splitpred_toks)
			if len(upred_prefix) == 1 and len(upred_list) > 0:
				upred_prefix += ['to']
			else:
				pass
			upred_prefix = [x.lower() for x in upred_prefix]

			if len(upred_list) > 0 and upred_prefix[-1] == upred_list[0]:
				upred_list = upred_prefix + upred_list[1:]
			else:
				upred_list = upred_prefix + upred_list
	else:
		raise AssertionError

	upred_list = [x.lower() for x in upred_list]
	if len(upred) == 1:
		# print(f"upred {orig_upred} len == 1; upred_list: {upred_list}!", file=sys.stderr)
		pass
	return upred_list


def rel2concise_str(upred, subj, obj, tsubj, tobj, lang):
	upred_list = upred2bow(upred, lang=lang)
	if lang == 'zh':
		upred_dumm_str = ''.join(upred_list)
	elif lang == 'en':
		upred_dumm_str = ' '.join(upred_list)
	else:
		raise AssertionError
	rel_str = f"{upred_dumm_str}::{subj}::{obj}::{tsubj}::{tobj}"
	return rel_str, upred_dumm_str


def readWordNet(wordnet_dir):
	tree = ET.parse(wordnet_dir)
	root = tree.getroot()
	lexicon = root.find('Lexicon')
	senseAxes = root.find('SenseAxes')
	lexicalEntries = {}
	sstids2lemmas = {}
	synsets = {}
	zh_en_alignments = []

	multiple_entry_count = 0

	for lexicalEntry in lexicon.findall('LexicalEntry'):
		ent_id = lexicalEntry.get('id')
		lemma_xml = lexicalEntry.find('Lemma')
		ent_written_form = lemma_xml.get('writtenForm')
		# assert len(ent_written_form.split('+')) <= 2
		out_written_form = ''.join(ent_written_form.split('+'))
		if len(ent_written_form.split('+')) > 1:
			if not (len(ent_written_form.split('+')) == 2 and ent_written_form.split('+')[1] in ['的', '地', '得']):
				print(ent_written_form)

		ent_pos = lemma_xml.get('partOfSpeech')
		ent_senses = []
		for sense in lexicalEntry.findall('Sense'):
			sense_syn = sense.get('synset')
			ent_senses.append(sense_syn)
			if sense_syn not in sstids2lemmas:
				sstids2lemmas[sense_syn] = []
			sstids2lemmas[sense_syn].append(out_written_form)

		lex_entry = {'id': ent_id, 'pos': ent_pos, 'senses': ent_senses}
		if out_written_form not in lexicalEntries:
			lexicalEntries[out_written_form] = []
		else:
			multiple_entry_count += 1
		lexicalEntries[out_written_form].append(lex_entry)

	print(f"Multiple entry count: {multiple_entry_count}!")

	all_reltypes = set()
	for synset in lexicon.findall('Synset'):
		syn_id = synset.get('id')
		syn_base_concept = synset.get('baseConcept')
		syn_rels = {}
		synset_rels = synset.find('SynsetRelations')
		for rel in synset_rels.findall('SynsetRelation'):
			reltype = rel.get('relType')
			reltgt = rel.get('targets')
			all_reltypes.add(reltype)
			if reltype not in syn_rels:
				syn_rels[reltype] = []
			syn_rels[reltype].append(reltgt)
		syn_entry = {'id': syn_id, 'baseConcept': syn_base_concept, 'rels': syn_rels}
		assert syn_id not in synsets
		synsets[syn_id] = syn_entry

	# print(f"all_reltypes: {all_reltypes}")

	for senseAxis in senseAxes.findall('SenseAxis'):
		axis_id = senseAxis.get('id')
		axis_type = senseAxis.get('relType')

		zh_tgt, en_tgt = senseAxis.findall('Target')
		zh_tgt_id = zh_tgt.get('ID')
		en_tgt_id = en_tgt.get('ID')
		axis_entry = {'id': axis_id, 'type': axis_type, 'zh_tgt_id': zh_tgt_id, 'en_tgt_id': en_tgt_id}
		zh_en_alignments.append(axis_entry)

	wordnet_dict = {'lexicalEntries': lexicalEntries, 'synsets': synsets, 'sstids2lemmas': sstids2lemmas, 'zh_en_alignments': zh_en_alignments}

	return wordnet_dict


def readWord2Vec(word2vec_path, verbose=True):
	word_vectors = {}
	redundant_words = []
	with open(word2vec_path, 'r', encoding='utf8') as input_fp:
		meta_info = input_fp.readline()
		num_words, dim = meta_info.split(' ')
		num_words = int(num_words)
		print(f"Total number of words: {num_words};")
		dim = int(dim)
		for lidx, line in enumerate(input_fp):
			if lidx % 10000 == 0 and verbose:
				print(f"lidx: {lidx}; number of redundant words: {len(redundant_words)}")
			line = line.rstrip('\n').split(' ')
			cur_word = line[0]
			cur_vecs = np.array([float(x) for x in line[1:] if len(x) > 0])
			assert cur_vecs.shape[0] == dim
			if cur_word in word_vectors:
				redundant_words.append(cur_word)
			word_vectors[cur_word] = cur_vecs
		assert len(word_vectors) + len(redundant_words) == num_words
		print(f"Total number of redundant words: {len(redundant_words)}")
		if verbose:
			print(redundant_words)
	return word_vectors


def build_upred_vectors_h5py(all_preds, word_vectors, h5file, pred_set_path):
	stopwords = ['X', '的', '【介宾】']
	all_upredstrs_global_ids = {}
	for type_pair in all_preds:
		for predstr in all_preds[type_pair]:
			if predstr not in all_upredstrs_global_ids:
				all_upredstrs_global_ids[predstr] = None

	# since the storage is not our biggest problem, we'll not merge preds with same surface forms but different argument types.
	num_all_upreds = len(all_upredstrs_global_ids)
	num_all_tpreds = sum([len(all_preds[x]) for x in all_preds])
	dim_vectors = None
	for k in word_vectors:  # this loop will only execute for once.
		dim_vectors = word_vectors[k].shape[0]
		break
	print(f"num_all_upreds: {num_all_upreds}")
	print(f"num_all_tpreds: {num_all_tpreds}")
	dset = h5file.create_dataset("upred_vecs", (num_all_upreds, dim_vectors,), dtype='f')
	mismatched_pred_instance_count = 0
	mismatched_predstrs = set()

	global_id_cursor = 0
	type_pairs_list = sorted(list(all_preds.keys()))
	for type_pair in type_pairs_list:
		print(f"Recording vector representation of predicates with type {type_pair}:")
		for pred_idx, upred_str in enumerate(all_preds[type_pair]):
			if pred_idx % 100000 == 0 and pred_idx > 0:
				print(f"pred_idx: {pred_idx}; cursor position: {global_id_cursor}")

			# if the upred has already been cached with other argument type-pairs,
			# then just take that previous cache and move on to the next predicate.
			if all_upredstrs_global_ids[upred_str] is not None:
				global_pred_idx = all_upredstrs_global_ids[upred_str]
				all_preds[type_pair][upred_str]['global_id'] = global_pred_idx
				if upred_str in mismatched_predstrs:
					mismatched_pred_instance_count += 1
			else:
				global_pred_idx = global_id_cursor
				global_id_cursor += 1
				all_upredstrs_global_ids[upred_str] = global_pred_idx
				all_preds[type_pair][upred_str]['global_id'] = global_pred_idx

				upred = all_preds[type_pair][upred_str]['p']
				upred_list = upred2bow(upred)
				upred_vecs_list = []
				mismatch_found = False
				for utok in upred_list:
					if utok in stopwords:
						continue
					if utok not in word_vectors:
						mismatch_found = True
						break
					upred_vecs_list.append(word_vectors[utok])
				if mismatch_found or len(upred_vecs_list) == 0:
					mismatched_pred_instance_count += 1
					mismatched_predstrs.add(upred_str)
					dset[global_pred_idx, :] = np.zeros((dim_vectors), dtype=np.float32)
				else:
					upred_vec = np.mean(upred_vecs_list, axis=0)
					dset[global_pred_idx, :] = upred_vec

	assert global_id_cursor == num_all_upreds
	print(f"Overall mismatched predicate instances count: {mismatched_pred_instance_count};")
	mismatched_predstrs = [x.encode('utf-8') for x in mismatched_predstrs]
	mismatches_dset = h5file.create_dataset("mismatched_predstrs", data=mismatched_predstrs)

	print(f"Dumping updated all_preds back into {pred_set_path}")
	with open(pred_set_path, 'w', encoding='utf8') as fp:
		for type_pair in all_preds:
			for key in all_preds[type_pair]:
				out_line = json.dumps({'type': type_pair, 'predstr': key, 'p': all_preds[type_pair][key]['p'],
									   'num_occ': all_preds[type_pair][key]['num_occ'], 'global_id': all_preds[type_pair][key]['global_id']}, ensure_ascii=False)
				fp.write(out_line+'\n')
	return dset, mismatches_dset


def calc_simscore(vec1, vec2):
	return cosine_similarity(vec1, vec2)


def truncate_merge_by_ratio(cur_negatives_ext, cur_negatives_nonext, first_ratio, second_ratio):
	assert 0 < first_ratio < 1 and first_ratio + second_ratio == 1
	combined_negatives = None
	# the first case indicates that the ext list is larger than desired proportion, truncate the ext
	if len(cur_negatives_ext) / first_ratio > len(cur_negatives_nonext) / second_ratio:
		desired_ext_size = int(len(cur_negatives_nonext) / second_ratio * first_ratio)
		ext_samples = random.sample(cur_negatives_ext, desired_ext_size)
		combined_negatives = ext_samples + cur_negatives_nonext
	elif len(cur_negatives_ext) / first_ratio < len(cur_negatives_nonext) / second_ratio:
		desired_nonext_size = int(len(cur_negatives_ext) / first_ratio * second_ratio)
		nonext_samples = random.sample(cur_negatives_nonext, desired_nonext_size)
		combined_negatives = nonext_samples + cur_negatives_ext
	else:
		combined_negatives = cur_negatives_ext + cur_negatives_nonext

	assert combined_negatives is not None
	return combined_negatives


def filter_sound_triples(all_triples, global_presence_thres):
	sound_tplstrs = set()

	for tplstr in all_triples:
		if all_triples[tplstr]['num_occ'] < global_presence_thres:
			continue
		assert tplstr not in sound_tplstrs
		sound_tplstrs.add(tplstr)
	return sound_tplstrs


def duration_format_print(dur, heading=''):
	dur_h = int(dur / 3600)
	dur_m = int((int(dur) % 3600) / 60)
	dur_s = (dur % 60)
	print(f"{heading}: time lapsed: {dur_h} hours {dur_m} minutes %.3f seconds;" % (dur_s))


# TODO: the function below is obsolete because it takes too much memory, the newer version gives up maintaining a dict
# TODO: of all possible readings, instead just maintain the present one, randomly chooses to swap or not when seeing a
# TODO: different reading.
# def load_triple_set(triples_path, triple_set_path=None, pred_set_path=None, max_num_lines=-1, verbose=True):
# 	if verbose:
# 		print(f"Loading triples from {triples_path}")
# 	all_triples = {}
# 	all_preds = {}
# 	ambiguity_count = 0
# 	with open(triples_path, 'r', encoding='utf8') as fp:
# 		for lidx, line in enumerate(fp):
# 			if 0 <= max_num_lines < lidx:
# 				break
# 			if lidx % 100000 == 0 and verbose:
# 				print(f"lidx: {lidx}; ambiguity_count: {ambiguity_count}")
# 			item = json.loads(line)
# 			for rel in item["rels"]:
# 				upred, subj, obj, tsubj, tobj = parse_rel(rel)
# 				triple_str, pred_str = rel2concise_str(upred, subj, obj, tsubj, tobj)
# 				rel_normalized = rel2normalform(rel)
# 				if triple_str not in all_triples:
# 					all_triples[triple_str] = [[], []]
# 				try:
# 					rel_idx = all_triples[triple_str][0].index(rel_normalized)
# 					all_triples[triple_str][1][rel_idx] += 1
# 				except ValueError as e:
# 					all_triples[triple_str][0].append(rel_normalized)
# 					all_triples[triple_str][1].append(1)
#
# 				if pred_str not in all_preds:
# 					all_preds[pred_str] = [[], []]
# 				try:
# 					upidx = all_preds[pred_str][0].index(upred)
# 					all_preds[pred_str][1][upidx] += 1
# 				except ValueError as e:
# 					if len(all_preds[pred_str][0]) > 0:
# 						uuuu = all_preds[pred_str]
# 						ambiguity_count += 1
# 					all_preds[pred_str][0].append(upred)
# 					all_preds[pred_str][1].append(1)
#
# 	if verbose:
# 		print(f"Ambiguity count: {ambiguity_count}")
# 	for tplidx, k in enumerate(all_triples):
# 		if tplidx % 100000 == 0:
# 			print(f"tplidx: {tplidx}")
# 		sum_instances = float(sum(all_triples[k][1]))
# 		all_triples[k][1] = [x/sum_instances for x in all_triples[k][1]]
# 		all_triples[k] = np.random.choice(all_triples[k][0], p=all_triples[k][1])
#
# 	print(f"Dumping triples...")
# 	if triple_set_path is not None:
# 		with open(triple_set_path, 'w', encoding='utf8') as fp:
# 			for kidx, k in enumerate(all_triples):
# 				if kidx % 1000000 == 0 and kidx > 0:
# 					print(f"line number {kidx}")
# 				out_line = json.dumps({k: all_triples[k]}, ensure_ascii=False)
# 				fp.write(out_line+'\n')
#
# 	if verbose:
# 		print(f"Number of unique concise triple sets: {len(all_triples)}")
#
# 	for pidx, k in enumerate(all_preds):
# 		if pidx % 100000 == 0:
# 			print(f"pidx: {pidx}")
# 		sum_instances = float(sum(all_preds[k][1]))
# 		all_preds[k][1] = [x/sum_instances for x in all_preds[k][1]]
# 		all_preds[k] = np.random.choice(all_preds[k][0], p=all_preds[k][1])
#
# 	print(f"Dumping predicates...")
# 	if pred_set_path is not None:
# 		with open(pred_set_path, 'w', encoding='utf8') as fp:
# 			for kidx, k in enumerate(all_preds):
# 				if kidx % 1000000 == 0 and kidx > 0:
# 					print(f"line number {kidx}")
# 				out_line = json.dumps({k: all_preds[k]}, ensure_ascii=False)
# 				fp.write(out_line+'\n')
#
# 	return all_triples, all_preds


def read_all_preds_set(predset_path):
	all_preds = dict()
	with open(predset_path, 'r', encoding='utf8') as ifp:
		for line in ifp:
			item = json.loads(line)
			assert item['type'] is not None
			if item['type'] not in all_preds:
				all_preds[item['type']] = dict()

			assert item['predstr'] not in all_preds[item['type']]
			all_preds[item['type']][item['predstr']] = {'p': item['p'], 'n': item['n'],
														'predstr': item['predstr'],
														'lemm_predstr': item['lemm_predstr'],
														'lemm_noprep_predstr': item['lemm_noprep_predstr']}
	return all_preds


def all_toks_are_preps(upred_list, start_id, end_id):
	PREPOSITION_POS_TAGS = ['IN', 'TO', 'RP']  # here the particle POS TAG ``RP'' is kept to avoid allowing in any noise.

	pred_str = ' '.join(upred_list)
	tokens = nltk.word_tokenize(pred_str)
	tagged = nltk.pos_tag(tokens)
	if len(tagged) != len(upred_list):
		print(f"all_toks_are_preps: Warning! `tagged' and `upred_list' are not of the same length!")
		print(f"upred_list: {upred_list}")
		print(f"tagged: {tagged}")
		return False
	res_flag = True
	for tok, postag in tagged[start_id:end_id]:
		if postag not in PREPOSITION_POS_TAGS:
			res_flag = False
			break
	return res_flag


# Update: if the query key contains banned words, return empty list directly; if the retrieved hlems contain banned words, skip them.
def fetch_wn_smoothings(upred_list: list, direction: str, gr_predstr2preds: dict, first_only: bool):
	assert direction in ['hyponym', 'hypernym']

	if check_vague(upred_list):
		return [[[]], [[]]]

	all_banned_words = ['be']
	hypo_banned_words = ['get', 'give', 'take', 'have', 'see', 'meet', 'make', 'say', 'set', 'move', 'be']
	for tok in upred_list:
		if tok in all_banned_words:
			return [[[]], [[]]]

	replacement_found_predstr_spans = []

	all_pred_replacements = []

	for start_id in range(len(upred_list)):
		for end_id in range(len(upred_list), start_id, -1):
			# Enumerate the spans:
			# 		By enumerating continuous spans, naturally parts of the predicates split by 'X'
			# 		will not be taken as consecutive strings.

			#		Those sub-spans of those already associated-with-negatives spans should not be attended anymore.
			#		This is enforced by maintaining a list of negfound_spans, then continueing when a span is a subspan of any of those.

			subspan_of_good_spans_flag = False  # if this flag is true, then there is no point in attending to the current span!
			for negfound_span in replacement_found_predstr_spans:
				if start_id >= negfound_span['start_id'] and end_id <= negfound_span['end_id']:
					subspan_of_good_spans_flag = True
					break
			if subspan_of_good_spans_flag:
				continue

			potential_replacements = set()
			# if all tokens are prepositions, then there is no point in replacing it.
			if not all_toks_are_preps(upred_list, start_id, end_id):
				span_to_replace = '_'.join(upred_list[start_id:end_id])

				syns = wordnet.synsets(span_to_replace)

				for syn in syns:
					if direction == 'hyponym':
						span_replacements = syn.hyponyms() + syn.instance_hyponyms()
					elif direction == 'hypernym':
						span_replacements = syn.hypernyms() + syn.instance_hypernyms()
					else:
						raise AssertionError

					# span_causes = syn.causes()
					# if len(span_causes) > 0:
					# 	print(f"posi_synset: {syn}; span_causes: {span_causes}")
					for hsyn in span_replacements:
						hypo_lemm_overlap_with_posi_flag = False
						cur_hsyn_all_predstrs = []
						h_lemmas = hsyn.lemma_names()
						for hlem in h_lemmas:
							if hlem == span_to_replace:
								hypo_lemm_overlap_with_posi_flag = True
								break
							hlem = hlem.replace('_', ' ')

							hlem_contains_banned_flag = False
							for hlem_tok in hlem.split(' '):
								if hlem_tok in all_banned_words:
									hlem_contains_banned_flag = True
									break
								else:
									pass
							# hypo hlem can contain the hypo_banned_words, but cannot be exactly them.
							if direction == 'hyponym' and hlem in hypo_banned_words:
								hlem_contains_banned_flag = True

							if hlem_contains_banned_flag is True:
								continue
							else:
								assert hlem_contains_banned_flag is False
							cur_hyponym = ' '.join(upred_list[:start_id] + [hlem] + upred_list[end_id:])
							cur_hsyn_all_predstrs.append(cur_hyponym)

						# all lemmas in the hypothesis synset must not be the same as the original span to replace.
						if hypo_lemm_overlap_with_posi_flag:
							continue
						else:
							potential_replacements.update(cur_hsyn_all_predstrs)
					if first_only:
						break

			pred_replacements = []
			for potrep in potential_replacements:
				if potrep in gr_predstr2preds:
					pred_replacements += gr_predstr2preds[potrep]

			if len(pred_replacements) > 0:
				replacement_found_predstr_spans.append({'start_id': start_id, 'end_id': end_id})
			all_pred_replacements += pred_replacements

	return [[all_pred_replacements], [[1.0]*len(all_pred_replacements)]]


def get_predstr2preds(_gr):
	upredstr2tpreds = {}
	for tpred in _gr.pred2Node:
		try:
			upred, tsubj, tobj = tpred.split('#')
		except Exception as e:
			print(f"Exception in get_predstr2preds (within qaeval_utils.py)!")
			print(f"Error message: {e};")
			continue
		_, upredstr = rel2concise_str(upred, '', '', tsubj, tobj, lang='en')
		if upredstr not in upredstr2tpreds:
			upredstr2tpreds[upredstr] = []
		upredstr2tpreds[upredstr].append(tpred)
	return upredstr2tpreds

