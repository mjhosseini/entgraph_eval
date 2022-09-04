import json
import time
import os
import argparse
import numpy as np
from scipy import stats


def reverse_12(typed_pred):
	tlist = typed_pred.split('#')
	t1 = tlist[-2]
	t2 = tlist[-1]
	if len(tlist) > 3:
		print(typed_pred)
	up = typed_pred.rstrip(t2).rstrip('#').rstrip(t1).rstrip('#')
	if t1.endswith('_2'):
		assert t2.endswith('_1')
		assert t1[:-2] == t2[:-2]
		typed_pred = '#'.join([up, t2, t1])
	elif t1.endswith('_1'):
		assert t2.endswith('_2')
		assert t1[:-2] == t2[:-2]
		typed_pred = '#'.join([up, t2, t1])
	else:
		pass
	return typed_pred


def calc_krackhardt_score(args):
	files = os.listdir(args.graph_dir)
	typewise_kscores = {}  # tuples: (k_score, num_edges)
	files = sorted(files)
	total_t1_error_count = 0
	total_t2_error_count = 0

	buckets = {'st1e-6': 0, '1e-6to1e-5': 0, '1e-5to1e-4': 0, '1e-4to1e-3': 0, '1e-3to1e-2': 0, '1e-2to0.1': 0,
			   '0.1to0.2': 0, '0.2to0.4': 0, '0.4to0.6': 0, '0.6to0.8': 0, '0.8to1': 0, '1': 0}
	all_scores = []
	perfile_stats = []

	for f in files:
		numerator = 0.0
		denominator = 0.0
		edge_count = 0.0
		this_t1_error_count = 0
		this_t2_error_count = 0
		cur_entscores = {}
		if not f.endswith(args.suffix):
			continue
		argtype = f.rstrip(args.suffix)
		cur_fp = open(os.path.join(args.graph_dir, f), 'r', encoding='utf8')
		header = cur_fp.readline()
		cur_prem_pred = None
		is_hyps_line = False
		for line in cur_fp:
			line = line.rstrip('\n')
			if line.startswith('predicate: '):
				assert is_hyps_line is False
				cur_prem_pred = line[11:]
				assert cur_prem_pred not in cur_entscores
				cur_entscores[cur_prem_pred] = {}
			elif line.startswith('max num neighbors: ') or line.startswith('num neighbors: '):
				assert is_hyps_line is False
				continue
			elif len(line) == 0:
				is_hyps_line = False
				continue
			elif line in ['contextualized sims', 'BInc sims', 'global sims']:
				assert is_hyps_line is False
				is_hyps_line = True
				continue
			else:
				if not is_hyps_line:
					continue
				cur_score = line.split(' ')[-1]
				hyp_pred = line.rstrip(cur_score).rstrip()
				cur_score = float(cur_score)
				assert hyp_pred not in cur_entscores[cur_prem_pred]
				cur_entscores[cur_prem_pred][hyp_pred] = cur_score

		cur_scores = []
		for prem_pred in cur_entscores:
			for hyp_pred in cur_entscores[prem_pred]:
				rev_prem_pred = reverse_12(prem_pred)
				rev_hyp_pred = reverse_12(hyp_pred)
				cur_scores.append(cur_entscores[prem_pred][hyp_pred])

				if hyp_pred not in cur_entscores and rev_hyp_pred not in cur_entscores:
					this_t1_error_count += 1
					numerator += cur_entscores[prem_pred][hyp_pred]
				else:
					if hyp_pred in cur_entscores and prem_pred in cur_entscores[hyp_pred]:
						numerator += max(cur_entscores[prem_pred][hyp_pred] - cur_entscores[hyp_pred][prem_pred], 0)
					elif rev_hyp_pred in cur_entscores and rev_prem_pred in cur_entscores[rev_hyp_pred]:
						numerator += max(cur_entscores[prem_pred][hyp_pred] - cur_entscores[rev_hyp_pred][rev_prem_pred], 0)
					else:
						if args.debug:
							print(f"prem_pred: {prem_pred}; hyp_pred: {hyp_pred}; rev_prem_pred: {rev_prem_pred}; rev_hyp_pred: {rev_hyp_pred}")
							time.sleep(1)
						this_t2_error_count += 1
						numerator += cur_entscores[prem_pred][hyp_pred]
				denominator += cur_entscores[prem_pred][hyp_pred]
				edge_count += 1

		all_scores += cur_scores
		cur_scores = np.array(cur_scores)
		cur_avgscore = np.mean(cur_scores)
		cur_sigma = np.std(cur_scores)
		cur_skew = stats.skew(cur_scores)
		cur_kurtosis = stats.kurtosis(cur_scores)
		perfile_stats.append((argtype, len(cur_scores), cur_avgscore, cur_sigma, cur_skew, cur_kurtosis))
		print(f"Stats for type pair \"{argtype}\": total entries - {len(cur_scores)}; mean - {cur_avgscore}; std - {cur_sigma}; skewness - {cur_skew}; kurtosis - {cur_kurtosis};")

		this_kscore = numerator / denominator if denominator != 0 else 0
		typewise_kscores[argtype] = (this_kscore, edge_count)

		print(f"Krackhardt hierarchy score for type pair \"{argtype}\": {this_kscore}; edge count: {edge_count}; this t1 err count: {this_t1_error_count}; this t2 err count: {this_t2_error_count}")
		total_t1_error_count += this_t1_error_count
		total_t2_error_count += this_t2_error_count
		cur_fp.close()
	all_scores.sort()

	print(f"Overall median: {all_scores[len(all_scores)//2]}")

	all_scores = np.array(all_scores)
	total_avgscore = np.mean(all_scores)
	total_sigma = np.std(all_scores)
	total_skew = stats.skew(all_scores)
	total_kurtosis = stats.kurtosis(all_scores)
	print("")
	print(f"Overall stats: total entries - {len(all_scores)}; mean - {total_avgscore}; std - {total_sigma}; skewness - {total_skew}; kurtosis - {total_kurtosis};")

	for scr in all_scores:
		if scr < 0.000001:
			buckets['st1e-6'] += 1
		elif scr < 0.00001:
			buckets['1e-6to1e-5'] += 1
		elif scr < 0.0001:
			buckets['1e-5to1e-4'] += 1
		elif scr < 0.001:
			buckets['1e-4to1e-3'] += 1
		elif scr < 0.01:
			buckets['1e-3to1e-2'] += 1
		elif scr < 0.1:
			buckets['1e-2to0.1'] += 1
		elif scr < 0.2:
			buckets['0.1to0.2'] += 1
		elif scr < 0.4:
			buckets['0.2to0.4'] += 1
		elif scr < 0.6:
			buckets['0.4to0.6'] += 1
		elif scr < 0.8:
			buckets['0.6to0.8'] += 1
		elif scr < 1:
			buckets['0.8to1'] += 1
		else:
			buckets['1'] += 1

	print("Bucket stats: ")
	for key in buckets:
		print(f"{key}: {buckets[key]}")

	kscore_sum = 0.0
	edge_count_sum = 0.0
	for argtype in typewise_kscores:
		kscore_sum += typewise_kscores[argtype][0] * typewise_kscores[argtype][1]
		edge_count_sum += typewise_kscores[argtype][1]
	avg_kscore = kscore_sum / edge_count_sum
	print("")
	print(f"Overall Krackhardt hierarchy score for graph {args.graph_dir}: {avg_kscore}; total edge count: {edge_count_sum}")
	print(f"Total t1 err percentage: {total_t1_error_count/edge_count_sum}; total t2 err percentage: {total_t2_error_count/edge_count_sum}!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--graph_dir', default='', type=str)
	parser.add_argument('--suffix', default='_sim.txt', type=str)
	parser.add_argument('--debug', action='store_true')

	args = parser.parse_args()
	calc_krackhardt_score(args)


