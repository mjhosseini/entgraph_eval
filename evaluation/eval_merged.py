import sys
sys.path.append("..")
import evaluation.util_chinese
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import argparse
import numpy as np
import sys


def en_zh_dictorder(en_preds, zh_preds, zh_factor=1.):
	assert len(en_preds) == len(zh_preds)
	merged_preds = []
	for pid, (en_p, zh_p) in enumerate(zip(en_preds, zh_preds)):
		if en_p > 0:
			merged_preds.append(en_p)
		else:
			merged_preds.append(zh_factor*zh_p)
	assert len(merged_preds) == len(en_preds)
	return merged_preds


def zh_en_dictorder(en_preds, zh_preds, zh_factor=1., en_lemma=True):
	assert len(en_preds) == len(zh_preds)
	merged_preds = []
	for pid, (en_p, zh_p) in enumerate(zip(en_preds, zh_preds)):
		if en_p == 1 and en_lemma:
			merged_preds.append(en_p)
		elif zh_p > 0:
			merged_preds.append(zh_factor*zh_p)
		else:
			merged_preds.append(en_p)
	assert len(merged_preds) == len(en_preds)
	return merged_preds


def max_reduce(en_preds, zh_preds, zh_factor=1., en_lemma=True):
	assert len(en_preds) == len(zh_preds)
	merged_preds = []
	max_is_zh_count = 0  # the count of the max being *EXCLUSIVELY* from Chinese EntGraph
	for pid, (en_p, zh_p) in enumerate(zip(en_preds, zh_preds)):
		if en_p == 1 and en_lemma:
			merged_preds.append(en_p)
		else:
			new_p = max(en_p, zh_factor*zh_p)
			if new_p == zh_factor*zh_p and new_p != en_p:
				max_is_zh_count += 1
			merged_preds.append(new_p)
	assert len(merged_preds) == len(en_preds)
	print(f"In {max_is_zh_count} entries, the maximum scores come exclusively from the Chinese Entailment Graph!")
	return merged_preds


def avg_reduce(en_preds, zh_preds, zh_factor=1., en_lemma=True):
	assert len(en_preds) == len(zh_preds)
	merged_preds = []
	for pid, (en_p, zh_p) in enumerate(zip(en_preds, zh_preds)):
		if en_p == 1 and en_lemma:
			merged_preds.append(en_p)
		else:
			new_p = (en_p + zh_factor * zh_p) / 2.0
			merged_preds.append(new_p)
	assert len(merged_preds) == len(en_preds)
	return merged_preds


def min_reduce(en_preds, zh_preds, zh_factor=1., en_lemma=True):
	assert len(en_preds) == len(zh_preds)
	merged_preds = []
	min_is_zh_count = 0  # the count of the min being *EXCLUSIVELY* from Chinese EntGraph
	for pid, (en_p, zh_p) in enumerate(zip(en_preds, zh_preds)):
		if en_p == 1 and en_lemma:
			merged_preds.append(en_p)
		else:
			new_p = min(en_p, zh_factor*zh_p)
			if new_p == zh_factor*zh_p and new_p != en_p:
				min_is_zh_count += 1
			merged_preds.append(new_p)
	assert len(merged_preds) == len(en_preds)
	print(f"In {min_is_zh_count} entries, the minimum scores come exclusively from the Chinese Entailment Graph!")
	return merged_preds


def display(golds, preds, args, label, from_zero=False, rec_from=None):
	(prec, rec, thres) = precision_recall_curve(golds, preds)

	try:
		if from_zero:
			main_auc = evaluation.util_chinese.get_auc(prec, rec)
		elif rec_from is not None:
			new_prec = []
			new_rec = []
			first_point = True
			for i, (p, r) in enumerate(zip(prec, rec)):
				if r >= rec_from:
					if i == 0:
						new_prec.append(p)
						new_rec.append(r)
						first_point = False
					elif first_point:
						diff_p = prec[i-1]-p
						last_r = rec[i-1]
						r_portion = (r-rec_from)/(r-last_r)
						edge_p = p + diff_p*r_portion
						new_prec.append(edge_p)
						new_rec.append(rec_from)
						new_prec.append(p)
						new_rec.append(r)
						first_point = False
					else:
						new_prec.append(p)
						new_rec.append(r)
			main_auc = evaluation.util_chinese.get_auc(new_prec, new_rec)

		else:
			main_auc = evaluation.util_chinese.get_auc(prec[:-1], rec[:-1])
	except Exception as e:
		print(e, file=sys.stderr)
		main_auc = 0.0
	print(f"Main AUC for {label} setting: {main_auc}!")
	with open(args.output % label, 'w') as fp:
		prec = prec.tolist()
		rec = rec.tolist()
		thres = thres.tolist()
		while len(thres) < len(prec):
			thres.append(100)
		fp.write(f'main auc: {main_auc}\n')
		for p, r, t in zip(prec, rec, thres):
			fp.write(f'{p}\t{r}\t{t}\n')
	with open(args.output_Y % label, 'w') as fp:
		for g, p in zip(golds, preds):
			fp.write(f'{g} {p}\n')
	return main_auc, [prec, rec, thres, golds, preds]


def store_details(auc, details, ofn, oYfn):
	prec, rec, thres, golds, preds = details
	with open(ofn, 'w') as fp:
		fp.write(f'main auc: {auc}\n')
		for p, r, t in zip(prec, rec, preds):
			fp.write(f'{p}\t{r}\t{t}\n')
	with open(oYfn, 'w') as fp:
		for g, p in zip(golds, preds):
			fp.write(f'{g} {p}\n')


def main():
	parser = argparse.ArgumentParser()
	#parser.add_argument('--en_input', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/results_Teddy/Aug_context_MC_dev_global_Y.txt')
	parser.add_argument('--en_input', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/results_en/pr_rec/global_scores_Y.txt')
	parser.add_argument('--zh_input', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/results/pr_rec_orig_dev/global_scores_orig_dev_binc_avg_Y.txt')
	parser.add_argument('--zh_bert', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/results/pr_rec_orig_dev/bert_sim_preds.txt')
	parser.add_argument('--output', type=str, default='../gfiles/results/pr_rec_merged_dev/scores_%s.txt')
	parser.add_argument('--output_Y', type=str, default='../gfiles/results/pr_rec_merged_dev/scores_Y_%s.txt')
	parser.add_argument('--start', type=float, default=0)
	parser.add_argument('--end', type=float, default=2)
	parser.add_argument('--step', type=float, default=0.1)

	args = parser.parse_args()

	en_golds = []
	en_preds = []
	zh_golds = []
	zh_preds = []
	zhbert_golds = []
	zhbert_preds = []

	with open(args.en_input, 'r') as fp:
		for line in fp:
			en_g, en_p = line.strip().split(' ')
			en_golds.append(int(en_g))
			en_preds.append(float(en_p))

	with open(args.zh_input, 'r') as fp:
		for line in fp:
			zh_g, zh_p = line.strip().split(' ')
			zh_golds.append(int(float(zh_g)))
			zh_preds.append(float(zh_p))

	with open(args.zh_bert, 'r') as fp:
		for line in fp:
			zhbert_g, zhbert_p = line.strip().split(' ')
			zhbert_golds.append(int(zhbert_g))
			zhbert_preds.append(float(zhbert_p))

	assert len(en_golds) == len(en_preds)
	assert len(zh_golds) == len(zh_preds)
	assert len(zhbert_golds) == len(zhbert_preds)
	assert len(en_golds) == len(zh_golds)
	assert len(en_golds) == len(zhbert_golds)

	for gid, (en_g, zh_g, zhbert_g) in enumerate(zip(en_golds, zh_golds, zhbert_golds)):
		if en_g != zh_g:
			raise AssertionError
		elif en_g != zhbert_g:
			raise AssertionError

	main_auc_en, en_details = display(en_golds, en_preds, args, 'pure_en')
	rec_from_en = en_details[1][-2]
	main_auc_zh, zh_details = display(en_golds, zh_preds, args, 'pure_zh', rec_from=None)
	main_auc_zhbert, zhbert_details = display(en_golds, zhbert_preds, args, 'pure_zhbert', rec_from=None)
	#store_details(main_auc_en, en_details, args.output%'pure_en', args.output_Y%'pure_en')
	#store_details(main_auc_zh, zh_details, args.output % 'pure_zh', args.output_Y % 'pure_zh')

	enzh_max_auc = 0
	enzh_best_zhfactor = -1
	enzh_best_details = None
	for zh_factor in np.arange(args.start, args.end, args.step):
		en_zh_preds = en_zh_dictorder(en_preds, zh_preds, zh_factor=zh_factor)
		main_auc, enzh_details = display(en_golds, en_zh_preds, args, 'enzh'+'_%.2f'%zh_factor, rec_from=rec_from_en)
		if main_auc > enzh_max_auc:
			enzh_max_auc = main_auc
			enzh_best_zhfactor = zh_factor
			enzh_best_details = enzh_details
	print(f"Best zh factor for EN-ZH Dictionary Order is {enzh_best_zhfactor}, getting best main auc @ {enzh_max_auc}", file=sys.stderr)
	store_details(enzh_max_auc, enzh_best_details, args.output%'enzh_best', args.output_Y%'enzh_best')

	zhen_max_auc = 0
	zhen_best_zhfactor = -1
	zhen_best_details = None
	for zh_factor in np.arange(args.start, args.end, args.step):
		zh_en_preds = zh_en_dictorder(en_preds, zh_preds, zh_factor=zh_factor)
		main_auc, zhen_details = display(en_golds, zh_en_preds, args, 'zhen'+'_%.2f'%zh_factor, rec_from=rec_from_en)
		if main_auc > zhen_max_auc:
			zhen_max_auc = main_auc
			zhen_best_zhfactor = zh_factor
			zhen_best_details = zhen_details
	print(f"Best zh factor for ZH-EN Dictionary Order is {zhen_best_zhfactor}, getting best main auc @ {zhen_max_auc}", file=sys.stderr)
	store_details(zhen_max_auc, zhen_best_details, args.output % 'zhen_best', args.output_Y % 'zhen_best')

	max_max_auc = 0
	max_best_zhfactor = -1
	max_best_details = None
	for zh_factor in np.arange(args.start, args.end, args.step):
		max_preds = max_reduce(en_preds, zh_preds, zh_factor=zh_factor)
		main_auc, max_details = display(en_golds, max_preds, args, 'max'+'_%.2f'%zh_factor, rec_from=rec_from_en)
		if main_auc > max_max_auc:
			max_max_auc = main_auc
			max_best_zhfactor = zh_factor
			max_best_details = max_details
	print(f"Best zh factor for MAX is {max_best_zhfactor}, getting best main auc @ {max_max_auc}", file=sys.stderr)
	store_details(max_max_auc, max_best_details, args.output % 'max_best', args.output_Y % 'max_best')

	avg_max_auc = 0
	avg_best_zhfactor = -1
	avg_best_details = None
	for zh_factor in np.arange(args.start, args.end, args.step):
		avg_preds = avg_reduce(en_preds, zh_preds, zh_factor=zh_factor)
		main_auc, avg_details = display(en_golds, avg_preds, args, 'avg'+'_%.2f'%zh_factor, rec_from=rec_from_en)
		if main_auc > avg_max_auc:
			avg_max_auc = main_auc
			avg_best_zhfactor = zh_factor
			avg_best_details = avg_details
	print(f"Best zh factor for AVG is {avg_best_zhfactor}, getting best main auc @ {avg_max_auc}", file=sys.stderr)
	store_details(avg_max_auc, avg_best_details, args.output % 'avg_best', args.output_Y % 'avg_best')

	min_max_auc = 0
	min_best_zhfactor = -1
	min_best_details = None
	for zh_factor in np.arange(args.start, args.end, args.step):
		min_preds = min_reduce(en_preds, zh_preds, zh_factor=zh_factor)
		main_auc, min_details = display(en_golds, min_preds, args, 'min'+'_%.2f'%zh_factor, rec_from=rec_from_en)
		if main_auc > min_max_auc:
			min_max_auc = main_auc
			min_best_zhfactor = zh_factor
			min_best_details = min_details
	print(f"Best zh factor for MIN is {min_best_zhfactor}, getting best main auc @ {min_max_auc}", file=sys.stderr)
	store_details(min_max_auc, min_best_details, args.output % 'min_best', args.output_Y % 'min_best')


if __name__ == '__main__':
	main()
