import json
import csv
import argparse
import os


def get_scores(fp):
	golds = []
	preds = []
	for line in fp:
		gold, pred = line.strip().split(' ')
		golds.append(int(gold))
		preds.append(float(pred))
	return golds, preds


def _print(x, file=None, verbose=True):
	assert verbose or file is not None
	if verbose:
		print(x)
	if file is not None:
		print(x, file=file)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--en_single', type=str, default='../gfiles/results_en/pr_rec/global_scores_test_Y.txt', help='Path to English single model.')
	parser.add_argument('-esb_root', type=str, default='../gfiles/results/pr_rec_merged_test/scores_Y_%s.txt')
	parser.add_argument('--ensembled', type=str, default='max_0.50')
	parser.add_argument('--threshold_esb', type=float, default=0.0207)
	parser.add_argument('--threshold_en', type=float, default=0.0119)
	parser.add_argument('--tp_path', type=str, default='../gfiles/case_study/true_pos_diff_%s.txt')
	parser.add_argument('--fp_path', type=str, default='../gfiles/case_study/false_pos_diff_%s.txt')
	parser.add_argument('--tn_path', type=str, default='../gfiles/case_study/true_neg_diff_%s.txt')
	parser.add_argument('--fn_path', type=str, default='../gfiles/case_study/false_neg_diff_%s.txt')

	args = parser.parse_args()

	esb_path = args.esb_root % args.ensembled
	tp_path = args.tp_path % args.ensembled
	fp_path = args.fp_path % args.ensembled
	tn_path = args.tn_path % args.ensembled
	fn_path = args.fn_path % args.ensembled

	tp_fp = open(tp_path, 'w', encoding='utf8')
	fp_fp = open(fp_path, 'w', encoding='utf8')
	tn_fp = open(tn_path, 'w', encoding='utf8')
	fn_fp = open(fn_path, 'w', encoding='utf8')

	all_rels_fp = open('../gfiles/test_ent_chinese/test_all_rels.txt', 'r', encoding='utf8')
	translated_fp = open('../gfiles/test_ent_chinese/test_translated.tsv', 'r', encoding='utf8')
	translated_raw_fp = open('../gfiles/test_ent_chinese/test_translated_raw.tsv', 'r', encoding='utf8')
	orig_en_fp = open('../gfiles/ent/test.txt', 'r', encoding='utf8')
	all_rels_reader = csv.reader(all_rels_fp, delimiter='\t')
	translated_reader = csv.reader(translated_fp, delimiter='\t')
	translated_raw_reader = csv.reader(translated_raw_fp, delimiter='\t')
	orig_en_reader = csv.reader(orig_en_fp, delimiter='\t')

	en_single_fp = open(args.en_single, 'r', encoding='utf8')
	esb_fp = open(esb_path, 'r', encoding='utf8')

	en_single_golds, en_single_scores = get_scores(en_single_fp)
	esb_golds, esb_scores = get_scores(esb_fp)

	for en_g, esb_g in zip(en_single_golds, esb_golds):
		assert en_g == esb_g

	idx = 0

	tp_count = 0
	fp_count = 0
	tn_count = 0
	fn_count = 0

	esb_tp_count = 0
	esb_fp_count = 0
	en_tp_count = 0
	en_fp_count = 0

	for idx, (rels, trans, trans_raw, en_raw) in enumerate(zip(all_rels_reader, translated_reader, translated_raw_reader, orig_en_reader)):
		if esb_scores[idx] > args.threshold_esb:
			if esb_golds[idx] == 1:
				esb_tp_count += 1
			elif esb_golds[idx] == 0:
				esb_fp_count += 1
			else:
				raise AssertionError

		if en_single_scores[idx] > args.threshold_en:
			if esb_golds[idx] == 1:
				en_tp_count += 1
			elif esb_golds[idx] == 0:
				en_fp_count += 1

		if esb_golds[idx] == 1 and esb_scores[idx] > args.threshold_esb and en_single_scores[idx] < args.threshold_en:
			_print("-----------------------------------------------", file=tp_fp)
			_print("ESB TRUE POS, EN_SINGLE FALSE NEG!", file=tp_fp)
			_print(f'Relations: {rels}', file=tp_fp)
			_print(f"Translations: {trans}", file=tp_fp)
			_print(f"Translations Raw: {trans_raw}", file=tp_fp)
			_print(f"Original English Raw: {en_raw}", file=tp_fp)
			_print("", file=tp_fp)
			tp_count += 1
		elif esb_golds[idx] == 0 and esb_scores[idx] > args.threshold_esb and en_single_scores[idx] < args.threshold_en:
			_print("-----------------------------------------------", file=fp_fp)
			_print("ESB FALSE POS, EN_SINGLE TRUE NEG!", file=fp_fp)
			_print(f'Relations: {rels}', file=fp_fp)
			_print(f"Translations: {trans}", file=fp_fp)
			_print(f"Translations Raw: {trans_raw}", file=fp_fp)
			_print(f"Original English Raw: {en_raw}", file=fp_fp)
			_print("", file=fp_fp)
			fp_count += 1
		elif esb_golds[idx] == 0 and esb_scores[idx] < args.threshold_esb and en_single_scores[idx] > args.threshold_en:
			_print("-----------------------------------------------", file=tn_fp)
			_print("ESB TRUE NEG, EN_SINGLE FALSE POS!", file=tn_fp)
			_print(f'Relations: {rels}', file=tn_fp)
			_print(f"Translations: {trans}", file=tn_fp)
			_print(f"Translations Raw: {trans_raw}", file=tn_fp)
			_print(f"Original English Raw: {en_raw}", file=tn_fp)
			_print("", file=tn_fp)
			tn_count += 1
		elif esb_golds[idx] == 1 and esb_scores[idx] < args.threshold_esb and en_single_scores[idx] > args.threshold_en:
			_print("-----------------------------------------------", file=fn_fp)
			_print("ESB FALSE NEG, EN_SINGLE TRUE POS!", file=fn_fp)
			_print(f'Relations: {rels}', file=fn_fp)
			_print(f"Translations: {trans}", file=fn_fp)
			_print(f"Translations Raw: {trans_raw}", file=fn_fp)
			_print(f"Original English Raw: {en_raw}", file=fn_fp)
			_print("", file=fn_fp)
			fn_count += 1
		else:
			pass

	print(f"Between {args.ensembled} and English single model;")
	print(f"Under thresholds {args.threshold_esb} for ensembled model, and {args.threshold_en} for English model;")
	print(f"{tp_count} entries are additionally correctly detected as positive;")
	print(f"{fp_count} entries are additionally incorrectly detected as positive;")
	print(f"{tn_count} entries are additionally correctly detected as negative;")
	print(f"{fn_count} entries are additionally incorrectly detected as negative!")

	print(f"Ensemble model predicted {esb_tp_count} true positives, {esb_fp_count} false positives and {len(esb_scores)-esb_tp_count-esb_fp_count} negatives in total!")
	print(f"English single model predicted {en_tp_count} true positives, {en_fp_count} false positives and {len(esb_scores) - en_tp_count - en_fp_count} negatives in total!")

	all_rels_fp.close()
	translated_fp.close()
	translated_raw_fp.close()
	orig_en_fp.close()
	en_single_fp.close()
	esb_fp.close()
	tp_fp.close()
	fp_fp.close()
	tn_fp.close()
	fn_fp.close()


if __name__ == '__main__':
	main()
