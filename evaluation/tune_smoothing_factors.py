import json
import argparse
import os
import torch
from evaluation.util_chinese import get_auc  # this get_auc is reusable in English setting as well.
from sklearn.metrics import precision_recall_curve as pr_curve_sklearn
from pytorch_lightning.metrics.functional.classification import precision_recall_curve as pr_curve_pt
from qaeval_chinese_general_functions import compute_ss_auc


def calc_bsln_prec(labels):
	posi_cnt = 0
	negi_cnt = 0
	for lbl in labels:
		if lbl == 1:
			posi_cnt += 1
		elif lbl == 0:
			negi_cnt += 1
		else:
			raise AssertionError
	bsln_prec = posi_cnt / (posi_cnt + negi_cnt)
	print(f"baseline precision: {bsln_prec};")
	return bsln_prec


def get_final_scores(tscores: list, uscores: list, smt_scores: list, factor: float):
	tscr_in_final_cnt = 0
	uscr_in_final_cnt = 0
	sscr_in_final_cnt = 0
	null_in_final_cnt = 0
	final_scores = []
	for eidx, (tscr, uscr, sscr) in enumerate(zip(tscores, uscores, smt_scores)):
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

		sscr *= factor
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
	print(
		f"Final scores from tscr: {tscr_in_final_cnt}; from uscr: {uscr_in_final_cnt}; from sscr: {sscr_in_final_cnt}; "
		f"empty: {null_in_final_cnt};")
	return final_scores


def calc_ss_aucs(final_scores, labels, bsln_prec):
	try:
		final_labels_pt = torch.tensor(labels)
		final_scores_pt = torch.tensor(final_scores)
		pt_prec, pt_rec, pt_thres = pr_curve_pt(final_scores_pt, final_labels_pt)
		ss_bsln_auc = compute_ss_auc(
			pt_prec, pt_rec,
			filter_threshold=bsln_prec
		)
		ss_50_auc = compute_ss_auc(
			pt_prec, pt_rec,
			filter_threshold=0.5
		)

		ss_rel_prec = torch.tensor([max(p - bsln_prec, 0) for p in pt_prec], dtype=torch.float)
		ss_rel_rec = torch.tensor([r for r in pt_rec], dtype=torch.float)
		ss_auc_norm = compute_ss_auc(
			ss_rel_prec, ss_rel_rec,
			filter_threshold=0.0
		)
		ss_auc_norm /= (1 - bsln_prec)
		print(f"S&S 50 AUC: {ss_50_auc}; S&S bsln AUC: {ss_bsln_auc}; S&S AUC NORM: {ss_auc_norm};")
		print("")

	except Exception as e:
		print(f"Exception when calculating S&S style AUC!")
		print(e)
		ss_50_auc = None
		ss_bsln_auc = None
		ss_auc_norm = None
	return ss_50_auc, ss_bsln_auc, ss_auc_norm


def main(args):
	input_path = os.path.join(args.root, args.in_fn)
	output_path = os.path.join(args.root, args.out_fn)

	with open(input_path, 'r', encoding='utf8') as in_fp:
		tscores, uscores, smt_scores, labels = [], [], [], []
		for line in in_fp:
			# fp.write(f"{t}\t{u}\t{smt}\t{s}\t{l}\n")
			tscr, uscr, smt_scr, fnl_scr, lbl = line.strip().split('\t')
			tscores.append(float(tscr))
			uscores.append(float(uscr))
			smt_scores.append(float(smt_scr))
			labels.append(int(lbl))

		factor_candidate_aucs = {x * 0.002: {'50': None, 'bsln': None, 'norm': None} for x in range(500)}
		bsln_prec = calc_bsln_prec(labels)
		best_auc_norm = None
		best_factor = None
		best_precs, best_recs, best_thres = [], [], []

		for smoothing_factor in factor_candidate_aucs:
			final_scores = get_final_scores(tscores, uscores, smt_scores, smoothing_factor)
			skl_prec, skl_rec, skl_thres = pr_curve_sklearn(labels, final_scores)
			assert len(skl_prec) == len(skl_rec) and len(skl_prec) == len(skl_thres) + 1
			skl_auc_value = get_auc(skl_prec[1:], skl_rec[1:])
			print(f"Hosseini Area under curve: {skl_auc_value};")
			ss_50_auc, ss_bsln_auc, ss_auc_norm = calc_ss_aucs(final_scores, labels, bsln_prec)

			factor_candidate_aucs[smoothing_factor] = {'50': ss_50_auc, 'bsln': ss_bsln_auc, 'norm': ss_auc_norm}
			if best_auc_norm is None or ss_auc_norm > best_auc_norm:
				best_auc_norm = ss_auc_norm
				best_factor = smoothing_factor
				best_precs = skl_prec
				best_recs = skl_rec
				best_thres = skl_thres

		print(
			f"Best smoothing factor: {best_factor}; best auc_50: {factor_candidate_aucs[best_factor]['50']}; best auc_bsln: {factor_candidate_aucs[best_factor]['bsln']}; best auc_norm: {factor_candidate_aucs[best_factor]['norm']};")
		out_item = {
			'factor': best_factor,
			'auc_50': factor_candidate_aucs[best_factor]['50'],
			'auc_bsln': factor_candidate_aucs[best_factor]['bsln'],
			'auc_norm': factor_candidate_aucs[best_factor]['norm'],
			'precs': best_precs,
			'recs': best_recs,
			'thres': best_thres
		}
		with open(output_path, 'w', encoding='utf8') as ofp:
			json.dump(out_item, ofp, ensure_ascii=False, indent=4)
		print(f"output written to {output_path}.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_fn', type=str, required=True)
	parser.add_argument('--root', type=str, default='../gfiles/qaeval_en_results_prd/15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_0.0_freqmap_%s')
	parser.add_argument('--subset', type=str, default='test')

	args = parser.parse_args()

	assert args.in_fn.endswith('_predictions.txt')
	args.out_fn = args.in_fn[:-16] + '_weighted_prt_vals.json'

	args.root = args.root % args.subset
	main(args)
