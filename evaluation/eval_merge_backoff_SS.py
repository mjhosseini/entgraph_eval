import sys

sys.path.append("..")
import argparse
import numpy as np
import sys
from pytorch_lightning.metrics.functional.classification import precision_recall_curve
from pytorch_lightning.metrics.functional import auc
import torch
import json


def compute_auc(precisions, recalls,
				filter_threshold: float = 0.5):
	xs, ys = [], []
	for p, r in zip(precisions, recalls):
		if p >= filter_threshold:
			xs.append(r)
			ys.append(p)

	return auc(
		torch.cat([x.unsqueeze(0) for x in xs], 0),
		torch.cat([y.unsqueeze(0) for y in ys], 0)
	)


def eg_ss_dictorder(eg_preds, ss_preds, ss_factor=1., eg_lemma=True, ent_in_gr_flags=None):
	assert ss_factor <= 1
	assert len(eg_preds) == len(ss_preds)
	assert ent_in_gr_flags is None or len(ent_in_gr_flags) == len(eg_preds)
	merged_preds = []
	for pid, (eg_p, ss_p) in enumerate(zip(eg_preds, ss_preds)):
		if eg_p == 1 and eg_lemma:
			merged_preds.append(eg_p)
		elif ent_in_gr_flags is not None and ent_in_gr_flags[pid] == 1:
			merged_preds.append((1 - ss_factor) * eg_p)
		elif ent_in_gr_flags is not None and ent_in_gr_flags[pid] != 1:
			assert ent_in_gr_flags[pid] == 0
			# assert eg_p == 0
			merged_preds.append(ss_factor * ss_p)
		elif ent_in_gr_flags is None and eg_p > 0:
			merged_preds.append((1 - ss_factor) * eg_p)
		elif ent_in_gr_flags is None and eg_p == 0:
			merged_preds.append(ss_factor * ss_p)
		else:
			raise AssertionError
	assert len(merged_preds) == len(eg_preds)
	return merged_preds


def ss_eg_dictorder(eg_preds, ss_preds, ss_factor=1., eg_lemma=True):
	assert ss_factor <= 1
	assert len(eg_preds) == len(ss_preds)
	merged_preds = []
	for pid, (eg_p, ss_p) in enumerate(zip(eg_preds, ss_preds)):
		if eg_p == 1 and eg_lemma:
			merged_preds.append(eg_p)
		elif ss_p > 0:
			merged_preds.append(ss_factor * ss_p)
		else:
			merged_preds.append((1 - ss_factor) * eg_p)
	assert len(merged_preds) == len(eg_preds)
	return merged_preds


def max_reduce(eg_preds, ss_preds, ss_factor=1., eg_lemma=True):
	assert ss_factor <= 1
	assert len(eg_preds) == len(ss_preds)
	merged_preds = []
	max_is_ss_count = 0  # the count of the max being *EXCLUSIVELY* from Chinese EntGraph
	for pid, (eg_p, ss_p) in enumerate(zip(eg_preds, ss_preds)):
		if eg_p == 1 and eg_lemma:
			merged_preds.append(eg_p)
		else:
			new_p = max((1 - ss_factor) * eg_p, ss_factor * ss_p)
			if new_p == ss_factor * ss_p and new_p != (1 - ss_factor) * eg_p:
				max_is_ss_count += 1
			merged_preds.append(new_p)
	assert len(merged_preds) == len(eg_preds)
	print(f"In {max_is_ss_count} entries, the maximum scores come exclusively from the S&S model!")
	return merged_preds


def avg_reduce(eg_preds, ss_preds, ss_factor=1., eg_lemma=True):
	assert ss_factor <= 1
	assert len(eg_preds) == len(ss_preds)
	merged_preds = []
	for pid, (eg_p, ss_p) in enumerate(zip(eg_preds, ss_preds)):
		if eg_p == 1 and eg_lemma:
			merged_preds.append(eg_p)
		else:
			new_p = ((1 - ss_factor) * eg_p + ss_factor * ss_p) / 2.0
			merged_preds.append(new_p)
	assert len(merged_preds) == len(eg_preds)
	return merged_preds


def min_reduce(eg_preds, ss_preds, ss_factor=1., eg_lemma=True):
	assert ss_factor <= 1
	assert len(eg_preds) == len(ss_preds)
	merged_preds = []
	min_is_ss_count = 0  # the count of the min being *EXCLUSIVELY* from Chinese EntGraph
	for pid, (eg_p, ss_p) in enumerate(zip(eg_preds, ss_preds)):
		if eg_p == 1 and eg_lemma:
			merged_preds.append(eg_p)
		else:
			new_p = min((1 - ss_factor) * eg_p, ss_factor * ss_p)
			if new_p == ss_factor * ss_p and new_p != (1 - ss_factor) * eg_p:
				min_is_ss_count += 1
			merged_preds.append(new_p)
	assert len(merged_preds) == len(eg_preds)
	print(f"In {min_is_ss_count} entries, the minimum scores come exclusively from SS!")
	return merged_preds


def display(golds, preds, output, output_Y, label):
	golds = torch.tensor(golds)
	preds = torch.tensor(preds)
	(prec, rec, thres) = precision_recall_curve(preds, golds)
	if args.eval_dir:
		filter_threshold = 0.5
	else:
		filter_threshold = 0.2191

	try:
		main_auc = compute_auc(prec, rec, filter_threshold=0.5)
		bsln_auc = compute_auc(prec, rec, filter_threshold=filter_threshold)
		rel_prec = torch.tensor([max(p - filter_threshold, 0) for p in prec], dtype=torch.float)
		rel_rec = torch.tensor([r for r in rec], dtype=torch.float)
		norm_auc = compute_auc(
			rel_prec, rel_rec,
			filter_threshold=0.0
		)
		norm_auc /= (1 - filter_threshold)

	except Exception as e:
		print(e, file=sys.stderr)
		raise
	print(f"Main AUC for {label} setting: {main_auc}; BSLN AUC: {bsln_auc}; NORM AUC: {norm_auc}!")
	with open(output % label, 'w') as fp:
		prec = prec.tolist()
		rec = rec.tolist()
		thres = thres.tolist()
		while len(thres) < len(prec):
			thres.append(100)
		fp.write(f'main auc: {main_auc}\n')
		for p, r, t in zip(prec, rec, thres):
			fp.write(f'{p}\t{r}\t{t}\n')
	with open(output_Y % label, 'w') as fp:
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


def main_dev(eg_input_fn, ss_input_fn, ingr_flags_input_fn, output_fn, output_Y_fn, lh_dev2_indev_path,
			 lh_dev2dir_indev_path, is_eval_dir, _start, _end, _step):
	with open(ingr_flags_input_fn, 'r', encoding='utf8') as fp:
		ingr_flags = json.load(fp)

	eg_golds = []
	eg_preds = []
	ss_golds = []
	ss_preds = []

	with open(eg_input_fn, 'r') as fp:
		for line in fp:
			eg_g, eg_p = line.strip().split(' ')
			eg_golds.append(int(eg_g))
			eg_preds.append(float(eg_p))

	with open(ss_input_fn, 'r') as fp:
		for line in fp:
			ss_g, ss_p = line.strip().split(' ')
			ss_golds.append(int(float(ss_g)))
			ss_p = (float(ss_p) + 1) / 2
			ss_preds.append(float(ss_p))

	assert len(ingr_flags) == len(eg_golds)
	with open(lh_dev2_indev_path, 'r', encoding='utf8') as fp:
		dev2_idxes = json.load(fp)
	if is_eval_dir:
		with open(lh_dev2dir_indev_path, 'r', encoding='utf8') as fp:
			dev2_dir_idxes = json.load(fp)
		n_eg_golds = []
		n_eg_preds = []
		n_ingr_flags = []
		for eidx in dev2_dir_idxes:
			n_eg_golds.append(eg_golds[eidx])
			n_eg_preds.append(eg_preds[eidx])
			n_ingr_flags.append(ingr_flags[eidx])
		n_ss_golds = []
		n_ss_preds = []

		# for dev2idx, devidx in enumerate(dev2_idxes):
		# 	if devidx in dev2_dir_idxes:
		# 		n_ss_golds.append(ss_golds[dev2idx])
		# 		n_ss_preds.append(ss_preds[dev2idx])
		eg_golds = n_eg_golds
		eg_preds = n_eg_preds
		ingr_flags = n_ingr_flags
		# ss_golds = n_ss_golds
		# ss_preds = n_ss_preds
	else:
		n_eg_golds = []
		n_eg_preds = []
		n_ingr_flags = []
		for eidx in dev2_idxes:
			n_eg_golds.append(eg_golds[eidx])
			n_eg_preds.append(eg_preds[eidx])
			n_ingr_flags.append(ingr_flags[eidx])

		eg_golds = n_eg_golds
		eg_preds = n_eg_preds
		ingr_flags = n_ingr_flags

	assert len(eg_golds) == len(eg_preds)
	assert len(ss_golds) == len(ss_preds)
	assert len(eg_golds) == len(ss_golds)

	for gid, (eg_g, ss_g) in enumerate(zip(eg_golds, ss_golds)):
		if eg_g != ss_g:
			print("!")

	main_auc_eg, eg_details = display(ss_golds, eg_preds, output_fn, output_Y_fn, 'pure_eg')
	main_auc_ss, ss_details = display(ss_golds, ss_preds, output_fn, output_Y_fn, 'pure_ss')
	# store_details(main_auc_eg, eg_details, args.output%'pure_eg', args.output_Y%'pure_eg')
	# store_details(main_auc_ss, ss_details, args.output % 'pure_ss', args.output_Y % 'pure_ss')

	egss_nodewise_max_auc = 0
	egss_nodewise_best_ssfactor = -1
	egss_nodewise_best_details = None
	for ss_factor in np.arange(_start, _end, _step):
		eg_ss_nodewise_preds = eg_ss_dictorder(eg_preds, ss_preds, ss_factor=ss_factor, ent_in_gr_flags=ingr_flags)
		main_auc, egss_details = display(ss_golds, eg_ss_nodewise_preds, output_fn, output_Y_fn,
										 'egss_node' + '_%.3f' % ss_factor)
		if main_auc > egss_nodewise_max_auc:
			egss_nodewise_max_auc = main_auc
			egss_nodewise_best_ssfactor = ss_factor
			egss_nodewise_best_details = egss_details
	print(
		f"Best ss factor for EG-SS Dictionary Order (Nodewise) is {egss_nodewise_best_ssfactor}, getting best main auc @ {egss_nodewise_max_auc}",
		file=sys.stderr)
	store_details(egss_nodewise_max_auc, egss_nodewise_best_details, output_fn % 'egss_node_best',
				  output_Y_fn % 'egss_node_best')

	egss_edgewise_max_auc = 0
	egss_edgewise_best_ssfactor = -1
	egss_edgewise_best_details = None
	for ss_factor in np.arange(_start, _end, _step):
		eg_ss_preds = eg_ss_dictorder(eg_preds, ss_preds, ss_factor=ss_factor, ent_in_gr_flags=None)
		main_auc, egss_details = display(ss_golds, eg_ss_preds, output_fn, output_Y_fn,
										 'egss_edge' + '_%.3f' % ss_factor)
		if main_auc > egss_edgewise_max_auc:
			egss_edgewise_max_auc = main_auc
			egss_edgewise_best_ssfactor = ss_factor
			egss_edgewise_best_details = egss_details
	print(
		f"Best ss factor for EG-SS Dictionary Order (Edgewise) is {egss_edgewise_best_ssfactor}, getting best main auc @ {egss_edgewise_max_auc}",
		file=sys.stderr)
	store_details(egss_edgewise_max_auc, egss_edgewise_best_details, output_fn % 'egss_edge_best',
				  output_Y_fn % 'egss_edge_best')

	sseg_max_auc = 0
	sseg_best_ssfactor = -1
	sseg_best_details = None
	for ss_factor in np.arange(_start, _end, _step):
		ss_eg_preds = ss_eg_dictorder(eg_preds, ss_preds, ss_factor=ss_factor)
		main_auc, sseg_details = display(ss_golds, ss_eg_preds, output_fn, output_Y_fn, 'sseg' + '_%.3f' % ss_factor)
		if main_auc > sseg_max_auc:
			sseg_max_auc = main_auc
			sseg_best_ssfactor = ss_factor
			sseg_best_details = sseg_details
	print(f"Best ss factor for SS-EG Dictionary Order is {sseg_best_ssfactor}, getting best main auc @ {sseg_max_auc}",
		  file=sys.stderr)
	store_details(sseg_max_auc, sseg_best_details, output_fn % 'sseg_best', output_Y_fn % 'sseg_best')

	max_max_auc = 0
	max_best_ssfactor = -1
	max_best_details = None
	for ss_factor in np.arange(_start, _end, _step):
		max_preds = max_reduce(eg_preds, ss_preds, ss_factor=ss_factor)
		main_auc, max_details = display(ss_golds, max_preds, output_fn, output_Y_fn, 'max' + '_%.3f' % ss_factor)
		if main_auc > max_max_auc:
			max_max_auc = main_auc
			max_best_ssfactor = ss_factor
			max_best_details = max_details
	print(f"Best ss factor for MAX is {max_best_ssfactor}, getting best main auc @ {max_max_auc}", file=sys.stderr)
	store_details(max_max_auc, max_best_details, output_fn % 'max_best', output_Y_fn % 'max_best')

	avg_max_auc = 0
	avg_best_ssfactor = -1
	avg_best_details = None
	for ss_factor in np.arange(_start, _end, _step):
		avg_preds = avg_reduce(eg_preds, ss_preds, ss_factor=ss_factor)
		main_auc, avg_details = display(ss_golds, avg_preds, output_fn, output_Y_fn, 'avg' + '_%.3f' % ss_factor)
		if main_auc > avg_max_auc:
			avg_max_auc = main_auc
			avg_best_ssfactor = ss_factor
			avg_best_details = avg_details
	print(f"Best ss factor for AVG is {avg_best_ssfactor}, getting best main auc @ {avg_max_auc}", file=sys.stderr)
	store_details(avg_max_auc, avg_best_details, output_fn % 'avg_best', output_Y_fn % 'avg_best')

	return egss_nodewise_best_ssfactor, egss_edgewise_best_ssfactor, sseg_best_ssfactor, max_best_ssfactor, avg_best_ssfactor


# min_max_auc = 0
# min_best_ssfactor = -1
# min_best_details = None
# for ss_factor in np.arange(args.start, args.end, args.step):
# 	min_preds = min_reduce(eg_preds, ss_preds, ss_factor=ss_factor)
# 	main_auc, min_details = display(eg_golds, min_preds, args, 'min'+'_%.3f'%ss_factor)
# 	if main_auc > min_max_auc:
# 		min_max_auc = main_auc
# 		min_best_ssfactor = ss_factor
# 		min_best_details = min_details
# print(f"Best ss factor for MIN is {min_best_ssfactor}, getting best main auc @ {min_max_auc}", file=sys.stderr)
# store_details(min_max_auc, min_best_details, args.output % 'min_best', args.output_Y % 'min_best')


def main_test(eg_input_fn, ss_input_fn, ingr_flags_input_fn, output_fn, output_Y_fn, lh_testdir_intest_path,
			  is_eval_dir, _start, _end, _step,
			  egss_nodewise_best_ssfactor, egss_edgewise_best_ssfactor, sseg_best_ssfactor, max_best_ssfactor,
			  avg_best_ssfactor):
	with open(ingr_flags_input_fn, 'r', encoding='utf8') as fp:
		ingr_flags = json.load(fp)

	eg_golds = []
	eg_preds = []
	ss_golds = []
	ss_preds = []

	with open(eg_input_fn, 'r') as fp:
		for line in fp:
			eg_g, eg_p = line.strip().split(' ')
			eg_golds.append(int(eg_g))
			eg_preds.append(float(eg_p))

	with open(ss_input_fn, 'r') as fp:
		for line in fp:
			ss_g, ss_p = line.strip().split(' ')
			ss_golds.append(int(float(ss_g)))
			ss_p = (float(ss_p) + 1) / 2
			ss_preds.append(float(ss_p))

	assert len(ingr_flags) == len(eg_golds)

	if is_eval_dir:
		with open(lh_testdir_intest_path, 'r', encoding='utf8') as fp:
			testdir_idxes = json.load(fp)
		n_eg_golds = []
		n_eg_preds = []
		n_ingr_flags = []
		for eidx in testdir_idxes:
			n_eg_golds.append(eg_golds[eidx])
			n_eg_preds.append(eg_preds[eidx])
			n_ingr_flags.append(ingr_flags[eidx])
		eg_golds = n_eg_golds
		eg_preds = n_eg_preds
		ingr_flags = n_ingr_flags
	else:
		pass

	assert len(eg_golds) == len(eg_preds)
	assert len(ss_golds) == len(ss_preds)
	assert len(eg_golds) == len(ss_golds)

	for gid, (eg_g, ss_g) in enumerate(zip(eg_golds, ss_golds)):
		if eg_g != ss_g:
			print("!")

	main_auc_eg, eg_details = display(ss_golds, eg_preds, output_fn, output_Y_fn, 'pure_eg')
	main_auc_ss, ss_details = display(ss_golds, ss_preds, output_fn, output_Y_fn, 'pure_ss')
	# store_details(main_auc_eg, eg_details, args.output%'pure_eg', args.output_Y%'pure_eg')
	# store_details(main_auc_ss, ss_details, args.output % 'pure_ss', args.output_Y % 'pure_ss')

	eg_ss_nodewise_preds = eg_ss_dictorder(eg_preds, ss_preds, ss_factor=egss_nodewise_best_ssfactor,
										   ent_in_gr_flags=ingr_flags)
	egss_nodewise_main_auc, egss_nodewise_details = display(ss_golds, eg_ss_nodewise_preds, output_fn, output_Y_fn,
									 'egss_node_best')
	print(
		f"Best dev set ss factor for EG-SS Dictionary Order (Nodewise) is {egss_nodewise_best_ssfactor}, getting best main auc @ {egss_nodewise_main_auc}",
		file=sys.stderr)

	eg_ss_edgewise_preds = eg_ss_dictorder(eg_preds, ss_preds, ss_factor=egss_edgewise_best_ssfactor,
										   ent_in_gr_flags=None)
	egss_edgewise_main_auc, egss_edgewise_details = display(ss_golds, eg_ss_edgewise_preds, output_fn, output_Y_fn,
									 'egss_edge_best')
	print(
		f"Best dev set ss factor for EG-SS Dictionary Order (Edgewise) is {egss_edgewise_best_ssfactor}, getting best main auc @ {egss_edgewise_main_auc}",
		file=sys.stderr)

	ss_eg_preds = ss_eg_dictorder(eg_preds, ss_preds, ss_factor=sseg_best_ssfactor)
	sseg_main_auc, sseg_details = display(ss_golds, ss_eg_preds, output_fn, output_Y_fn, 'sseg' + '_%.3f' % sseg_best_ssfactor)
	print(f"Best dev set ss factor for SS-EG Dictionary Order is {sseg_best_ssfactor}, getting best main auc @ {sseg_main_auc}",
		  file=sys.stderr)

	max_preds = max_reduce(eg_preds, ss_preds, ss_factor=max_best_ssfactor)
	max_main_auc, max_details = display(ss_golds, max_preds, output_fn, output_Y_fn, 'max' + '_%.3f' % max_best_ssfactor)
	print(f"Best ss factor for MAX is {max_best_ssfactor}, getting best main auc @ {max_main_auc}", file=sys.stderr)

	avg_preds = avg_reduce(eg_preds, ss_preds, ss_factor=avg_best_ssfactor)
	avg_main_auc, avg_details = display(ss_golds, avg_preds, output_fn, output_Y_fn, 'avg' + '_%.3f' % avg_best_ssfactor)
	print(f"Best ss factor for AVG is {avg_best_ssfactor}, getting best main auc @ {avg_main_auc}", file=sys.stderr)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--eg_input', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/results_Teddy/Aug_context_MC_dev_global_Y.txt')
	parser.add_argument('--eg_input', type=str,
						default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/results_en/pr_rec/global_binc_%s_Y.txt')
	parser.add_argument('--ss_input', type=str,
						default='/Users/teddy/Downloads/egen_ns_binc_g_egdev_18k_73/en_levyholt_%s_Y.txt')
	parser.add_argument('--ingr_flags_input', type=str,
						default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/results_en/pr_rec/global_binc_%s_exact_found_labels.txt')
	parser.add_argument('--lh_dev2_indev_path', type=str,
						default='/Users/teddy/PycharmProjects/multilingual-lexical-inference/lm-lexical-inference/levy_holt_SS/dev_orig_lidxes.json',
						help='levy-holt dev2 indices in levy-holt dev.')
	parser.add_argument('--lh_dev2dir_indev_path', type=str,
						default='/Users/teddy/PycharmProjects/multilingual-lexical-inference/lm-lexical-inference/levy_holt_SS/dev2_dir_devidxes.json',
						help='')
	parser.add_argument('--lh_testdir_intest_path', type=str,
						default='/Users/teddy/PycharmProjects/multilingual-lexical-inference/lm-lexical-inference/levy_holt_SS/test_dir_idxes.json',
						help='')
	parser.add_argument('--output', type=str, default='../gfiles/results_en/pr_rec_ss_ensemble_%s/scores_%s.txt')
	parser.add_argument('--output_Y', type=str, default='../gfiles/results_en/pr_rec_ss_ensemble_%s/scores_Y_%s.txt')

	parser.add_argument('--eval_dir', action='store_true')
	parser.add_argument('--start', type=float, default=0)
	parser.add_argument('--end', type=float, default=1)
	parser.add_argument('--step', type=float, default=0.005)

	args = parser.parse_args()
	eg_input_fn = args.eg_input % 'dev'
	ingr_flags_input_fn = args.ingr_flags_input % 'dev'
	if args.eval_dir:
		egss_nodewise_best_ssfactor, egss_edgewise_best_ssfactor, sseg_best_ssfactor, max_best_ssfactor, avg_best_ssfactor = \
			0.765, 0.06, 0.005, 0.865, 0.135
	else:
		ss_input_fn = args.ss_input % 'dev'
		output_fn = args.output % ('dev', '%s')
		output_Y_fn = args.output_Y % ('dev', '%s')
		egss_nodewise_best_ssfactor, egss_edgewise_best_ssfactor, sseg_best_ssfactor, max_best_ssfactor, avg_best_ssfactor = \
			main_dev(eg_input_fn, ss_input_fn, ingr_flags_input_fn, output_fn, output_Y_fn, args.lh_dev2_indev_path,
					 args.lh_dev2dir_indev_path, args.eval_dir, args.start, args.end, args.step)

	eg_input_fn = args.eg_input % 'test'
	ingr_flags_input_fn = args.ingr_flags_input % 'test'
	if args.eval_dir:
		ss_input_fn = args.ss_input % 'dir_test'
		output_fn = args.output % ('dir_test', '%s')
		output_Y_fn = args.output_Y % ('dir_test', '%s')
	else:
		ss_input_fn = args.ss_input % 'test'
		output_fn = args.output % ('test', '%s')
		output_Y_fn = args.output_Y % ('test', '%s')
	main_test(eg_input_fn, ss_input_fn, ingr_flags_input_fn, output_fn, output_Y_fn, args.lh_testdir_intest_path,
			  args.eval_dir, args.start, args.end, args.step,
			  egss_nodewise_best_ssfactor, egss_edgewise_best_ssfactor, sseg_best_ssfactor, max_best_ssfactor, avg_best_ssfactor)
