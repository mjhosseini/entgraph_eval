import matplotlib.pyplot as plt
import argparse
import sys


def plot_from_file(ifn, label, only_highprec):
	precrecs = [[], []]
	with open(ifn, 'r', encoding='utf8') as fp:
		for line in fp:
			if 'auc' in line:
				continue
			else:
				val_list = line.strip().split('\t')
				val_list_ = []
				for val in val_list:
					val_list_ += val.split(' ')
				val_list = val_list_
				val_list = [float(x) for x in val_list]
				if len(val_list) == 2:
					pr, rec = val_list
				elif len(val_list) == 3:
					pr, rec, thres = val_list
				else:
					raise AssertionError
				if (pr > 0.5 or not only_highprec) and rec != 0:
					precrecs[0].append(pr)
					precrecs[1].append(rec)
	plt.plot(precrecs[1], precrecs[0], label=label)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='dev_merge', help='[basic/baselines/google_PLM/nosame/dev_merge/test_merge]')
	parser.add_argument('--grid_search', type=int, default=1, help='Whether or not to grid search a best proportion between 0 and 2')
	parser.add_argument('--only_highprec', type=int, default=1, help='Whether or not to only plot the area with 50% plus precision.')
	args = parser.parse_args()
	args.grid_search = True if args.grid_search > 0 else False
	args.only_highprec = True if args.only_highprec > 0 else False

	input_list = []
	if args.mode == 'basic':
		input_list.append(['./results/pr_rec_orig_test/bert_sim_scores.txt', 'bert'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test.txt', 'V2'])
	elif args.mode == 'baselines_dev':
		input_list.append(['./results/pr_rec_orig_dev/bert_sim_scores.txt', 'Bert'])
		# input_list.append(['./results/pr_rec_orig_dev_exhaust_jia_22/global_scores_orig_dev_apooling_binc_JIA.txt', 'Jia'])
		input_list.append(['./results/pr_rec_orig_dev_exhaust_bsl_22/global_scores_orig_dev_apooling_binc_BSL.txt', 'DDPORE'])
		input_list.append(['./results_en/pr_rec/global_scores.txt', 'Housseini 2018'])
		input_list.append(['./results/pr_rec_orig_dev_exhaust_22/global_scores_orig_dev_apooling_binc.txt', 'Zh_EG'])
		input_list.append(['./results/pr_rec_merged_dev/scores_avg_0.30.txt', 'Ensemble AVG'])
		input_list.append(['./results/pr_rec_merged_dev_plusplus/scores_avg_0.10.txt', 'Ensemble++ AVG'])
	elif args.mode == 'baselines_test':
		input_list.append(['./results/pr_rec_orig_test/bert_sim_scores.txt', 'Bert'])
		input_list.append(['./results/pr_rec_orig_test_exhaust_jia_22/global_scores_orig_test_apooling_binc_JIA.txt', 'Jia'])
		input_list.append(['./results/pr_rec_orig_test_exhaust_bsl_22/global_scores_orig_test_apooling_binc_BSL.txt', 'DDPORE'])
		input_list.append(['./results/pr_rec_orig_test_exhaust_22/global_scores_orig_test_apooling_binc.txt', 'Zh_EG'])
		input_list.append(['./results_en/pr_rec/global_scores_test.txt', 'Housseini 2018'])
		input_list.append(['./results/pr_rec_merged_test/scores_max_0.50.txt', 'Ensemble MAX'])
		input_list.append(['./results_Teddy/Aug_context_MC_test_global.txt', 'Housseini 2021'])
		input_list.append(['./results/pr_rec_merged_test_plusplus/scores_avg_0.10.txt', 'Ensemble++ AVG'])
	elif args.mode == 'google_PLM':
		input_list.append(['./results/pr_rec_google_PLM/bert_sim_scores.txt', 'Bert Sims'])
		input_list.append(['./results/pr_rec_google_PLM/global_scores_V1_google_PLM.txt', 'V1'])
		input_list.append(['./results/pr_rec_google_PLM/global_scores_google_PLM.txt', 'V2'])
	elif args.mode == 'nosame':
		input_list.append(['./results/pr_rec_nosame/bert_sim_scores.txt', 'Bert Sims'])
		input_list.append(['./results/pr_rec_nosame/global_scores_V1_nosame.txt', 'V1'])
		input_list.append(['./results/pr_rec_nosame/global_scores_nosame.txt', 'V2'])
	elif args.mode == 'local_scores_orig_test':
		input_list.append(['./results/pr_rec_orig_test/bert_sim_lemma_scores.txt', 'bert'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_cos.txt', 'cos'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_weeds.txt', 'weeds'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_weeds_pmi.txt', 'weeds_pmi'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_lin.txt', 'lin'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_binc.txt', 'binc'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_binc_g.txt', 'binc_g'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_weeds_pr.txt', 'weeds_pr'])
	elif args.mode == 'local_scores_orig_dev':
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev_cos.txt', 'cos'])
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev_weeds.txt', 'weeds'])
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev_weeds_pmi.txt', 'weeds_pmi'])
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev_lin.txt', 'lin'])
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev_binc.txt', 'binc'])
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev_binc_G.txt', 'binc_g'])
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev_weeds_pr.txt', 'weeds_pr'])
	elif args.mode == 'tricks_orig_dev_22':
		input_list.append(['./results/pr_rec_orig_dev_exhaust_22/global_scores_orig_dev_apooling_binc.txt', 'exhaust_apooling'])
		input_list.append(['./results/pr_rec_orig_dev_exhaust_22/global_scores_orig_dev_mpooling_binc.txt', 'exhaust_mpooling'])
		input_list.append(['./results/pr_rec_orig_dev_22/global_scores_orig_dev_alltype_apooling_binc.txt', 'alltype-apooling'])
		input_list.append(['./results/pr_rec_orig_dev_22/global_scores_orig_dev_alltype_mpooling_binc.txt', 'alltype-mpooling'])
		input_list.append(['./results/pr_rec_orig_dev_22/global_scores_orig_dev_binc.txt', 'plain'])
		input_list.append(['./results/pr_rec_orig_dev_22/global_scores_orig_dev_binc_avg.txt', 'plain_avg'])
		input_list.append(['./results/pr_rec_orig_dev_22_G/global_scores_orig_dev_binc_G.txt', 'global'])
		input_list.append(['./results/pr_rec_orig_dev_exhaust_22_G/global_scores_orig_dev_apooling_binc_G.txt', 'global_exhaust_apooling'])
	elif args.mode == 'tricks_orig_dev_23':
		input_list.append(['./results/pr_rec_orig_dev_exhaust_23/global_scores_orig_dev_apooling_binc.txt', 'exhaust_apooling'])
		input_list.append(['./results/pr_rec_orig_dev_exhaust_23/global_scores_orig_dev_mpooling_binc.txt', 'exhaust_mpooling'])
		input_list.append(['./results/pr_rec_orig_dev_23/global_scores_orig_dev_alltype_apooling_binc.txt', 'alltype-apooling'])
		input_list.append(['./results/pr_rec_orig_dev_23/global_scores_orig_dev_alltype_mpooling_binc.txt', 'alltype-mpooling'])
		input_list.append(['./results/pr_rec_orig_dev_23/global_scores_orig_dev_binc.txt', 'plain'])
		input_list.append(['./results/pr_rec_orig_dev_23/global_scores_orig_dev_binc_avg.txt', 'plain_avg'])
	elif args.mode == 'tricks_orig_test_22':
		input_list.append(
			['./results/pr_rec_orig_test_exhaust_22/global_scores_orig_test_apooling_binc.txt', 'exhaust_apooling'])
		input_list.append(
			['./results/pr_rec_orig_test_exhaust_22/global_scores_orig_test_mpooling_binc.txt', 'exhaust_mpooling'])
		#input_list.append(
			#'./results/pr_rec_orig_test_22/global_scores_orig_test_alltype_apooling_binc.txt', 'alltype-apooling'])
		#input_list.append(
			#['./results/pr_rec_orig_test_22/global_scores_orig_test_alltype_mpooling_binc.txt', 'alltype-mpooling'])
		input_list.append(['./results/pr_rec_orig_test_22/global_scores_orig_test_binc.txt', 'plain'])
		input_list.append(['./results/pr_rec_orig_test_22/global_scores_orig_test_binc_avg.txt', 'plain_avg'])
	elif args.mode == 'dev_merge':
		input_list.append(['./results_en/pr_rec/global_scores.txt', 'English'])
		input_list.append(['./results/pr_rec_orig_dev/global_scores_orig_dev.txt', 'Chinese'])
		input_list.append(['./results/pr_rec_merged_dev/scores_pure_zhbert.txt', 'Chinese Bert Cosine Sims'])
		if args.grid_search > 0:
			suff = 'best'
		else:
			suff = '1.00'
		input_list.append(['./results/pr_rec_merged_dev/scores_enzh_%s.txt' % suff, 'En Then Zh'])
		input_list.append(['./results/pr_rec_merged_dev/scores_zhen_%s.txt' % suff, 'Zh Then En'])
		input_list.append(['./results/pr_rec_merged_dev/scores_max_%s.txt' % suff, 'MAX'])
		input_list.append(['./results/pr_rec_merged_dev/scores_avg_%s.txt' % suff, 'AVG'])
		input_list.append(['./results/pr_rec_merged_dev/scores_min_%s.txt' % suff, 'MIN'])
	elif args.mode == 'test_merge':
		input_list.append(['./results_en/pr_rec/global_scores_test.txt', 'English'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test.txt', 'Chinese'])
		input_list.append(['./results/pr_rec_merged_test/scores_pure_zhbert.txt', 'Chinese Bert Cosine Sims'])
		if args.grid_search > 0:
			suff = 'best'
		else:
			suff = '1.00'
		input_list.append(['./results/pr_rec_merged_test/scores_enzh_%s.txt' % suff, 'En Then Zh'])
		input_list.append(['./results/pr_rec_merged_test/scores_zhen_%s.txt' % suff, 'Zh Then En'])
		input_list.append(['./results/pr_rec_merged_test/scores_max_%s.txt' % suff, 'MAX'])
		input_list.append(['./results/pr_rec_merged_test/scores_avg_%s.txt' % suff, 'AVG'])
		input_list.append(['./results/pr_rec_merged_test/scores_min_%s.txt' % suff, 'MIN'])
	elif args.mode == 'test_merge_dev_factor':
		factor_list = ['0.50', '1.90', '0.50', '0.30', '1.90']
		print(f"Factor List: {factor_list}", file=sys.stderr)
		input_list.append(['./results_en/pr_rec/global_scores_test.txt', 'English'])
		input_list.append(['./results/pr_rec_orig_test/global_scores_orig_test_weeds_pmi.txt', 'Chinese'])
		input_list.append(['./results/pr_rec_merged_test/scores_pure_zhbert.txt', 'Chinese Bert Cosine Sims'])
		input_list.append(['./results/pr_rec_merged_test/scores_enzh_%s.txt' % factor_list[0], 'En Then Zh'])
		input_list.append(['./results/pr_rec_merged_test/scores_zhen_%s.txt' % factor_list[1], 'Zh Then En'])
		input_list.append(['./results/pr_rec_merged_test/scores_max_%s.txt' % factor_list[2], 'MAX'])
		input_list.append(['./results/pr_rec_merged_test/scores_avg_%s.txt' % factor_list[3], 'AVG'])
		input_list.append(['./results/pr_rec_merged_test/scores_min_%s.txt' % factor_list[4], 'MIN'])
	else:
		raise AssertionError

	for ifn, label in input_list:
		plot_from_file(ifn, label, args.only_highprec)
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.title("Precision Recall Curves")
	plt.legend()
	plt.draw()
	plt.show()


if __name__ == '__main__':
	main()