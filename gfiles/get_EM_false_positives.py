preds_fp = open('./results/pr_rec/global_scores_Y.txt', 'r', encoding='utf8')
ent_fp = open('./chinese_ent/implications_dev_translated.tsv', 'r', encoding='utf8')
raw_fp = open('./chinese_ent/implications_dev_translated_raw.tsv', 'r', encoding='utf8')

threshold=0.7

for lid, (preds_line, ent_line, raw_line) in enumerate(zip(preds_fp, ent_fp, raw_fp)):
	gold, predicted = preds_line.strip().split()
	if float(gold) == 0 and float(predicted) > threshold:
		print(lid)
		print(preds_line.strip())
		print(ent_line.strip())
		print(raw_line.strip())
		print("")
