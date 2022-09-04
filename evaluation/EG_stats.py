import os
import numpy as np
import argparse

def bucket_preds(num_preds_list, bars):
	print(f"total number of entailment graphs: {len(num_preds_list)}")
	for _bar in bars:
		accepted_num_graphs = 0
		for g in num_preds_list:
			if g > _bar:
				accepted_num_graphs += 1

		print(f"number of entailment graphs with >{_bar} predicates: {accepted_num_graphs}")



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/typedEntGrDir_Chinese2_2_V3')

args = parser.parse_args()

files = os.listdir(args.input)
files = list(np.sort(files))

num_preds_list = []

for f in files:
	path = os.path.join(args.input, f)
	if '_rels.txt' not in f:
		continue

	num_preds = 0
	with open(path, 'r', encoding='utf8') as fp:
		if len(num_preds_list) % 50 == 0:
			print(len(num_preds_list))
		for line in fp:
			if 'predicate: ' in line[:11]:
				num_preds += 1
	num_preds_list.append(num_preds)

bucket_preds(num_preds_list, [100, 1000, 10000, 74584, 100000])



