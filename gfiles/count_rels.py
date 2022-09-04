import os
import matplotlib.pyplot as plt

root_dir = 'typedEntGrDir_Chinese3_3_V3'
# root_dir = './typed_rels_aida_figer_3_3_f'
file_list = os.listdir(root_dir)

rel_count_bucket = {}  # number of total rels
SVO_count_bucket = {}  # number of unique rels
rel_count = 0
preds = []
for item in file_list:
	fn = root_dir + '/' + item
	if fn.endswith('_rels.txt'):
		cur_rel_count = 0
		with open(fn, 'r', encoding='utf8') as fp:
			for line in fp:
				line = line.strip()
				if line[:6] == 'types:':
					continue
				elif line[:10] == 'predicate:':
					preds.append(line[11:].strip())
				elif len(line) == 0:
					continue
				elif line[:10] == 'inv idx of':
					break
				else:
					line = line.split(': ')
					#assert len(line) == 2
					cur_count = float(line[-1].strip())
					rel_count += cur_count
					cur_rel_count += cur_count
					if cur_count not in SVO_count_bucket:
						SVO_count_bucket[cur_count] = 0
					SVO_count_bucket[cur_count] += 1
		rel_count_bucket[item.strip('_rels.txt')] = cur_rel_count

SVO_count_bucket = {k: v for k, v in sorted(SVO_count_bucket.items(), key=lambda x: x[0])}

print("total rel count: ", rel_count)

print("rel count bucket: ", rel_count_bucket)
print("SVO count bucket: ")
print(SVO_count_bucket)
# keys = list(SVO_count_bucket.keys())
vals = [SVO_count_bucket[key] for key in SVO_count_bucket]

# plt.plot(keys, vals)
# plt.draw()
# plt.show()

total = sum(vals)
upto1_portion = SVO_count_bucket[1.] / float(total)
upto3_portion = sum([SVO_count_bucket[x] for x in [1., 2., 3.]]) / float(total)
upto5_portion = sum([SVO_count_bucket[x] for x in [1., 2., 3., 4., 5.]]) / float(total)
upto10_portion = sum([SVO_count_bucket[x] for x in [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]]) / float(total)

print(f"up to 1 portion: {upto1_portion};")
print(f"up to 3 portion: {upto3_portion};")
print(f"up to 5 portion: {upto5_portion};")
print(f"up to 10 portion: {upto10_portion};")
#with open('./SVO_count_bucket.tsv', 'w', encoding='utf8') as fp:
#	for key in SVO_count_bucket:
#		fp.write(f"{key}\t{SVO_count_bucket[key]}\n")

print(f"Total number of unique S-V-O triples: {total}")

preds = list(set(preds))
print("total predicate count: ", len(preds))


