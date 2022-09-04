lines = []

with open('global_scores_test.txt', 'r', encoding='utf8') as fp:
	for orig_line in fp:
		if 'auc' in orig_line:
			lines.append(orig_line)
			continue
		line = orig_line.split('.')
		assert len(line) == 4
		line = line[0] + '.' + line[1] + '.' + line[2][:-1] + ' ' + line[2][-1] + '.' + line[3]
		lines.append(line)

with open('global_scores_test.txt', 'w', encoding='utf8') as fp:
	for line in lines:
		fp.write(line)
