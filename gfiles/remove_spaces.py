import os

file_list = os.listdir('typedEntGrDir_Chinese3_3_V2')

for item in file_list:
	fn = './typedEntGrDir_Chinese3_3_V2/' + item
	if fn.endswith('_sim.txt'):
		out_lines = []
		with open(fn, 'r', encoding='utf8') as fp:
			for line in fp:
				if 'predicate: ' not in line:
					out_lines.append(line.strip('\n'))
				else:  # predicate: (斩获.1, 斩获.2)#art#award
					pred = line[11:].strip('\n')
					pred_name, pred_type1, pred_type2 = pred.split('#')
					pred_name = pred_name.split(' ')
					assert len(pred_name) == 2
					pred_name = ''.join(pred_name)
					pred = pred_name+'#'+pred_type1+'#'+pred_type2
					new_line = 'predicate: '+pred
					out_lines.append(new_line)
		with open(fn, 'w', encoding='utf8') as fp:
			for line in out_lines:
				fp.write(line+'\n')



