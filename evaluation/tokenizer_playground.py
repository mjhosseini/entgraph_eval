from transformers import BertTokenizer
import sys


def en_tokenize_and_map(tokenizer, sent_lists, word_mappings):
	sent_strs = [' '.join(slst) for slst in sent_lists]

	batch_toks = tokenizer(sent_strs, padding=True)
	batch_input_ids = batch_toks['input_ids']
	batch_token_mappings = []

	for input_ids, slst, wmap in zip(batch_input_ids, sent_lists, word_mappings):
		cur_failure_flag = False
		assert input_ids[0] == 101
		perword_encoded = [tokenizer.encode(x, add_special_tokens=False) for x in slst]
		cur_tok_mapping = [0]  # this `0' is the label for the prepended [CLS] token
		tokens_ite = 1  # the first token is always 101, which is absent in perword_encoded, so we start with the second token.
		for i, word_toks in enumerate(perword_encoded):
			if cur_failure_flag:
				break
			for t in word_toks:
				if t != input_ids[tokens_ite]:
					print(f"tokenize_and_map: Warning! Mismatch between sentential encoding and individual token encoding!", file=sys.stderr)
					print(f"input_ids: {input_ids}", file=sys.stderr)
					print(f"perword_encoded: {perword_encoded}", file=sys.stderr)
					print(f"Falling back to extracting [CLS] token!", file=sys.stderr)
					print(f"")
					cur_failure_flag = True
					break
				else:
					# copy the mapping label of word i to this current token, since this token belongs to word i.
					cur_tok_mapping.append(wmap[i])
				tokens_ite += 1

		if cur_failure_flag:
			pass
		elif input_ids[tokens_ite] != 102:
			print(f"tokenize_and_map: Warning! DIFFERENT SEQ LENGTH between sentential encoding and individual token encoding!",
				  file=sys.stderr)
			print(f"input_ids: {input_ids}", file=sys.stderr)
			print(f"perword_encoded: {perword_encoded}", file=sys.stderr)
			print(f"Falling back to extracting [CLS] token!", file=sys.stderr)
			print(f"")
			cur_failure_flag = True
		else:
			# this following pair of operations is to move behind the [SEP] token into the range of [PAD] tokens
			cur_tok_mapping.append(0)
			tokens_ite += 1
			for t in input_ids[tokens_ite:]:
				if t != 0:
					print(
						f"tokenize_and_map: Warning! [SEP] token not followed by [PAD]!",
						file=sys.stderr)
					print(f"input_ids: {input_ids}", file=sys.stderr)
					print(f"Falling back to extracting [CLS] token!", file=sys.stderr)
					print(f"")
					cur_failure_flag = True
					break
				else:
					cur_tok_mapping.append(0)
		if cur_failure_flag:
			batch_token_mappings.append([1]+[0]*(len(input_ids)-1))
		else:
			assert len(cur_tok_mapping) == len(input_ids)
			batch_token_mappings.append(cur_tok_mapping)

	for input_ids, tok_mapping in zip(batch_input_ids, batch_token_mappings):
		assert len(input_ids) == len(tok_mapping)

	return batch_toks, batch_token_mappings


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

en_tokenize_and_map(tokenizer, [['I', 'have', 'a', 'rediculously', 'unfriendly', 'cat'], ['I', 'have', 'a', 'cat']],
				 [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0]])