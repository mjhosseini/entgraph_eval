import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from qaeval_chinese_general_functions import prepare_string_for_T5Tokenizer
import torch

t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')

sents = ['john shopped in the store, does this mean john went to the store? Yes ; john went to the store, does this mean john shopped in the store? No ; Mary owns the car, does this mean Mary bought the car? <extra_id_0>',
		 'john shopped in the store, does this mean john went to the store? Yes ; john went to the store, does this mean john shopped in the store? No ; Mary bought the car, does this mean Mary owns the car? <extra_id_0>',
		 'A drug kills infections, does this mean a drug is useful in infections? yes ; a drug is useful in infections, does this mean a drug kills infections? no ; touchdown run for yard, does this mean touchdown go with yard? <extra_id_0>',
		 'john shopped in the store, does this mean john went to the store? Yes; john went to the store, does this mean john shopped in the store? No; touchdown run for yard, does this mean touchdown go with yard? <extra_id_0>']

# sent = 'China is a <extra_id_0> of the world. </s>'

encoded = t5_tokenizer(sents, add_special_tokens=True, return_tensors='pt', padding=True)
outputs = t5_model.generate(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], num_beams=20,
							num_return_sequences=10, max_length=3, return_dict_in_generate=True, output_scores=True)
print(outputs.sequences.shape)
output_sequences = outputs.sequences.view(-1, 10, 3)
output_sequence_scores = outputs.sequences_scores.view(-1, 10)
decoded = t5_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
output_seqs = []
i = 0
for i, x in enumerate(decoded):
	if i % 10 == 0:
		output_seqs.append([])
	output_seqs[-1].append(x)
assert i % 10 == 10-1
assert len(output_seqs) == output_sequence_scores.shape[0]
print(output_seqs)

# for sent in sents:
# 	encoded = t5_tokenizer.encode_plus(sent, add_special_tokens=True, return_tensors='pt')
# 	input_ids = encoded['input_ids']
#
# 	outputs = t5_model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=10, max_length=10,
# 								 return_dict_in_generate=True, output_scores=True)
# 	print(outputs.sequences.shape)
# 	print(outputs.sequences_scores.shape)
# 	_0_index = sent.index('<extra_id_0>')
# 	_result_prefix = sent[:_0_index]
# 	_result_suffix = sent[_0_index+12:]  # 12 is the length of <extra_id_0>
#
# 	end_token = '<extra_id_1>'
# 	exp_sequences_scores = torch.exp(outputs.sequences_scores)
# 	assert outputs.sequences.shape[0] == exp_sequences_scores.shape[0]
# 	for op, op_scr in zip(outputs.sequences[:10], exp_sequences_scores[:10]):
# 		result = t5_tokenizer.decode(op[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
#
# 		if end_token in result:
# 			_end_token_idx = result.index(end_token)
# 			result = result[:_end_token_idx]
# 		else:
# 			pass
# 		print(_result_prefix+'【'+result+'】'+_result_suffix+f";\tscore: {op_scr}")
# 	print("")