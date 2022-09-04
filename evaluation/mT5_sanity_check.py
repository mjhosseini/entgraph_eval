import transformers
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from qaeval_chinese_general_functions import prepare_string_for_T5Tokenizer
import torch

mt5_tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
mt5_model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

sents = ['咽炎是<extra_id_0>的原因。',
		'咽炎不会导致肺炎。咽炎可能造成肿痛。咽炎需要治疗。西瓜霜治疗咽炎。咽炎是<extra_id_0>的原因。',
		#'咽 炎 不 会 引 起 发 烧 。 咽 炎 导 致 肿 痛 。 西 瓜 霜 治 疗 咽 炎 。 咽 炎 是 <extra_id_0> 的 原 因 。 </s>',
		 '印度是<extra_id_0>的国家。',
		 '中国2020年GDP约为<extra_id_0>亿元。',
		 '印 度 是 <extra_id_0> 的 国 家 。',
		'印度 是 <extra_id_0> 的 国家 。',
		'中 国 是 <extra_id_0> 的 国 家 。',
		'美 国 是 <extra_id_0> 的 国 家 。',
		'法 国 是 <extra_id_0> 的 国 家 。',
		 'China is a <extra_id_0> of the world.',
		 'United States is a <extra_id_0> of the world.',
		 'France is a <extra_id_0> of the world.']

# sent = 'China is a <extra_id_0> of the world. </s>'

for sent in sents:
	sent = prepare_string_for_T5Tokenizer(sent)
	encoded = mt5_tokenizer.encode_plus(sent, add_special_tokens=True, return_tensors='pt')
	input_ids = encoded['input_ids']

	outputs = mt5_model.generate(input_ids=input_ids, num_beams=200, num_return_sequences=50, max_length=10,
								 return_dict_in_generate=True, output_scores=True)
	print(outputs.sequences.shape)
	print(outputs.sequences_scores.shape)
	_0_index = sent.index('<extra_id_0>')
	_result_prefix = sent[:_0_index]
	_result_suffix = sent[_0_index+12:]  # 12 is the length of <extra_id_0>

	end_token = '<extra_id_1>'
	exp_sequences_scores = torch.exp(outputs.sequences_scores)
	assert outputs.sequences.shape[0] == exp_sequences_scores.shape[0]
	for op, op_scr in zip(outputs.sequences[:10], exp_sequences_scores[:10]):
		result = mt5_tokenizer.decode(op[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)

		if end_token in result:
			_end_token_idx = result.index(end_token)
			result = result[:_end_token_idx]
		else:
			pass
		print(_result_prefix+'【'+result+'】'+_result_suffix+f";\tscore: {op_scr}")
	print("")