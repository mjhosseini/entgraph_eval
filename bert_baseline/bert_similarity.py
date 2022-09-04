from transformers import BertTokenizer
from transformers import BertModel
from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaModel
import torch
from torch.nn import CosineSimilarity
import time
import sys


class BertSimilarity_Calculator:
	def __init__(self, model_name):
		if model_name == 'bert':
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
			self.model = BertModel.from_pretrained('bert-base-chinese')
		elif model_name == 'xlm-roberta':
			self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
			self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
		else:
			print("Error: model name not recognized!", file=sys.stderr)
			raise AssertionError
		self.model_name = model_name
		self.cos = CosineSimilarity(dim=0)
		self.debug = False
		self.report_every = 100

	def calc_sim(self, sent1, sent2):
		sent1_tokenized = self.tokenizer.tokenize(sent1)
		sent1_tokenized = self.tokenizer.convert_tokens_to_ids(sent1_tokenized)
		sent1_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(sent1_tokenized)])
		sent2_tokenized = self.tokenizer.tokenize(sent2)
		sent2_tokenized = self.tokenizer.convert_tokens_to_ids(sent2_tokenized)
		sent2_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(sent2_tokenized)])
		sent1_repr = self.model(sent1_tokenized).pooler_output
		sent2_repr = self.model(sent2_tokenized).pooler_output
		sim = float(self.cos(sent1_repr[0], sent2_repr[0]))
		return (sim+1)/2  # to squeeze the sim score into the range of [0, 1]

	def calc_sim_file(self, fn):
		fp = open(fn, 'r', encoding='utf8')
		Y_dev_bert_sim = []
		lidx = 0
		st = time.time()
		for line in fp:
			if lidx % self.report_every == 0 and lidx > 0:
				ct = time.time()
				dur = ct - st
				dur_h = int(dur) // 3600
				dur_m = (int(dur) % 3600) // 60
				dur_s = int(dur) % 60
				print(
					f"Calculating {self.model_name} cosine similarity at index {lidx}; time lapsed: {dur_h} hours {dur_m} minutes {dur_s} seconds.")
			line = line.split('\t')
			sent1 = line[0].strip()
			sent2 = line[1].strip()
			if self.debug:
				print(sent1)
				print(sent2)
				print("")
				time.sleep(3)
			sim_score = self.calc_sim(sent1, sent2)
			Y_dev_bert_sim.append(sim_score)
			lidx += 1
		assert len(Y_dev_bert_sim) == lidx
		fp.close()
		return Y_dev_bert_sim

	def calc_sim_relative(self, sent1, sent2, sent1_masked, sent2_masked):
		sent1_tokenized = self.tokenizer.tokenize(sent1)
		sent1_tokenized = self.tokenizer.convert_tokens_to_ids(sent1_tokenized)
		sent1_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(sent1_tokenized)])
		sent2_tokenized = self.tokenizer.tokenize(sent2)
		sent2_tokenized = self.tokenizer.convert_tokens_to_ids(sent2_tokenized)
		sent2_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(sent2_tokenized)])
		sent1_masked_tokenized = self.tokenizer.tokenize(sent1_masked)
		sent1_masked_tokenized = self.tokenizer.convert_tokens_to_ids(sent1_masked_tokenized)
		sent1_masked_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(sent1_masked_tokenized)])
		# sent2_masked_tokenized = self.tokenizer.tokenize(sent2_masked)
		# sent2_masked_tokenized = self.tokenizer.convert_tokens_to_ids(sent2_masked_tokenized)
		# sent2_masked_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(sent2_masked_tokenized)])
		sent1_repr = self.model(sent1_tokenized).pooler_output
		sent2_repr = self.model(sent2_tokenized).pooler_output
		sent1_masked_repr = self.model(sent1_masked_tokenized).pooler_output
		# sent2_masked_repr = self.model(sent2_masked_tokenized).pooler_output
		sim_1_1m = self.cos(sent1_repr[0], sent1_masked_repr[0])
		# sim_2_2m = self.cos(sent2_repr[0], sent2_masked_repr[0])
		sim_1_2 = self.cos(sent1_repr[0], sent2_repr[0])
		rel_sim_1 = float(torch.sigmoid((1-sim_1_2)/(1-sim_1_1m)-1))  # is 2 a more similar option than [MASK] for 1
		# rel_sim_2 = float(torch.sigmoid((1-sim_1_2)/(1-sim_2_2m)-1))  # is 2 a more similar option than [MASK] for 1
		return rel_sim_1

	def calc_sim_relative_file(self, fn):
		fp = open(fn, 'r', encoding='utf8')
		Y_dev_bert_rel_sim = []
		lidx = 0
		st = time.time()
		for line in fp:
			if lidx % self.report_every == 0 and lidx > 0:
				ct = time.time()
				dur = ct - st
				dur_h = int(dur) // 3600
				dur_m = (int(dur) % 3600) // 60
				dur_s = int(dur) % 60
				print(
					f"Calculating {self.model_name} relative cosine similarity at index {lidx}; time lapsed: {dur_h} hours {dur_m} minutes {dur_s} seconds.")
			line = line.split('\t')
			assert len(line) >= 4
			sent1 = line[0].strip()
			sent2 = line[1].strip()
			sent1_masked = line[2].strip()
			sent2_masked = line[3].strip()
			if self.debug:
				print(sent1)
				print(sent2)
				print(sent1_masked)
				print(sent2_masked)
				print("")
				time.sleep(5)
			rel_sim = self.calc_sim_relative(sent1, sent2, sent1_masked, sent2_masked)
			Y_dev_bert_rel_sim.append(rel_sim)
			lidx += 1
		assert len(Y_dev_bert_rel_sim) == lidx
		fp.close()
		return Y_dev_bert_rel_sim


if __name__ == '__main__':
	calc = BertSimilarity_Calculator(model_name='bert')
	calc.calc_sim_file('../gfiles/chinese_ent/implications_dev_translated_raw.tsv')

