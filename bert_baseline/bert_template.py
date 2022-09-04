from transformers import BertTokenizer
from transformers import BertForNextSentencePrediction
import torch
from torch.nn import CosineSimilarity
from torch.nn.functional import softmax
import time


class BertTemplate_Calculator:
	def __init__(self):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
		self.model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
		self.cos = CosineSimilarity(dim=0)
		self.debug = False
		self.report_every = 100

	def calc_sim(self, sent1, sent2):
		sent_template_tokenized = self.tokenizer.tokenize(sent1+'。因此'+sent2)  # XXXXX。因此，XXXXXXX。
		sent_template_tokenized = self.tokenizer.convert_tokens_to_ids(sent_template_tokenized)
		sent_template_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(sent_template_tokenized)])
		template_output = self.model(sent_template_tokenized)  # optionally, label=XXX
		logits = template_output.logits
		logits = softmax(logits, dim=1)
		loss = template_output.loss
		return float(logits[0, 0])  # softmaxed likelihood of '因此，'+sent2 to be the next sentence of sent1

	def calc_sim_file(self, fn):
		fp = open(fn, 'r', encoding='utf8')
		Y_dev_bert_temp_sim = []
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
					f"Calculating template similarity at index {lidx}; time lapsed: {dur_h} hours {dur_m} minutes {dur_s} seconds.")
			line = line.split('\t')
			sent1 = line[0].strip()
			sent2 = line[1].strip()
			if self.debug:
				print(sent1)
				print(sent2)
				print("")
				time.sleep(3)
			sim_score = self.calc_sim(sent1, sent2)
			Y_dev_bert_temp_sim.append(sim_score)
			lidx += 1
		assert len(Y_dev_bert_temp_sim) == lidx
		fp.close()
		return Y_dev_bert_temp_sim

	def calc_sim_relative(self, sent1, sent2, sent1_masked, sent2_masked):
		template_1_2_tokenized = self.tokenizer.tokenize(sent1+'，因此，'+sent2)
		template_1_2_tokenized = self.tokenizer.convert_tokens_to_ids(template_1_2_tokenized)
		template_1_2_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(template_1_2_tokenized)])
		template_1_1m_tokenized = self.tokenizer.tokenize(sent1+'，因此，'+sent1_masked)
		template_1_1m_tokenized = self.tokenizer.convert_tokens_to_ids(template_1_1m_tokenized)
		template_1_1m_tokenized = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(template_1_1m_tokenized)])

		template_1_2_output = self.model(template_1_2_tokenized)
		template_1_1m_output = self.model(template_1_1m_tokenized)
		logits_1_2 = softmax(template_1_2_output.logits, dim=1)
		logits_1_1m = softmax(template_1_1m_output.logits, dim=1)
		sim_1_2 = float(logits_1_2[0,0])
		sim_1_1m = float(logits_1_1m[0,0])
		rel_sim_1 = float(torch.sigmoid(torch.tensor((sim_1_2-sim_1_1m)*5)))  # is 2 a more similar option than [MASK] for 1
		return rel_sim_1

	def calc_sim_relative_file(self, fp):
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
					f"Calculating relative template similarity at index {lidx}; time lapsed: {dur_h} hours {dur_m} minutes {dur_s} seconds.")
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
		return Y_dev_bert_rel_sim
