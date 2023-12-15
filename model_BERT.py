import torch
from torch import nn
from torch.nn.functional import gelu
from transformers.modeling_outputs import MaskedLMOutput
# from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
import time
import math

class CustomBERTModel(nn.Module):
	def __init__(self, k, batch_size):
		super(CustomBERTModel, self).__init__()
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", return_token_type_ids=False)
		self.distilbert = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
		self.topk = k
		self.batch_size = batch_size
		# self.out = nn.Linear(30522, 30522)
		self.out = nn.Linear(self.topk, self.topk)
		nn.init.ones_(self.out.weight)
		# torch.nn.init.kaiming_uniform_(self.out.weight, a=math.sqrt(5))
		# self.out = nn.Sequential(
		# 			nn.Linear(self.topk, 256),
		# 			nn.ReLU(),
		# 			nn.Linear(256, 256),
		# 			nn.ReLU(),
		# 			nn.Linear(256, self.topk),
		# 			# nn.Sigmoid()
		# 		)
		# self.out.weight = nn.Parameter(torch.eye(self.topk))

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

		st_time = time.time()
		output = self.distilbert(input_ids)

		o_time = time.time()-st_time

		inputs = torch.zeros_like(output.logits)


		i_time = time.time()-st_time-o_time

		for i in range(len(output.logits)): # number of sentence
			masked_index = (input_ids[i] == self.tokenizer.mask_token_id).nonzero().item()
			j = masked_index
			logits = output.logits[i, j, :] #[sentence, word, probable token]

			# probs = logits.softmax(dim=0)
			# values, indices = probs.topk(self.topk)

			"""Feeding Logits to the model instead of the probabilities"""

			values, indices = logits.topk(self.topk)
			"""
			Adding the agent here to process the topk tokens 
			"""
			output_values  = self.out(values)
			# NOTE: Added later to make the negative values disappear from the last layer.
			layer_output = output_values.softmax(dim=0)
			# layer_output = output_values
			#Filling in the topk logits in the dictionary with the actual values
			for k in range(len(indices)):
				# inputs[i, j, indices[k]] = output.logits[i, j, indices[k]]
				inputs[i, j, indices[k]] = layer_output[k]
			# print("size of inputs: ", inputs.size())
		return inputs
		# return MaskedLMOutput(logits=inputs, hidden_states=output.hidden_states)