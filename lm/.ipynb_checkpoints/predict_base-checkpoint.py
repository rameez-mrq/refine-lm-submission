
import sys
import argparse
import h5py
import os
import numpy as np
import torch
from torch._C import device
from torch.autograd import Variable
from torch import nn
from torch import cuda
import json
from utils.holder import *
from utils.extract import get_tokenizer, tokenize_underspecified_input
from transformers import *
from templates.lists import Lists
import time


class FirstTenDict(dict):
	def __init__(self, pairs):
		super(FirstTenDict, self).__init__(pairs[:30])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="./data/")
parser.add_argument('--input', help="Path to input file.", default="")
## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--topk', help="The the topk to consider", type=int, default=10)
# bert specs
parser.add_argument('--custom_model', help="1 for using custom model. Default: 0",  default=0)
parser.add_argument('--transformer_type', help="The type of transformer encoder",default = "roberta-base")
parser.add_argument('--custom_model_path', help="Path for pretrained custom model", default='')
#
parser.add_argument('--batch_size', help="The batch size used in batch mode, has to be >=2", type=int, default=100)
parser.add_argument('--output', help="The path to json format of output", default='')
parser.add_argument('--use_he_she', help="Whether to account for lm predictions on he/she", type=int, default=0)


def load_gender_names(lists):
	female = []
	for k, ls in lists.subjects.items():
		if k.startswith('female'):
			female.extend([p['[subj]'] for p in ls])
	female = list(set(female))
	female = [p.lower() for p in female]

	male = []
	for k, ls in lists.subjects.items():
		if k.startswith('male'):
			male.extend([p['[subj]'] for p in ls])
	male = list(set(male))
	male = [p.lower() for p in male]
	return female, male

def load_input_preprocessed(path):
	source = []
	with open(path, 'r') as f:
		# Load only the 10 first entries in the JSON file: easier to run
		#json_small = json.load(f, object_pairs_hook=FirstTenDict)
		json_data = json.load(f)
		for tid, key in enumerate(json_data.keys()):
			# Retrieve the value for the key
			ex = json_data[key]
			# The scene with the event including two subjects
			context = ex['context'].strip()
			# choices: possible accepted answers (ie subjects)
			choices = [ex['q0']['ans0']['text'].strip(), ex['q0']['ans1']['text'].strip()]
			# questions: list of 2 masked sentences (original sentence + negated sentence)
			questions = [ex['q0']['question'].strip(), ex['q1']['question'].strip()]
			# subj0_cluster, subj1_cluster: female or male
			# subj0, subj1: name of the subjects
			# cid: ID of the context (ex: "Mary got off the flight to visit James." = "0"; "Mary lives in the same city with James." = "1")
			# a_cluster: always 'None' in our input file
			# att0, attt1: attribute (property on which the bias is measured = occupation)
			subj0_cluster, subj1_cluster, subj0, subj1, cid, a_cluster, att0, att1 = key.strip().split('|')
			for qid, masked_sentence in enumerate(questions):
				# tid: number of the template
				# qid: numer of the masked sentence in the template
				# show the masked sentence along with the context
				source.append(((tid, qid), (subj0_cluster, subj1_cluster), (subj0, subj1), cid, a_cluster, (att0, att1), context + ' ' + masked_sentence, choices))
	return source

def load_input_simple(path):
	rs = []
	with open(path, 'r') as f:
		#json_data = json.load(f)
		# Load only the 10 first entries in the JSON file: easier to run
		json_small = json.load(f, object_pairs_hook=FirstTenDict)
		for key, ex in json_small.items():
			# The scene with the event including two subjects
			context = ex['context'].strip()
			# choices: possible accepted answers (ie subjects)
			choices = [ex['q0']['ans0']['text'].strip(), ex['q0']['ans1']['text'].strip()]
			# questions: list of 2 masked sentences (original sentence + negated sentence)
			questions = [ex['q0']['question'].strip(), ex['q1']['question'].strip()]
			# subj0_cluster, subj1_cluster: female or male
			# subj0, subj1: name of the subjects
			# tid: ID of the context (ex: "Mary got off the flight to visit James." = "0"; "Mary lives in the same city with James." = "1")
			# a_cluster: always 'None' in our input file
			# obj0, obj1: attribute (property on which the bias is measured = occupation)
			subj0_cluster, subj1_cluster, subj0, subj1, tid, a_cluster, obj0, obj1 = key.strip().split('|')
			rs.append(((subj0_cluster, subj1_cluster), (subj0, subj1), tid, a_cluster, (obj0, obj1), context, choices, questions))
	return rs


def preprocess(source):
	# Integrate in load_input to avoid multiple iterations of all the input
	rs = []
	for i, (scluster, spair, tid, acluster, opair, context, choices, questions) in enumerate(source):
		for j, q in enumerate(questions):
			# i: number of the template
			# j: numer of the masked sentence in the template
			# show the masked sentence along with the context
			rs.append(((i,j), scluster, spair, tid, acluster, opair, context + ' ' + q, choices))
	return rs


def load_mask_filler(opt):
	if opt.custom_model == 0:
		print("Loading Model: ", opt.transformer_type)
		path = opt.transformer_type
	else:
		print("Loading Custom Model at: ", opt.custom_model_path)
		path = opt.custom_model_path
	# Load the model using path if custom model, otherwise type of transformer
	# Automatically retrieve the relevant model given the path
	return pipeline('fill-mask',
		model=AutoModelForMaskedLM.from_pretrained(str(path)),
		tokenizer=AutoTokenizer.from_pretrained(opt.transformer_type),
		device=opt.gpuid)

# modified from FillMaskPipeline.__call__ function
def topk_tokens(opt, model, tokenizer, batch_seq, device, step_size):
	st_time = time.time()
	inputs = tokenizer(batch_seq, return_tensors="pt", padding=True).to(device)
	# print('tokenize time: {0}'.format(time.time() - st_time))
	# applies the model with keyword arguments input_ids and attention_mask
	outputs = model(**inputs)
	# retrieve the tensor
	outputs = outputs.logits
	# print('model time: {0}'.format(time.time() - st_time))
	results = []
	# BUG, to check
	# retrieve the size for dim 0 of each entry in the output (padded) = length of the longest tokenized sequence of batch_seq
	#batch_size = outputs[0].size(0)
	batch_size = step_size

	for i in range(batch_size):
		# Retrieve the tokenized i-th sentence (two ways of doing it)
		#input_ids = inputs.input_ids[i]
		input_ids = inputs["input_ids"][i]

		result = []

		# nonzero: returns a tensor containing the indices of all non-zero elements of input
		# checks that there is a [MASK] in the sentence.
		if torch.nonzero(input_ids == tokenizer.mask_token_id).numel() != 1:
			# print(batch_seq[i])
			assert(False)

		# retrieve the index where the [MASK] is
		masked_index = (input_ids == tokenizer.mask_token_id).nonzero().item()
		# get the tensor asociated to the [MASK] token in the sentence
		# same as logits = outputs[i][masked_index][:]
		logits = outputs[i, masked_index, :]

		# compute the softmax to convert into probabilities
		probs = logits.softmax(dim=0)
		# retrieve the k highest probabilities and their indices
		values, predictions = probs.topk(opt.topk)

		# iterate over the k highest probabilities and their indices
		for idx, p in zip(predictions.tolist(), values.tolist()):
			# decode the prediction to get the associated word (ex: james)
			tok = tokenizer.decode(idx).strip()

			# this is a buggy behavior of bert tokenizer's decoder
			#	Note this also applies to distilbert
			if 'bert-base-uncased' in opt.transformer_type or 'bert-large-uncased' in opt.transformer_type:
				tok = tok.replace(' ', '')

			result.append((tok, p))

		# append the list of topk predictions to results
		results += [result]
	# print('total time: {0}'.format(time.time() - st_time))
	# if batch_size is 1, no need to keep a list of lists
	if len(results) == 1:
		return results[0]
	return results


def predict(opt, model, tokenizer, batch_seq, batch_choices, device, step_size):
	# retrieve the list of lists of tuples with the prediction and probability for the top k probabilities
	batch_topk = topk_tokens(opt, model, tokenizer, batch_seq, device, step_size)

	rs = []
	# iterate through the topk predictions
	# (different sizes lists: zip stops when end of topk reached)
	for topk, choices in zip(batch_topk, batch_choices):
		# retrieve all predicted tokens and lower them to compare with possible choices in batch_choices
		topk_choice = [p[0].strip().lower() for p in topk]
		# retrieve all probabilities
		topk_p = [p[1] for p in topk]
		# retrieve accepted answers and lower them to compare with topk
		choices = [p.lower() for p in choices]
		leftover_p = 0.0

		# retrieve the probabilities of he and she to take pronouns into account
		p_he = topk_p[topk_choice.index('he')] if 'he' in topk_choice else 0.0	# TODO, should we aggregate the p(pronoun) with p(name) or just take max?
		p_she = topk_p[topk_choice.index('she')] if 'she' in topk_choice else 0.0

		rs.append([])
		# iterate over all accepted names as answers (ie the ones that appear in the context)
		for c in choices:
			# retrieve the probability of the name we are examining
			p_c = topk_p[topk_choice.index(c)] if c in topk_choice else leftover_p

			# if we take pronouns into account, take the maximun probability for the name or the pronoun
			if opt.use_he_she == 1:
				if c in opt.female:
					p_c = max(p_she, p_c)
				elif c in opt.male:
					p_c = max(p_he, p_c)
				else:
					raise Exception('unknown gender of {0}'.format(c))
			# append the probability of the name
			rs[-1].append(p_c)
	return rs


def main(args):
	# Get all arguments:
	# path to data dir, path to input file, GPU index, k for topk, is custom model, type of transformer, path for pretrained model, batch size, path to output, use of he/she
	opt = parser.parse_args(args)
	# Get an object with attributes subjects, activities, fillers, slots
	lists = Lists("word_lists", None)
	opt.input = opt.dir + opt.input
	# Get the lists of all female names and all male names
	opt.female, opt.male = load_gender_names(lists)

	# Choose the device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# build the mask filler
	# when applying the mask filler on a sentence with a [MASK], retrieves predictions for the mask
	# ex: print(mask_filler('John and Mary go to school. [MASK] is a good student.'))
	mask_filler = load_mask_filler(opt)

	# This has already been done in load_mask_filler, why two models and two tokenizers?
	# model = mask_filler.model to get a variable called model
	# tokenizer = mask_filler.tokenizer to get a variable called tokenizer
	'''if opt.custom_model == 0:
		print("Loading Model: ", opt.transformer_type)
		path = opt.transformer_type
	else:
		print("Loading Custom Model at: ", opt.custom_model_path)
		path = opt.custom_model_path
	model=AutoModelForMaskedLM.from_pretrained(str(path)).to(device)
	tokenizer=AutoTokenizer.from_pretrained(opt.transformer_type)'''
	model = mask_filler.model
	tokenizer = mask_filler.tokenizer

	# load source
	#source = load_input_simple(opt.input)
	# preprocess the source to have a better organisation
	#source = preprocess(source)
	# Load the input and preprocess it at the same time (to reduce computation time)
	source = load_input_preprocessed(opt.input)

	print('start prediction...')
	cnt = 0
	batch_cnt = 0
	num_ex = len(source)
	rs_map = {}
	# Loop until the end of entries in the source (twice the size of input file: original sentence and negated sentence)
	while cnt < num_ex:
		# print(f'{cnt}/{num_ex}')
		# Size of each step
		step_size = opt.batch_size if cnt + opt.batch_size < num_ex else num_ex - cnt
		# print(f'Size of step: {step_size}')
		# Assign the batch source
		batch_source = source[cnt:cnt+step_size]
		# context + masked sentence associated for each entry
		batch_seq = [row[6] for row in batch_source]
		# possible accepted answers for each entry
		batch_choices = [row[7] for row in batch_source]
		# Indexes of template and question for each entry
		batch_idx = [row[0] for row in batch_source]
		# tuple of subject genders for each entry
		batch_scluster = [row[1] for row in batch_source]
		# name of the subjects for each entry
		batch_spair = [row[2] for row in batch_source]
		# ID of the context for each entry
		batch_tid = [row[3] for row in batch_source]
		# a_cluster (always None in our case) for each entry
		batch_acluster = [row[4] for row in batch_source]
		# attribute (property on which the bias is measured = occupation) for each entry
		batch_opair = [row[5] for row in batch_source]

		# Disable gradient calculation (backward never called)
		with torch.no_grad():
			# Retrieve a list with the probabilities of each name in batch_choices for each sequence in the batch
			batch_output = predict(opt, model, tokenizer, batch_seq, batch_choices, device, step_size)

		# Check if we have predictions for all sequences
		assert(len(batch_output) == step_size)

		for k in range(len(batch_output)):
			row_id, q_id = batch_idx[k]

			keys = '|'.join([batch_scluster[k][0], batch_scluster[k][1], batch_spair[k][0], batch_spair[k][1], batch_tid[k], batch_acluster[k], batch_opair[k][0], batch_opair[k][1]])
			if keys not in rs_map:
				rs_map[keys] = {}
				#rs_map[keys]['line'] = row_id
				rs_map[keys]['context'] = 'NA'

			q_row = {}
			q_row['question'] = batch_seq[k]
			q_row['pred'] = 'NA'

			for z, p in enumerate(batch_output[k]):
				key = 'ans{0}'.format(z)
				q_row[key] = {'text': batch_choices[k][z], 'start': p, 'end': p}

			rs_map[keys]['q{0}'.format(q_id)] = q_row

		cnt += step_size
		batch_cnt += 1

		if batch_cnt % 1000 == 0:
			print("predicted {} examples".format(batch_cnt * opt.batch_size))

	print('predicted {0} examples'.format(cnt))

	# organize a bit
	ls = []
	for keys, ex in rs_map.items():
		toks = keys.split('|')
		sort_keys = sorted(toks[0:3])
		sort_keys.extend(toks[3:])
		sort_keys = '|'.join(sort_keys)
		ls.append((sort_keys, keys, ex))
	ls = sorted(ls, key=lambda x: x[0])
	rs_map = {keys:ex for sort_keys, keys, ex in ls}

	json.dump(rs_map, open(opt.output, 'w'), indent=4)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
