import sys
import argparse
import h5py
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
import json
from utils.holder import *
from utils.extract import get_tokenizer, tokenize_underspecified_input
from transformers import *
from templates.lists import Lists
import time
# from model_optimised import CustomBERTModel
from model_BERT import CustomBERTModel

# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

transformer_type = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="./data/")
parser.add_argument('--input', help="Path to input file.", default="")
## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=0)
parser.add_argument('--topk', help="The the topk to consider", type=int, default=10)
# bert specs
parser.add_argument('--custom_model', help="1 for using custom model. Default: 0",  default=0)
parser.add_argument('--transformer_type', help="The type of transformer encoder",default = "roberta-base")
parser.add_argument('--custom_model_path', help="Path for pretrained custom model", default='')
#
parser.add_argument('--batch_size', help="The batch size used in batch mode, has to be >=2", type=int, default=20)
parser.add_argument('--output', help="The path to json format of output", default='')
parser.add_argument('--use_he_she', help="Whether to account for lm predictions on he/she", type=int, default=0)
#
# parser.add_argument('--start_ex', help="Starting point to load examples", type=int, default=0)
# parser.add_argument('--end_ex', help="Starting point to load examples", type=int, default=200000)


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



def load_input(path):
	rs = []
	with open(path, 'r') as f:
		json_data = json.load(f)
		for key, ex in json_data.items():
			context = ex['context'].strip()
			choices = [ex['q0']['ans0']['text'].strip(), ex['q0']['ans1']['text'].strip()]
			questions = [ex['q0']['question'].strip(), ex['q1']['question'].strip()]
			subj0_cluster, subj1_cluster, subj0, subj1, tid, a_cluster, obj0, obj1 = key.strip().split('|')
			rs.append(((subj0_cluster, subj1_cluster), (subj0, subj1), tid, a_cluster, (obj0, obj1), context, choices, questions))
	return rs


def preprocess(source):
	rs = []
	for i, (scluster, spair, tid, acluster, opair, context, choices, questions) in enumerate(source):
		for j, q in enumerate(questions):
			rs.append(((i,j), scluster, spair, tid, acluster, opair, context + ' ' + q, choices))
	return rs


# def load_mask_filler(opt):
# 	return pipeline('fill-mask', 
# 		model=AutoModelForMaskedLM.from_pretrained(opt.transformer_type), 
# 		tokenizer=AutoTokenizer.from_pretrained(opt.transformer_type),
# 		device=opt.gpuid)

def get_tokens(inputs, outputs, batch_size, tokenizer, k):
    results = []
    for i in range(batch_size):
        result = []
        input_ids = inputs["input_ids"][i]
        masked_index = (input_ids == tokenizer.mask_token_id).nonzero().item()
        # logits = outputs.logits[i, masked_index, :]
        """Changing for the logits outputs from the models"""
        logits = outputs[i, masked_index, :]
        # print("Logits: ", logits.topk(k))
        # probs = logits.softmax(dim=0)
        # values, indices = probs.topk(k)
        values, indices = logits.topk(k)
        # print("values: ", values, "indices: ", indices)
        leftover = 0 #NOTE: Filling in zeroes instead of leftover.
        probs = torch.full((len(logits),), leftover, dtype=values.dtype, device = device)
        # probs_res = torch.full((len(logits),), leftover, dtype=values.dtype, device = device)
        # print("Values: ", values, "Indices: ", indices)
        # probs, indices = activation_function(logits, k)
        probs_res = probs.scatter_(0, indices, values)
        values = torch.tensor([probs_res[i] for i in indices], requires_grad=True)
        for idx, p in zip(indices, values):
            tok = tokenizer.decode(idx).strip()
            tok = tok.replace(' ', '')

            result.append((tok, p))
        results += [result]

    if len(results) == 1:
        return results[0]
    return results

def predict(opt, topk, batch_seq, batch_choices, tokeniser, model):
    lists = Lists("word_lists", None)
    female, male = load_gender_names(lists)
    k=topk
    inputs = tokeniser(batch_seq, return_tensors='pt', padding=True).to(device)
    outputs = model(**inputs)

    # print("OUTPUTS: ", outputs)

    batch_topk = get_tokens(inputs, outputs, len(batch_seq), model.tokenizer, topk)  #ERROR: probs are being transffered as zero here
    # print("batch_topk: ", batch_topk)
    rs = []
    for topk, choices in zip(batch_topk, batch_choices):
        topk_choice = [p[0].strip().lower() for p in topk]
        topk_p = [p[1] for p in topk]
        choices = [p.lower() for p in choices]
        leftover_p = torch.zeros(1)[0].to(device)
        topk_p = torch.stack(topk_p).to(device) #Added later to make the code work on GPU.
        p_he = topk_p[topk_choice.index('he')] if 'he' in topk_choice else torch.zeros(1)[0].to(device)  # TODO, should we aggregate the p(pronoun) with p(name) or just take max?
        p_she = topk_p[topk_choice.index('she')] if 'she' in topk_choice else torch.zeros(1)[0].to(device)

        # print(" probs of he: ", p_he, " probs of she: ", p_she  )
        # rs.append([])
        for c in choices:
            # print(c)
            p_c = topk_p[topk_choice.index(c)] if c in topk_choice else leftover_p
            # print(p_c)
            # p_c = torch.stack(p_c)
            if opt.use_he_she == 1:
                if c in female:
                    # p_c = p_she + p_c
                    p_c = max(p_she, p_c)
                elif c in male:
                    # p_c = p_he + p_c
                    p_c = max(p_he, p_c)
                else:
                    raise Exception('unknown gender of {0}'.format(c))
            #rs[-1].append(p_c)
    #return rs
            # print("p_c updated: ", p_c)
            rs.append(p_c)

    # print(rs)
    rs_tensor = torch.stack(rs)
   
    return rs_tensor.view(-1,2)

def load_input(path):
	rs = []
	with open(path, 'r') as f:
		json_data = json.load(f)
		for key, ex in json_data.items():
			context = ex['context'].strip()
			choices = [ex['q0']['ans0']['text'].strip(), ex['q0']['ans1']['text'].strip()]
			questions = [ex['q0']['question'].strip(), ex['q1']['question'].strip()]
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




def main(args):
	# Get all arguments:
	# path to data dir, path to input file, GPU index, k for topk, is custom model, type of transformer, path for pretrained model, batch size, path to output, use of he/she
	opt = parser.parse_args(args)
	# dist.init_process_group("nccl")
	# rank = dist.get_rank()
	# Get an object with attributes subjects, activities, fillers, slots
	lists = Lists("word_lists", None)
	opt.input = opt.dir + opt.input
	# Get the lists of all female names and all male names

	# female = ["mary", "patricia", "linda", "barbara", "elizabeth"]#, "jennifer", "maria", "susan", "margaret", "dorothy"]
	# male = ["james", "john", "robert", "michael", "william"]#, "david", "richard", "charles", "joseph", "thomas"]
	opt.female, opt.male = load_gender_names(lists)
	
	# opt.female, opt.male = female, male
	topk = opt.topk

	# Choose the device
	
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
	# 	torch.cuda.manual_seed_all(1)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# build the mask filler
	# when applying the mask filler on a sentence with a [MASK], retrieves predictions for the mask
	# ex: print(mask_filler('John and Mary go to school. [MASK] is a good student.'))
	#mask_filler = load_mask_filler(opt)
	
	model_path = opt.custom_model_path
	model = torch.load(model_path)
	# Changing to load from dictionary

	# torch.set_flush_denormal(True)
	# model = CustomBERTModel(topk, 20)
	# model.load_state_dict(torch.load(model_path))

	model.to(device)


	
	# load source
	print('loading input...', flush=True)
	source = load_input(opt.input)
	# preprocess the source to have a better organisation
	source = preprocess(source)
	# Load the input and preprocess it at the same time (to reduce computation time)
	# start_ex = opt.start_ex
	# end_ex = opt.end_ex
	# source = load_input_preprocessed(opt.input)
	print("Total examples: ", len(source), flush=True)
	source = source
	print("Loaded examples: ", len(source), flush=True)
	print('start prediction...', flush=True)
	cnt = 0
	batch_cnt = 0
	num_ex = len(source)
	rs_map = {}
	# Loop until the end of entries in the source (twice the size of input file: original sentence and negated sentence)
	while cnt < num_ex:
		# print(f'{cnt}/{num_ex}', flush=True)
		# Size of each step
		step_size = opt.batch_size if cnt + opt.batch_size < num_ex else num_ex - cnt
		# print(f'Size of step: {step_size}', flush=True)

		start_time = time.time()
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
		#Preprocessing time
		# print(f'Preprocessing time: {time.time() - start_time}', flush=True)
		# Disable gradient calculation (backward never called)
		with torch.no_grad():
			# Retrieve a list with the probabilities of each name in batch_choices for each sequence in the batch
			batch_output = predict(opt, topk, batch_seq, batch_choices, model.tokenizer, model)
		# Time taken for the prediction
		# print(f'Prediction time: {time.time() - start_time}', flush=True)
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
				q_row[key] = {'text': batch_choices[k][z], 'start': p.item(), 'end': p.item()}

			rs_map[keys]['q{0}'.format(q_id)] = q_row

		cnt += step_size
		batch_cnt += 1
		# Time taken for a batch
		# print(f'Batch time: {time.time() - start_time}', flush=True)
		if batch_cnt % 1000 == 0:
			print("predicted {} examples".format(batch_cnt * opt.batch_size), flush=True)

	print('predicted {0} examples'.format(cnt), flush=True)

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
	# print(type(rs_map), flush=True)
	json.dump(rs_map, open(opt.output, 'w'), indent=4)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
	
	

# def main(args):
# 	opt = parser.parse_args(args)

# 	lists = Lists("word_lists", None)

# 	opt.input = opt.dir + opt.input
# 	opt.female, opt.male = load_gender_names(lists)

# 	if opt.gpuid != -1:
# 		torch.cuda.set_device(opt.gpuid)
# 		torch.cuda.manual_seed_all(1)

# 	# build model
# 	mask_filler = load_mask_filler(opt)

# 	# load source
# 	source = load_input(opt.input)
# 	source = preprocess(source)

# 	#
# 	print('start prediction...')
# 	cnt = 0
# 	batch_cnt = 0
# 	num_ex = len(source)
# 	rs_map = {}
# 	while cnt < num_ex:
# 		step_size = opt.batch_size if cnt + opt.batch_size < num_ex else num_ex - cnt
# 		batch_source = source[cnt:cnt+step_size]
# 		batch_seq = [row[6] for row in batch_source]
# 		batch_choices = [row[7] for row in batch_source]
# 		batch_idx = [row[0] for row in batch_source]
# 		batch_scluster = [row[1] for row in batch_source]
# 		batch_spair = [row[2] for row in batch_source]
# 		batch_tid = [row[3] for row in batch_source]
# 		batch_acluster = [row[4] for row in batch_source]
# 		batch_opair = [row[5] for row in batch_source]

# 		with torch.no_grad():
# 			batch_output = predict(opt, mask_filler, batch_seq, batch_choices)

# 		assert(len(batch_output) == step_size)

# 		for k in range(len(batch_output)):
# 			row_id, q_id = batch_idx[k]
			
# 			keys = '|'.join([batch_scluster[k][0], batch_scluster[k][1], batch_spair[k][0], batch_spair[k][1], batch_tid[k], batch_acluster[k], batch_opair[k][0], batch_opair[k][1]])
# 			if keys not in rs_map:
# 				rs_map[keys] = {}
# 				#rs_map[keys]['line'] = row_id
# 				rs_map[keys]['context'] = 'NA'

# 			q_row = {}
# 			q_row['question'] = batch_seq[k]
# 			q_row['pred'] = 'NA'

# 			for z, p in enumerate(batch_output[k]):
# 				key = 'ans{0}'.format(z)
# 				q_row[key] = {'text': batch_choices[k][z], 'start': p, 'end': p}
	
# 			rs_map[keys]['q{0}'.format(q_id)] = q_row

# 		cnt += step_size
# 		batch_cnt += 1

# 		if batch_cnt % 10 == 0:
# 			print("predicted {} examples".format(batch_cnt * opt.batch_size))

# 	print('predicted {0} examples'.format(cnt))

# 	# organize a bit
# 	ls = []
# 	for keys, ex in rs_map.items():
# 		toks = keys.split('|') 
# 		sort_keys = sorted(toks[0:3])
# 		sort_keys.extend(toks[3:])
# 		sort_keys = '|'.join(sort_keys)
# 		ls.append((sort_keys, keys, ex))
# 	ls = sorted(ls, key=lambda x: x[0])
# 	rs_map = {keys:ex for sort_keys, keys, ex in ls}

# 	json.dump(rs_map, open(opt.output, 'w'), indent=4)


# if __name__ == '__main__':
# 	sys.exit(main(sys.argv[1:]))
