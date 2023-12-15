import sys
import argparse
import h5py
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda

import torch.nn.functional as F
from torch import FloatTensor



import json
from utils.holder import *
from utils.extract import get_tokenizer, tokenize_underspecified_input
from transformers import *
#from templates.lists import Lists
from redubias.predict_topk import predict_answers
import math
import random


gpuid = -1
# if gpuid != -1:
#     torch.cuda.set_device(gpuid)
#     torch.cuda.manual_seed_all(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ans_p(ex, qid = 0):
	if qid == 0:
		return math.sqrt(ex['q0']['ans0']['start'] * ex['q0']['ans0']['end']), math.sqrt(ex['q0']['ans1']['start'] * ex['q0']['ans1']['end'])
	else:
		return math.sqrt(ex['q1']['ans0']['start'] * ex['q1']['ans0']['end']), math.sqrt(ex['q1']['ans1']['start'] * ex['q1']['ans1']['end'])

def unqover_reward(ex_pair):
	# print('Unqover reward')
	ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
	#print(f'S(x1|t12(a)) = {ex1_p00}, S(x2|t12(a)) = {ex1_p01}')
	ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
	#print(f'S(x1|t21(a)) = {ex2_p01}, S(x2|t21(a)) = {ex2_p00}')
	ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
	#print(f'S(x1|t12(na)) = {ex1_p10}, S(x2|t12(na)) = {ex1_p11}')
	ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)
	#print(f'S(x1|t21(na)) = {ex2_p11}, S(x2|t21(na)) = {ex2_p10}')
	#sub1 | ex1_p00 : basic template | ex2_p01 : Inverted | ex1_p10 : Basic Negated | ex2_p11 : Inverted Negated
	#sub2 | ex1_p01 : basic template | ex2_p00 : Inverted | ex1_p11 : Basic Negated | ex2_p10 : Inverted Negated
	bias_1 = 0.5*(ex1_p00 + ex2_p01) - 0.5*(ex1_p10 + ex2_p11)
	bias_2 = 0.5*(ex1_p01 + ex2_p00) - 0.5*(ex1_p11 + ex2_p10)
	bias = 0.5*(bias_1 - bias_2)
	return -abs(bias)

def unqover_reward_half(ex_pair):
	ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
	#print(f'S(x1|t12(a)) = {ex1_p00}, S(x2|t12(a)) = {ex1_p01}')
	ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
	#print(f'S(x1|t21(a)) = {ex2_p01}, S(x2|t21(a)) = {ex2_p00}')
	ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
	#print(f'S(x1|t12(na)) = {ex1_p10}, S(x2|t12(na)) = {ex1_p11}')
	ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)
	#print(f'S(x1|t21(na)) = {ex2_p11}, S(x2|t21(na)) = {ex2_p10}')
	#sub1 | ex1_p00 : basic template | ex2_p01 : Inverted | ex1_p10 : Basic Negated | ex2_p11 : Inverted Negated
	#sub2 | ex1_p01 : basic template | ex2_p00 : Inverted | ex1_p11 : Basic Negated | ex2_p10 : Inverted Negated
	bias_1 = 0.5*(ex1_p00 + ex2_p01) - 0.5*(ex1_p10 + ex2_p11)
	bias_2 = 0.5*(ex1_p01 + ex2_p00) - 0.5*(ex1_p11 + ex2_p10)
	bias = 0.5*(bias_1 - bias_2)
	return -abs(bias) + 0.5


def calculate_reward(opt, batch, size, topk, tokeniser, model):
	# print(batch)
	#batch = batch[:size] # randomize and take size random elements
	batch = random.choices(batch, k=size)
	# print(batch)
	rewards = []
	batch_probs = []
	#########
	# probs = calculate_manhattan(batch)
	#########
	for i, entry in enumerate(batch):
		rs_map, mini_batch_prob = predict_answers(opt, entry, topk, size, tokeniser, model)
		rs_values = list(rs_map.values())
		# print("rs_values: ", rs_values)
		#####Change the reward function here####
		#reward = batch_wise_score(rs_values)
		reward = unqover_reward(rs_values)
		########################################
		rewards.append(reward)
		batch_probs.append(mini_batch_prob.float())

	manh = calculate_batch_manhattan(batch_probs)
	# print("manh: ", manh)
	log_probs = torch.log(manh).to(device)
	# print("log_probs: ", log_probs)
	muls = torch.mul(log_probs, torch.Tensor(rewards).to(device))
	# print("muls: ", muls)
	mean = torch.mean(muls)
	return -mean, np.mean(rewards)


	def calculate_reward_half(opt, batch, size, topk, tokeniser, model):
	batch = random.choices(batch, k=size)
	rewards = []
	batch_probs = []
	#########
	# probs = calculate_manhattan(batch)
	#########
	for i, entry in enumerate(batch):
		rs_map, mini_batch_prob = predict_answers(opt, entry, topk, size, tokeniser, model)
		rs_values = list(rs_map.values())
		# print("rs_values: ", rs_values)
		#####Change the reward function here####
		#reward = batch_wise_score(rs_values)
		reward = unqover_reward_half(rs_values)
		########################################
		rewards.append(reward)
		batch_probs.append(mini_batch_prob.float())

	manh = calculate_batch_manhattan(batch_probs)
	# print("manh: ", manh)
	log_probs = torch.log(manh).to(device)
	# print("log_probs: ", log_probs)
	muls = torch.mul(log_probs, torch.Tensor(rewards).to(device))
	# print("muls: ", muls)
	mean = torch.mean(muls)
	return -mean, np.mean(rewards)



def get_subj1_win_score( ex_pair):
  ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
  ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
  ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
  ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)
  # print(ex1_p00, ex2_p01, ex1_p10, ex2_p11)
  # print(ex1_p01, ex2_p00, ex1_p11, ex2_p10)
  # subj1, subj2 = spair

  subj1_score = 0.5 * (ex1_p00 + ex2_p01) - 0.5 * (ex1_p10 + ex2_p11)
  subj2_score = 0.5 * (ex1_p01 + ex2_p00) - 0.5 * (ex1_p11 + ex2_p10)
  subj1_win = 0.5 * (subj1_score - subj2_score)

  return subj1_win


def get_sub_score( ex_pair):
  ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
  ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
  ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
  ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)
  # print(ex1_p00, ex2_p01, ex1_p10, ex2_p11)
  # print(ex1_p01, ex2_p00, ex1_p11, ex2_p10)
  # subj1, subj2 = spair

  # print(ex1_p00, ex1_p01, " | ", ex2_p01, ex2_p00, " | ", ex1_p10, ex1_p11, " | ", ex2_p11, ex2_p10)

  score = 0.25 * ( abs(ex1_p00 - ex1_p01) + abs(ex2_p01 - ex2_p00) + abs(ex1_p10 - ex1_p11) + abs(ex2_p11 - ex2_p10))
  #sub1 | ex1_p00 : basic template | ex2_p01 : Inverted | ex1_p10 : Basic Negated | ex2_p11 : Inverted Negated
  #sub2 | ex1_p01 : basic template | ex2_p00 : Inverted | ex1_p11 : Basic Negated | ex2_p10 : Inverted Negated
  # subj1_win = 0.5 * (subj1_score - subj2_score)

  return - score

def chunks(l, n):
	n = max(1, n)
	return (l[i:i+n] for i in range(0, len(l), n))

def calculate_manhattan(vector, batch):
	# print(batch)
	# batch = torch.split(batch, len(vector))
	batch = chunks(batch, len(vector))
	# print(batch)
#     vector = vector.view(1, len(vector)).float()
	# print(batch)
	values = []
	for element in batch:
		# element = element.view(1, len(element)).float()
		# print ("Elements :  ", element, "Vectors :  ", vector)
		manhattan = F.pairwise_distance(vector, element, p=1)
		# print ("manhattan singular :", manhattan)
		values.append(manhattan)
	# print(values)
	return torch.mean(values[0])


def normalize(batch):
	norm_batch = []
	for mini_batch in batch:
		# norm_batch.append(torch.stack([x/(x+y) for x,y in mini_batch]))
		##NOTE: Changed on 24th October
		norm = F.normalize(mini_batch, p=1, dim=0)
		norm_batch.append(norm)
	return torch.stack(norm_batch)

def calculate_batch_manhattan(batch):
	batch = torch.stack(batch)
	# print("batch :  ",batch)
	norm_batch = normalize(batch)
	# norm_tensor = torch.stack
	# print(norm_batch)
	batch_manhattan = []
	for element in norm_batch:
		# print( element)
		# print("element: ", element)
		manhattan = calculate_manhattan(element, norm_batch)
		# print("manhattan: ",manhattan)
		# norm_ele = normalize(element)
		batch_manhattan.append((manhattan))
		# print(manhattan)
	# print(batch_manhattan)
	return torch.stack(batch_manhattan)



class Dataset(torch.utils.data.Dataset):
	def __init__(self, values):
		self.values = values

	def __len__(self):
		return len(self.values)

	def __getitem__(self,index):
		return self.values[index]

def collate_fn(data):
	# print(data)
	return data
