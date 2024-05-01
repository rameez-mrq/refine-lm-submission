import sys
import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.functional import gelu
from torch import cuda
import json
from utils.holder import *
from utils.extract import get_tokenizer, tokenize_underspecified_input
from transformers import *
from templates.lists import Lists
from model_LLM import CustomLLMModel
from torch.nn.functional import normalize

import random


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


#####custom settings#######
gpuid = -1
if gpuid != -1:
    torch.cuda.set_device(gpuid)
    # torch.cuda.manual_seed_all(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#################


"""Function to get Top K Predictions"""


# def activation_function(logits, k):
#     # st_time = time.time()
#     values, indices = logits.topk(k)
#     s = torch.sum(logits)
#     # print(f'Time to get sum of logits: {time.time() - st_time}', flush=True)
#     s1 = torch.sum(values)
#     leftover = 0 #NOTE: Filling in zeroes instead of leftover.
#     # print("leftover: ", leftover)
#     ### NOTE: Most expensive function!!!
#     # probs_slow = torch.tensor([logit*0.9/s1 if logit in values else leftover for logit in logits], requires_grad=True).to(device)
#     ### FIX: Using Scatter Add function
#     # print("Devices logits: ", logits.device)
#     probs = torch.full((len(logits),), leftover, dtype=values.dtype, device = device)
#     # values_modified = torch.mul(0.9/s1, values) #NOTE: Normalising the topk values
#     values = normalize(values, p=1.0, dim=0).to(torch.device("cuda"))
#     probs_res = probs.scatter_(0, indices, values)

#     return probs_res, indices

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
        leftover = 0  # NOTE: Filling in zeroes instead of leftover.
        probs = torch.full((len(logits),), leftover, dtype=values.dtype, device=device)
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


"""Functions to predict and arrange answers"""


def predict(opt, topk, batch_seq, batch_choices, tokeniser, model):
    lists = Lists("word_lists", None)
    female, male = load_gender_names(lists)
    k = topk

    if isinstance(model, CustomLLMModel):
        tokenized_inputs = tokeniser(batch_seq, return_tensors="pt", padding=True, truncation=True,
                                     max_length=1024)

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        batch_topk = model.forward(input_ids, attention_mask, batch_choices)

    else:
        inputs = tokeniser(batch_seq, return_tensors='pt', padding=True).to(device)
        outputs = model(**inputs)
        batch_topk = get_tokens(inputs, outputs, len(batch_seq), model.tokenizer,
                                topk)  # ERROR: probs are being transffered as zero here

    rs = []
    # print("batch topk : ",batch_topk)
    for topk, choices in zip(batch_topk, batch_choices):
        # print("top k ", topk)

        topk_choice = [p[0].strip().lower() for p in topk]
        topk_p = [p[1] for p in topk]
        choices = [p.lower() for p in choices]
        leftover_p = torch.zeros(1)[0].to(device)
        topk_p = torch.stack(topk_p).to(device)  # Added later to make the code work on GPU.
        p_he = topk_p[topk_choice.index('he')] if 'he' in topk_choice else torch.zeros(1)[0].to(
            device)  # TODO, should we aggregate the p(pronoun) with p(name) or just take max?
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
            # rs[-1].append(p_c)
            # return rs
            # print("p_c updated: ", p_c)
            rs.append(p_c)

    # print(rs)
    rs_tensor = torch.stack(rs)

    # print(rs_tensor.view(-1,2))

    return rs_tensor.view(-1, 2)


"""Function for predicting Answers and data post processing"""


def predict_answers(opt, source, topk, batch_size, tokeniser, model):
    #   mask_filler = load_mask_filler(transformer_type)
    cnt = 0
    batch_cnt = 0
    num_ex = len(source)
    rs_map = {}

    while cnt < num_ex:
        step_size = batch_size if cnt + batch_size < num_ex else num_ex - cnt
        batch_source = source
        batch_seq = [row[6] for row in batch_source]
        batch_choices = [row[7] for row in batch_source]
        batch_idx = [row[0] for row in batch_source]
        batch_scluster = [row[1] for row in batch_source]
        batch_spair = [row[2] for row in batch_source]
        batch_tid = [row[3] for row in batch_source]
        batch_acluster = [row[4] for row in batch_source]
        batch_opair = [row[5] for row in batch_source]

        # with torch.no_grad():
        batch_output = predict(opt, topk, batch_seq, batch_choices, tokeniser, model)
        if batch_output is None:
            return None
        # print("Batch output: ", batch_output)

        assert (len(batch_output) == step_size)

        for k in range(len(batch_output)):
            row_id, q_id = batch_idx[k]

            keys = '|'.join(
                [batch_scluster[k][0], batch_scluster[k][1], batch_spair[k][0], batch_spair[k][1], batch_tid[k],
                 batch_acluster[k], batch_opair[k][0], batch_opair[k][1]])
            # print(keys)
            if keys not in rs_map:
                rs_map[keys] = {}
                # rs_map[keys]['line'] = row_id
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
        return rs_map, batch_output
