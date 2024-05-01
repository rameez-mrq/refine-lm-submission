import subprocess
import sys
import argparse
import os
import threading

import numpy as np
import torch
from torch._C import device
from torch.autograd import Variable
from torch import nn
from torch import cuda
import json

from tqdm import tqdm
from utils.holder import *
from utils.extract import get_tokenizer, tokenize_underspecified_input
from transformers import *
from templates.lists import Lists
import time
from model_LLM import CustomLLMModel


class FirstTenDict(dict):
    def __init__(self, pairs):
        super(FirstTenDict, self).__init__(pairs[:30])


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="./data/")
parser.add_argument('--input', help="Path to input file.", default="")
## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--topk', help="The the topk to consider", type=int, default=5)
# bert specs
parser.add_argument('--custom_model', help="1 for using custom model. Default: 0", default=0)
parser.add_argument('--transformer_type', help="The type of transformer encoder")
parser.add_argument('--custom_model_path', help="Path for pretrained custom model", default='')
#
parser.add_argument('--batch_size', help="The batch size used in batch mode, has to be >=2", type=int, default=64)
parser.add_argument('--output', help="The path to json format of output", default='')
parser.add_argument('--use_he_she', help="Whether to account for lm predictions on he/she", type=int, default=0)

parser.add_argument(
    "--prompt", help='Template for the LLM prompt', required=False, default=None)


def nvidia_smi_task():
    file_name = 'nvidia_smi_output.txt'
    with open(file_name, 'w') as file:
        file.write("--- START TRACKING ---")

    while True:
        with open(file_name, 'a') as file:
            subprocess.run(["nvidia-smi"], stdout=file)
        time.sleep(30)


# threading.Thread(target=nvidia_smi_task).start()

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


def load_input_preprocessed(path, prompt):
    def create_sentence(context, q, choices):
        return prompt.replace("[ans0]", choices[0]).replace("[ans1]", choices[1]).replace("[context]", context).replace(
            "[question]", q)

    source = []
    limit = 2
    with open(path, 'r') as f:
        # Load only the 10 first entries in the JSON file: easier to run
        # json_small = json.load(f, object_pairs_hook=FirstTenDict)
        json_data = json.load(f)
        for tid, key in enumerate(json_data.keys()):
            # if limit <=0:
            #    return source
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
                source.append(((tid, qid), (subj0_cluster, subj1_cluster), (subj0, subj1), cid, a_cluster, (att0, att1),
                               create_sentence(context, masked_sentence, choices), choices))
                limit -= 1
    return source


def load_input_simple(path):
    rs = []
    with open(path, 'r') as f:
        # json_data = json.load(f)
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
            rs.append(((subj0_cluster, subj1_cluster), (subj0, subj1), tid, a_cluster, (obj0, obj1), context, choices,
                       questions))
    return rs


def preprocess(source):
    # Integrate in load_input to avoid multiple iterations of all the input
    rs = []
    for i, (scluster, spair, tid, acluster, opair, context, choices, questions) in enumerate(source):
        for j, q in enumerate(questions):
            # i: number of the template
            # j: numer of the masked sentence in the template
            # show the masked sentence along with the context
            rs.append(((i, j), scluster, spair, tid, acluster, opair, context + ' ' + q, choices))
    return rs


def topk_tokens(opt, model, tokenizer, batch_seq, device, batch_size, batch_choices):
    tokenized_inputs = tokenizer(batch_seq, return_tensors="pt", padding=True, truncation=True,
                                 max_length=1024)

    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device)

    if isinstance(model, CustomLLMModel):
        batch_topk = model.forward(input_ids, attention_mask, batch_choices)
        for i in range(len(batch_topk)):
            for j in range(len(batch_topk[i])):
                # print(batch_topk[i][j])
                batch_topk[i][j] = (batch_topk[i][j][0], batch_topk[i][j][1].item())

        return batch_topk

    output = None

    # Pour fix les -inf,nan ou <0
    for i in range(100):
        try:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                output_scores=True,
                return_dict_in_generate=True,
                num_return_sequences=1,
            )
            break
        except RuntimeError as e:
            print(f"Erreur lors de la tentative {i + 1}: {e}")
            if i == 9:
                raise RuntimeError("Échec après plusieurs tentatives de génération.")

    resultat = []
    for j in range(input_ids.shape[0]):

        tokens_for_a1 = tokenizer.encode(batch_choices[j][0].lower(), add_special_tokens=False)
        tokens_for_a2 = tokenizer.encode(batch_choices[j][1].lower(), add_special_tokens=False)

        # generated_sequence = tokenizer.decode(output.sequences[j], skip_special_tokens=True)
        # print(f"Generated sequence {j + 1}: {generated_sequence}")

        logits = output.scores[0][j]

        val, idx = torch.topk(logits, opt.topk)
        val = val.softmax(dim=0)

        seq_res = []
        for i in range(opt.topk):
            proba = val[i].item()
            token = idx[i].item()
            word = tokenizer.decode([token])

            if any(char.isupper() for char in word):
                word = word.lower()
                token = tokenizer.encode(word, add_special_tokens=False)[0]

            if token in tokens_for_a1:
                word = batch_choices[j][0]
            elif token in tokens_for_a2:
                word = batch_choices[j][1]

            seq_res.append((word, proba))

        # print("--")
        resultat.append(seq_res)
    return resultat


def predict(opt, model, tokenizer, batch_seq, batch_choices, device, step_size):
    # retrieve the list of lists of tuples with the prediction and probability for the top k probabilities
    batch_topk = topk_tokens(opt, model, tokenizer, batch_seq, device, step_size, batch_choices)

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
        p_he = topk_p[topk_choice.index(
            'he')] if 'he' in topk_choice else 0.0  # TODO, should we aggregate the p(pronoun) with p(name) or just take max?
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

    if opt.custom_model:
        model = CustomLLMModel(opt.transformer_type, opt.topk, opt.batch_size)
        print("Loading model at : ", opt.custom_model_path)
        model.out.load_state_dict(torch.load("saved_models/" + opt.custom_model_path + ".pt", map_location="cuda:0"))

        tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, device_map="auto", local_files_only=True)
    else:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(opt.transformer_type, quantization_config=bnb_config,
                                                     device_map="auto", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, device_map="auto", local_files_only=True)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = 'left'

    source = load_input_preprocessed(opt.input, opt.prompt)

    print('start prediction...')
    cnt = 0
    batch_cnt = 0
    num_ex = len(source)
    rs_map = {}
    # Loop until the end of entries in the source (twice the size of input file: original sentence and negated sentence)
    while cnt < num_ex:
        print(f'{cnt}/{num_ex}')
        # Size of each step
        step_size = opt.batch_size if cnt + opt.batch_size < num_ex else num_ex - cnt
        # print(f'Size of step: {step_size}')
        # Assign the batch source
        batch_source = source[cnt:cnt + step_size]
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
        assert (len(batch_output) == step_size)

        for k in range(len(batch_output)):
            row_id, q_id = batch_idx[k]

            keys = '|'.join(
                [batch_scluster[k][0], batch_scluster[k][1], batch_spair[k][0], batch_spair[k][1], batch_tid[k],
                 batch_acluster[k], batch_opair[k][0], batch_opair[k][1]])
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
    rs_map = {keys: ex for sort_keys, keys, ex in ls}

    json.dump(rs_map, open(opt.output, 'w'), indent=4)
    print(opt.output)
    print('done')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
