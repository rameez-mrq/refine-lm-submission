import sys
import argparse
import os
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch import cuda

import torch.nn.functional as F
from torch import FloatTensor

import json
from utils.holder import *
from utils.extract import get_tokenizer, tokenize_underspecified_input
from transformers import *
import math
from redubias.calc_bias import Dataset, calculate_reward, calculate_batch_manhattan, collate_fn, unqover_reward
# from redubias.predict_topk import predict_answers
from redubias.predict_topk import predict_answers
# from model import CustomBERTModel
from model_LLM import CustomLLMModel
import _pickle as pickle
from time import process_time
import random
from templates.lists import Lists

gpuid = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--mini_batch_size', type=int, default=20, help="Size of batch per update")
    parser.add_argument('--batch_size', type=int, default=70, help="Samples per Template")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning Rate")
    parser.add_argument('--topk', type=int, default=10, help="TopK")
    parser.add_argument('--output', type=str, default="new_model", help="Name of the model")
    parser.add_argument('--ppdata', type=str, default="", help="Path of the preprocessed data")
    parser.add_argument('--use_he_she', help="Whether to account for lm predictions on he/she", type=int, default=0)
    parser.add_argument('--llm_name', help="Name of the LLM model to use", type=str)

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    ppdata_path = args.ppdata
    ## Loading Pre-processed Data

    lists = Lists("word_lists", None)

    from transformers import set_seed

    set_seed(0)

    with open(ppdata_path, 'rb') as file:
        pp_data = pickle.load(file)

    keys = list(pp_data.keys())
    values = list(pp_data.values())
    values = [pp_data[k] for k in keys]
    batch_size = args.batch_size
    training_values = Dataset(values)
    training_generator = torch.utils.data.DataLoader(training_values, batch_size=batch_size, collate_fn=collate_fn,
                                                     num_workers=2)

    #######VARIABLES########
    mini_batch_size = args.mini_batch_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    topk = args.topk
    name = args.output
    """Defining Model"""
    print("Defining Model", flush=True)
    # NOTE: Insert the name of the model here
    model = CustomLLMModel(args.llm_name, topk, batch_size)

    for layer_name, param in model.llm.named_parameters():
        param.requires_grad = False

    print("Number of Samples: ", len(training_generator), flush=True)
    print(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(training_generator)
    print("Number of training steps : ", num_training_steps, flush=True)
    step = 0
    model.train()
    rewards = []
    print("Starting training")
    start = process_time()
    cp_path = 'data/logs/training_logs/' + name + '.pt'
    print('Name:', name)
    finished = False
    for epoch in range(num_epochs):
        if finished:
            break
        # Training
        for local_batch in training_generator:
            #if process_time() - start > 68400:
            #    print("Arrêt après 19 heures d'exécution")
            #    finished = True
            #    break

            if process_time() - start > 18000:
                print("Arrêt après 5 heures d'exécution")
                finished = True
                break

            with torch.cuda.device(0):
                # Transfer to GPU
                loss, reward = calculate_reward(args, local_batch, mini_batch_size, topk, model.tokenizer, model)
                if step % 25 == 0:
                    model_path = 'saved_models/' + name + '_out_old.pt'
                    torch.save(model.out.state_dict(), model_path)
                print("Step ", step, "| Loss:  ", loss.item(), "| Reward: ", reward, flush=True)

                if reward is not None and not torch.isnan(loss):
                    loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                rewards.append(reward)

                step += 1

    stop = process_time()
    print("time elapsed: ", stop - start)
    # Save model
    model_path = 'saved_models/' + name + '.pt'
    torch.save(model.out.state_dict(), model_path)


if __name__ == '__main__':
    main()
