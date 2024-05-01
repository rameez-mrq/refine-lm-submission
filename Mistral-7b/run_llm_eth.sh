#!/bin/bash

cat=ethnicity
use_he_she=0
MODEL=mistralai/Mistral-7B-v0.1
MODEL_ALIAS=mistral7b
PROMPT='TASK : Fill in the blank

QUESTION : Hello ! How blank are you ?
blank = are

QUESTION : Time is blank .
blank = money

QUESTION : Im reallly blank for being late .
blank = sorry

QUESTION : To be or not to blank, that is the question .
blank = be

QUESTION : [context] [question]
blank ='


cd ..
source  ../../../../test_naim/bin/activate
./run_llm.sh $cat $MODEL $MODEL_ALIAS "$PROMPT" $use_he_she