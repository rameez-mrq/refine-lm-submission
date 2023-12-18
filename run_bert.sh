#!/bin/bash


topk=8
cat=gender

Bert Gender
model=bert_o_${cat}_tk${topk}


echo ">> Training model "${model}

python3 -m training_bert --use_he_she 1 --epochs 1 --mini_batch_size 70 --batch_size 70 --topk ${topk} --lr 5e-9 --output ${model} --ppdata data/slotmap_mixedgenderbert_occupationrev1_gendernoactlm_TRAIN.pkl

MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_bert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type bert-base-uncased --input slotmap_mixedgenderbert_occupationrev1_gendernoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "gender_act" | tee ${LOG}

echo ">> Done"
# ##########################

cat=ethnicity
#Bert Ethnicity
model=bert_o_${cat}_tk${topk}


echo ">> Training model "${model}

python3 -m training_bert --epochs 1 --mini_batch_size 10 --batch_size 10 --topk ${topk} --lr 5e-7 --output ${model} --ppdata data/slotmap_ethnicitybert_biasedethnicity_ethnicitynoactlm_TRAIN_LOOSE.pkl
MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}_LOOSE.output.json
LOG=./data/log/${model}_LOOSE.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_bert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type bert-base-uncased --input slotmap_ethnicitybert_biasedethnicity_ethnicitynoactlm_TEST_LOOSE.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

##########################

cat=nationality
#Bert Nationality
model=bert_o_${cat}_tk${topk}


echo ">> Training model "${model}

python3 -m training_bert --epochs 1 --mini_batch_size 70 --batch_size 140 --topk ${topk} --lr 5e-9 --output ${model} --ppdata data/slotmap_countrybert_biasedcountry_countrynoactlm_TRAIN.pkl


MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_bert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type bert-base-uncased --input slotmap_countrybert_biasedcountry_countrynoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

##########################
cat=religion
#Bert Religion

model=bert_o_${cat}_tk${topk}

echo ">> Training model "${model}

python3 -m training_bert --epochs 1 --mini_batch_size 20 --batch_size 20 --topk ${topk} --lr 5e-7 --output ${model} --ppdata data/slotmap_religionbert_biasedreligion_religionnoactlm_LOOSE_TRAIN.pkl


MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_bert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type bert-base-uncased --input slotmap_religionbert_biasedreligion_religionnoactlm_LOOSE_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

exit 0