#!/bin/bash



topk=8
######################################################################################################################


cat=gender

# roberta Gender
model=roberta_L_${cat}_tk${topk}


# echo ">> Training model "${model}

python3 -m training_roberta --use_he_she 1 --epochs 1 --mini_batch_size 70 --batch_size 70 --topk ${topk} --lr 5e-5 --output ${model} --ppdata data/slotmap_mixedgenderroberta_occupationrev1_gendernoactlm_TRAIN.pkl

MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict --custom_model 1 --custom_model_path ${MODEL_PATH} --input slotmap_mixedgenderroberta_occupationrev1_gendernoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20


echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "gender_act" | tee ${LOG}

echo ">> Done"
######################################################################################################################

# topk=20
cat=ethnicity
#roberta Ethnicity
model=roberta_L_${cat}_tk${topk}


echo ">> Training model "${model}

python3 -m training_roberta --epochs 1 --mini_batch_size 10 --batch_size 10 --topk ${topk} --lr 5e-5 --output ${model} --ppdata data/slotmap_ethnicityroberta_biasedethnicity_ethnicitynoactlm_TRAIN.pkl
MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type roberta-base --input slotmap_ethnicityroberta_biasedethnicity_ethnicitynoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

######################################################################################################################
# topk=20
cat=nationality
#roberta Nationality
model=roberta_L_${cat}_tk${topk}


echo ">> Training model "${model}

python3 -m training_roberta --epochs 1 --mini_batch_size 70 --batch_size 140 --topk ${topk} --lr 5e-7 --output ${model} --ppdata data/slotmap_countryroberta_biasedcountry_countrynoactlm_TRAIN.pkl


MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type roberta-base --input slotmap_countryroberta_biasedcountry_countrynoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

######################################################################################################################
cat=religion
#roberta Religion

model=roberta_L_${cat}_tk${topk}

echo ">> Training model "${model}

python3 -m training_roberta --epochs 1 --mini_batch_size 20 --batch_size 20 --topk ${topk} --lr 5e-9 --output ${model} --ppdata data/slotmap_religionroberta_biasedreligion_religionnoactlm_TRAIN.pkl


MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type roberta-base --input slotmap_religionroberta_biasedreligion_religionnoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

exit 0


