#!/bin/bash


cd /home/rqureshi/refine-lm/refine-code
source /home/rqureshi/.bashrc
source activate redubias

topk=8
echo "Topk: "${topk}
#Bert Gender

cat=gender

model=distilbert_o_${cat}_tk${topk}

echo ">> Training model "${model}



python3 -m training_distilbert --use_he_she 1 --epochs 1 --mini_batch_size 70 --batch_size 70 --topk ${topk} --lr 5e-9 --output ${model} --ppdata data/slotmap_mixedgenderbert_occupationrev1_gendernoactlm_TRAIN.pkl

MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_distilbert --custom_model 1 --custom_model_path ${MODEL_PATH} --input slotmap_mixedgenderbert_occupationrev1_gendernoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20


# python3 -u -m lm.predict_optimised_new --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type distilbert-base-uncased --input slotmap_mixedgenderbert_occupationrev1_gendernoactlm.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "gender_act" | tee ${LOG}

echo ">> Done"


# Ethnicity

cat=ethnicity
model=distilbert_o_${cat}_tk${topk}

echo ">> Training model "${model}

python3 -m training_distilbert --epochs 1 --mini_batch_size 10 --batch_size 10 --topk ${topk} --lr 5e-7 --output ${model} --ppdata data/slotmap_ethnicitybert_biasedethnicity_ethnicitynoactlm_TRAIN_LOOSE.pkl
MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}_LOOSE.output.json
LOG=./data/log/${model}_LOOSE.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_distilbert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type distilbert-base-uncased --input slotmap_ethnicitybert_biasedethnicity_ethnicitynoactlm_TEST_LOOSE.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"


#Religion

cat=religion
# topk=5
model=distilbert_o_${cat}_tk${topk}


echo ">> Training model "${model}

python3 -m training_distilbert --epochs 1 --mini_batch_size 20 --batch_size 20 --topk ${topk} --lr 5e-7 --output ${model} --ppdata data/slotmap_religionbert_biasedreligion_religionnoactlm_LOOSE_TRAIN.pkl
MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}_LOOSE.output.json
LOG=./data/log/${model}_LOOSE.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_distilbert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type distilbert-base-uncased --input slotmap_religionbert_biasedreligion_religionnoactlm_LOOSE_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

# Nationality

# topk=20
cat=nationality
model=distilbert_o_${cat}_tk${topk}



echo ">> Training model "${model}

python3 -m training_distilbert --epochs 1 --mini_batch_size 70 --batch_size 140 --topk ${topk} --lr 5e-9 --output ${model} --ppdata data/slotmap_countrybert_biasedcountry_countrynoactlm_TRAIN.pkl
MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}_LOOSE.output.json
LOG=./data/log/${model}_LOOSE.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_distilbert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type distilbert-base-uncased --input slotmap_countrybert_biasedcountry_countrynoactlm_TEST.source.json --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --cat ${model} --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "subj" | tee ${LOG}

echo ">> Done"

exit 0






exit 0
