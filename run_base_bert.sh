
CAT=gender

# bert Gender
TYPE=slot_act_map
SUBJ=mixed_gender_bert
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_mixedgenderbert_occupationrev1_gendernoactlm_TRAIN
MODEL=bert-base-uncased

python3 -u -m lm.predict_base --gpuid 0 --transformer_type $MODEL --use_he_she 1 --input ${FILE}.source.json --output ./data/base_bert_gender_TRAIN.output.json

python3 analysis.py --cat ${CAT} --metrics subj_bias,pos_err,attr_err,model --input ./data/base_bert_gender_TRAIN.output.json --group_by gender_act | tee ./data/base_bert_${DATA}_TRAIN.log.txt


echo ">> Done"
######################################################################################################################
for DATA in country religion ethnicity; do
    echo ">> Predicting ${DATA}"
    TYPE=slot_act_map
    SUBJ=${DATA}_bert
    SLOT=${DATA}_noact_lm
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}_TRAIN
    python3 -u -m lm.predict_base --gpuid 0 --transformer_type $MODEL --use_he_she 0 --input ${FILE}.source.json --output ./data/bert_${DATA}_TRAIN.output.json

    python3 analysis.py --cat base_${MODEL_}${DATA} --metrics subj_bias,pos_err,attr_err,model --input ./data/bert_${DATA}_TRAIN.output.json --group_by subj | tee ./data/base_bert_${DATA}_TRAIN.log.txt

done
exit 0