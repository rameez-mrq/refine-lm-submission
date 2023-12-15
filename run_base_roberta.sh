

CAT=gender

# roberta Gender
TYPE=slot_act_map
SUBJ=mixed_gender_roberta
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_mixedgenderroberta_occupationrev1_gendernoactlm_TEST
MODEL=roberta-base

python3 -u -m lm.predict_base --gpuid 0 --transformer_type $MODEL --use_he_she 1 --input ${FILE}.source.json --output ./data/roberta_gender.output.json

python3 analysis.py --cat ${CAT} --metrics subj_bias,pos_err,attr_err,model --input ./data/roberta_gender.output.json --group_by gender_act | tee ./data/roberta_${DATA}.log.txt


echo ">> Done"
######################################################################################################################
for DATA in country religion ethnicity; do
    echo ">> Predicting ${DATA}"
    TYPE=slot_act_map
    SUBJ=${DATA}_roberta
    SLOT=${DATA}_noact_lm
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}_TEST
    python3 -u -m lm.predict_base --gpuid 0 --transformer_type $MODEL --use_he_she 0 --input ${FILE}.source.json --output ./data/roberta_${DATA}.output.json

    python3 analysis.py --cat ${MODEL_}${DATA} --metrics subj_bias,pos_err,attr_err,model --input ./data/roberta_${DATA}.output.json --group_by subj | tee ./data/roberta_${DATA}.log.txt

done
# exit 0


