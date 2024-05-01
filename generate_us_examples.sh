# BERT and DistilBERT

TYPE=slot_act_map
SUBJ=mixed_gender_roberta_test
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
MODEL=distilbert-base-uncased
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
--subj $SUBJ --act $ACT --slot $SLOT \
--output ./data/${FILE}.source.json


TYPE=slot_act_map
SUBJ=mixed_gender_roberta_train
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
MODEL=distilbert-base-uncased
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
--subj $SUBJ --act $ACT --slot $SLOT \
--output ./data/${FILE}.source.json



# Other categories

for DATA in country religion ethnicity; do
    TYPE=slot_act_map
    SUBJ=${DATA}_bert
    SLOT=${DATA}_noact_lm-TEST
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
    --subj $SUBJ --act $ACT --slot $SLOT \
    --output ./data/${FILE}.source.json
done


for DATA in country religion ethnicity; do
    TYPE=slot_act_map
    SUBJ=${DATA}_bert
    SLOT=${DATA}_noact_lm-TRAIN
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
    --subj $SUBJ --act $ACT --slot $SLOT \
    --output ./data/${FILE}.source.json
done

###################################################################
# RoBERTa
#Gender

TYPE=slot_act_map
SUBJ=mixed_gender_roberta_test
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
MODEL=distilbert-base-uncased
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
--subj $SUBJ --act $ACT --slot $SLOT \
--output ./data/${FILE}.source.json

TYPE=slot_act_map
SUBJ=mixed_gender_roberta_train
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
MODEL=distilbert-base-uncased
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
--subj $SUBJ --act $ACT --slot $SLOT \
--output ./data/${FILE}.source.json



# Other categories

for DATA in country religion ethnicity; do
    TYPE=slot_act_map
    SUBJ=${DATA}_roberta
    SLOT=${DATA}_noact_lm-TEST
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
    --subj $SUBJ --act $ACT --slot $SLOT \
    --output ./data/${FILE}.source.json
done


for DATA in country religion ethnicity; do
    TYPE=slot_act_map
    SUBJ=${DATA}_roberta
    SLOT=${DATA}_noact_lm-TRAIN
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
    --subj $SUBJ --act $ACT --slot $SLOT \
    --output ./data/${FILE}.source.json
done