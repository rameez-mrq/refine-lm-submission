#!/bin/bash
# Ce script génère les templates llm pour les différents catégories.

for cat in country ethnicity religion; do
    echo "Start ${cat}"

    TYPE="slot_act_map"
    SUBJ="${cat}_bert"
    ACT="biased_${cat}"

    SLOT="${cat}_noact_lm-TRAIN"
    python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
    --subj $SUBJ --act $ACT --slot $SLOT --lm_mask blank \
    --output ./data/llm_${cat}-TRAIN.source.json

    SLOT="${cat}_noact_lm-TEST"
    python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
    --subj $SUBJ --act $ACT --slot $SLOT --lm_mask blank \
    --output ./data/llm_${cat}-TEST.source.json
done


#Gender
cat=gender
echo "Start ${cat}"

TYPE=slot_act_map
SLOT=gender_noact_lm
ACT=occupation_rev1

echo "Generate training set"
SUBJ=mixed_gender_roberta_train
python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
--subj $SUBJ --act $ACT --slot $SLOT --lm_mask blank \
--output ./data/llm_${cat}-TRAIN.source.json

echo "Generate test set"
SUBJ=mixed_gender_roberta_test
python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
--subj $SUBJ --act $ACT --slot $SLOT --lm_mask blank \
--output ./data/llm_${cat}-TEST.source.json

echo "Done all"