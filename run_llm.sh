cat=$1
MODEL=$2
MODEL_ALIAS=$3
PROMPT=$4
use_he_she=$5

LOG=./data/log/${MODEL_ALIAS}/${cat}
mkdir -p $LOG

set -eu

# START BASE LLM
echo "Predict  base LLM"
mkdir -p ./data/${MODEL_ALIAS}
python3 -u -m lm.predict_llm --transformer_type $MODEL --use_he_she $use_he_she --prompt "$PROMPT" \
--input llm_${cat}-TEST.source.json --output ./data/${MODEL_ALIAS}/llm_no_train_${cat}.output.json --batch_size 64 --topk 10

echo "Analysing base LLM"
python3 analysis.py --metrics subj_bias,pos_err,attr_err,model --input ./data/${MODEL_ALIAS}/llm_no_train_${cat}.output.json --group_by "subj" 2>&1 | tee ${LOG}/analyse_base.txt
#END BASE LLM

# START FINE TUNING
echo "Preprocess data for fine tuning"
python3 preprocess_data.py --input_path ./data/llm_${cat}-TRAIN.source.json --output ./data/llm_${cat}-TRAIN.source.pkl --prompt "$PROMPT"

echo "Start fine tuning LLM"
NEW_MODEL=LLM_${MODEL_ALIAS}_${cat}_out_softend
python3 training_llm.py --llm_name $MODEL --epochs 1 --device 0 --batch_size 256 --mini_batch_size 128 \
--lr 2e-5 --topk 10 --output $NEW_MODEL --ppdata ./data/llm_${cat}-TRAIN.source.pkl 2>&1 | tee ${LOG}/training.txt

python3 -u -m lm.predict_llm --custom_model 1 --custom_model_path ${NEW_MODEL} --transformer_type $MODEL --use_he_she $use_he_she --prompt "$PROMPT" \
--input llm_${cat}-TEST.source.json --output ./data/${MODEL_ALIAS}/llm_finetuned_${cat}.output.json --batch_size 126 --topk 10

python3 analysis.py --metrics subj_bias,pos_err,attr_err,model --input ./data/${MODEL_ALIAS}/llm_finetuned_${cat}.output.json --group_by "subj" 2>&1 | tee ${LOG}/analyse_llm_finetuned.txt
# END FINE TUNING

echo "End ${cat}"
exit 0


