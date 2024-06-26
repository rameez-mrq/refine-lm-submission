# Refine-LM

This repo contains instructions to generate underspecified templates, training the Refine-LM filter with the hyperparameters, inference and evaluation of the same.

---

Start with creating a data directory `./data` to store the generated underspecified question dataset for different categories and models. Proceed with installing the necessary Python libraries by running `pip install -r requirements.txt`

## Generating Underspecified question

Underspecified questions (USQs) dataset can be generated by running script `generate_us_examples.sh` or `generate_llm_example.sh`. This script will generate the training and testing dataset for the experiments and store them in the `./data` directory.

## Preprocessing

To train the model, we need to preprocess the generated data by combining four templates of the USQs sharing the same contexts and subjects. Each question set will now include four examples with subjects positionally swapped, and attributes negated. To preprocess the data for training purposes, run the following:

`python preprocess_data.py --input_path {usq_json} --output {processed_pickle.pkl}`

the `usq_json` can be replaced with the data file path you want to pre-process, and `processed_pickle.pkl` with the path of the file used for training in the next step.

## Training & evaluation

For training and evaluation, an example script is given below.

```[shell]
topk=8
cat=gender

Bert Gender
model=bert_o_${cat}_tk${topk}


echo ">> Training model "${model}

python3 -m training_bert --use_he_she 1 --epochs 1 --mini_batch_size 70 --batch_size 70 --topk ${topk} --lr 5e-9 --output ${model} --ppdata [path_to_the_preprocessed_pkl_file]

MODEL_PATH=saved_models/${model}
OUTPUT=data/output/${model}.output.json
LOG=./data/log/${model}.log.txt

echo ">> Predicting model "${model}

python3 -u -m lm.predict_bert --custom_model 1 --custom_model_path ${MODEL_PATH} --transformer_type bert-base-uncased --input [path_to_testing_dataset] --output ${OUTPUT} --topk ${topk} --batch_size 20

echo ">> Aggregating model "${model}

python3 analysis.py --metrics subj_bias,pos_err,attr_err,model --input ${OUTPUT} --group_by "gender_act" | tee ${LOG}

echo ">> Done"
```
Replace the paths of input for training and testing files.

For ease of access, we have also included simple scripts, namely `run_bert.sh`, `run_distil_bert.sh`,  `run_roberta.sh` and files like `./Llama-7b/run_llm_religion.sh` for LLMS, respectively, with all the hyperparameters used for reproducing the results reported in the paper. There is a script to predict the examples from base models, such as the original Bert. Script `run_base_berts.sh` can be used for that purpose.

## Results

Logs of the output can be found in `./data/logs`. Create a subdirectory for that if required.


## References

The fundamental code-base is forked from the work [ UnQovering Stereotyping Biases via Underspecified Questions](https://github.com/allenai/unqover).

