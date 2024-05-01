import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from model_LLM import CustomLLMModel


def generate_sentence(obj):
    story = obj['story']
    question = obj['question']
    answer_options = obj['answer_options']

    return f"""TASK : ANSWER WITH A B C OR D

Story :Paul planted four types of flowers in his garden: roses, tulips, daisies, and sunflowers. The daisies bloomed first.
Question: Which type of flower bloomed first in Paul's garden?
Choices:
A: Roses
B: Tulips
C: Daisies
D: Sunflowers
Answer: C

Story: Mia has five different colored pens: red, blue, green, black, and purple. She uses the blue pen to write her notes because it's her favorite color.
Question: What color pen does Mia use to write her notes?
Choices:
A: Red
B: Blue
C: Green
D: Black
Answer: B

Story: Tom visited the zoo and saw many animals, but he was most excited to see the elephants and giraffes. He took many pictures of the giraffes.
Question: Which animal did Tom take many pictures of at the zoo?
Choices:
A: Lions
B: Tigers
C: Elephants
D: Giraffes
Answer: D

Story: Emily read three books last week: a mystery, a romance, and a science fiction novel. She enjoyed the science fiction novel the most.
Question: Which type of book did Emily enjoy the most last week?
Choices:
A: Mystery
B: Romance
C: Science Fiction
D: History
Answer: C

Story: {story}
Question: {question}
Choices:
A : {answer_options['A']}
B : {answer_options['B']}
C : {answer_options['C']}
D : {answer_options['D']}
Answer :""".replace("\\newline", "")


def test(model, resultat_file):
    good = {
        1: 0,
        3: 0,
        5: 0
    }

    loop = tqdm(batch_seq)
    for i in loop:
        prompt = i['prompt']
        answer = i['answer']

        tokenized_inputs = tokenizer(prompt, return_tensors="pt")

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        with torch.no_grad():
            output = model.forward(input_ids, attention_mask, None)[0]
            output = sorted(output, key=lambda x: x[1].item(), reverse=True)

            for topk_eval in [1, 3, 5]:
                # resultat_file.write(f"{topk_eval},")
                model_answers = []

                for j in range(topk_eval):
                    model_answers.append(output[j][0].lower())

                if answer.lower() in model_answers:
                    good[topk_eval] += 1
    for z in good.keys():
        s = f"{z},{good[z] / len(batch_seq)}"
        print(s)
        resultat_file.write(s + "\n")


if __name__ == '__main__':
    print("Start ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATASET
    dataset = load_dataset("sagnikrayc/mctest", trust_remote_code=True)

    batch_seq = []
    for i in tqdm(range(len(dataset['test'])), desc="Preparing data"):
        batch_seq.append({
            "prompt": generate_sentence(dataset['test'][i]),
            "answer": dataset['test'][i]['answer'],
            "answer_options": dataset['test'][i]['answer_options']
        })

    data = {
        # Llama 7b
        "meta-llama/Llama-2-7b-hf": [
            "./saved_models/LLM_llama7b_religion_out_softend",
            "./saved_models/LLM_llama7b_ethnicity_out_softend",
            "./saved_models/LLM_llama7b_gender_out_softend",
            "./saved_models/LLM_llama7b_country_out_softend"
        ],

        # Llama 7b chat
        "meta-llama/Llama-2-7b-chat-hf": [
            "./saved_models/LLM_llama7b-chat_religion_out_softend",
            "./saved_models/LLM_llama7b-chat_ethnicity_out_softend",
            "./saved_models/LLM_llama7b-chat_gender_out_softend",
            "./saved_models/LLM_llama7b-chat_country_out_softend"
        ],

        # Llama 7b instruct
        "codellama/CodeLlama-7b-Instruct-hf": [
            "./saved_models/LLM_llama7b-instruct_religion_out_softend",
            "/saved_models/LLM_llama7b-instruct_ethnicity_out_softend",
            "./saved_models/LLM_llama7b-instruct_gender_out_softend",
            "./saved_models/LLM_llama7b-instruct_country_out_softend"
        ],

        # Mistral 7b
        "mistralai/Mistral-7B-v0.1": [
            "./saved_models/LLM_mistral7b_religion_out_softend",
            "./saved_models/LLM_mistral7b_ethnicity_out_softend",
            "/saved_models/LLM_mistral7b_gender_out_softend",
            "./saved_models/LLM_mistral7b_country_out_softend"
        ],

        # Llama 13b
        "meta-llama/Llama-2-13b-hf": [
            "./saved_models/LLM_llama13b_religion_out_softend",
            "./saved_models/LLM_llama13b_ethnicity_out_softend"
            "./saved_models/LLM_llama13b_gender_out_softend",
            "./saved_models/LLM_llama13b_country_out_softend",
        ],

        # Llama 13b chat
        "meta-llama/Llama-2-13b-chat-hf": [
            "./saved_models/LLM_llama13b-chat_religion_out_softend",
            "./saved_models/LLM_llama13b-chat_ethnicity_out_softend",
            "./saved_models/LLM_llama13b-chat_gender_out_softend",
            "./saved_models/LLM_llama13b-chat_country_out_softend"
        ]
    }
    resultat_file = open("resultat.txt", "w")
    k = 10
    for model_name in data.keys():
        resultat_file.write(f"Doing: {model_name}\n")

        # TOKENIZER
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", local_files_only=True)

        # BASIC MODEL
        basic_model = CustomLLMModel(model_name, k, None, False)
        basic_model.eval()

        test(basic_model, resultat_file)

        resultat_file.write(f"Now doing the finetuned models: \n")
        for custom_model_path in data[model_name]:
            resultat_file.write(f"Using : {custom_model_path} \n")

            print("Loading model at : ", custom_model_path)
            finetuned_model = CustomLLMModel(model_name, k, None, True)
            finetuned_model.out.load_state_dict(torch.load(custom_model_path + ".pt", map_location="cuda:0"))
            finetuned_model.eval()

            resultat_file.write(f"\nFine-tuned: \n")
            test(finetuned_model, resultat_file)
            resultat_file.flush()

        resultat_file.write(f"\n\n")

    resultat_file.close()