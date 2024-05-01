import torch
from torch import nn
from torch.nn.functional import gelu
from transformers.modeling_outputs import MaskedLMOutput
# from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import math


class IdentityLayer(nn.Module):
    def __init__(self, size):
        super(IdentityLayer, self).__init__()
        self.size = size
        self.out = nn.Linear(self.size, self.size).to('cuda')

        self.out.weight.data = torch.eye(self.size).to('cuda')
        self.out.bias.data = torch.zeros(self.size).to('cuda')

    def forward(self, x):
        return self.out(x)


class CustomLLMModel(nn.Module):
    def __init__(self, llm_name, k, batch_size, use_out=True):
        super(CustomLLMModel, self).__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("Use llm : ", llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto", quantization_config=bnb_config,
                                                        local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, device_map='auto', local_files_only=True)
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = 'left'

        print(k)
        self.topk = k
        self.batch_size = batch_size
        self.out = nn.Linear(self.topk, self.topk).to("cuda:0")
        nn.init.ones_(self.out.weight)

        self.USE_OUT = use_out

    def forward(self, input_ids=None, attention_mask=None, batch_choices=None):
        output = None

        # Pour fix les -inf,nan ou <0
        for i in range(100):
            try:
                output = self.llm.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_return_sequences=1,
                )
                break
            except RuntimeError as e:
                print(f"Erreur lors de la tentative {i + 1}: {e}")
                if i == 99:
                    raise RuntimeError("Échec après plusieurs tentatives de génération.")

        resultat = []
        for j in range(input_ids.shape[0]):

            if batch_choices is not None:
                tokens_for_a1 = self.tokenizer.encode(batch_choices[j][0].lower(), add_special_tokens=False)
                tokens_for_a2 = self.tokenizer.encode(batch_choices[j][1].lower(), add_special_tokens=False)

            # generated_sequence = self.tokenizer.decode(output.sequences[j], skip_special_tokens=True)
            # print(f"Generated sequence {j + 1}: {generated_sequence}")

            logits = output.scores[0][j]
            val, idx = torch.topk(logits, self.topk)
            val[val == float('-inf')] = -1e10

            val = val.softmax(dim=0).to("cuda:0")
            if self.USE_OUT:
                val = self.out(val)
                val = val.softmax(dim=0)
            seq_res = []
            for i in range(self.topk):
                proba = val[i]
                token = idx[i].item()
                word = self.tokenizer.decode([token])

                if any(char.isupper() for char in word):
                    word = word.lower()
                    token = self.tokenizer.encode(word, add_special_tokens=False)[0]

                if batch_choices is not None:
                    if token in tokens_for_a1:
                        word = batch_choices[j][0]
                    elif token in tokens_for_a2:
                        word = batch_choices[j][1]

                seq_res.append((word, proba))

            # print("--")
            resultat.append(seq_res)
        # print(resultat)

        return resultat
