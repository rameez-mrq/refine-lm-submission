#!/bin/bash

source  ../../../../venv_llama2/bin/activate

echo "Start eval"
srun python eval_llm.py
