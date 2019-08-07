# Summarization

This repository has been created to explore different possible models and techniques for the task of summarization.

## Current Models

- LSTM - classic neural sequence-to-sequence model with attention for abstractive summarization, set parameters to:
    - USE_TRANSFORMER = False
    - POINTER_GEN = False
- LSTM Pointer-Generator Model (from "Get to the Point: Summarization with Pointer-Generator Networks": https://arxiv.org/pdf/1704.04368.pdf), set parameters to:
    - USE_TRANSFORMER = False
    - POINTER_GEN = True
- Transformer (Same as LSTM except LSTM is replaced with Transformer), set parameters to:
    - USE_TRANSFORMER = True
    - POINTER_GEN = False
- Transformer Pointer-Generator Model (Same as LSTM Pointer-Generator Model except LSTM is replaced with Transformer), set parameters to:
    - USE_TRANSFORMER = True
    - POINTER_GEN = True

## Datasets

- Newsroom summarization dataset from Cornell: https://summari.es/
- CNN/Daily Mail summarization dataset, download processed version here: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

## Setup the repo

First run:

    pip install -r requirements.txt

Unfortunately, the spacy module and model has to be downloaded separately:

    pip install spacy
    python -m spacy download en_core_web_sm

## Configuring the model

In order to configure the model, edit the parameters at the top of the `train_model.py` script.  You can edit the default parameters in `parameters.py`.  Any parameters not appearing at the top of the `train_model.py` script will be set to the default parameters.

## Run Training

You can train the model, saving to a logfile as well, by running:

    python -u train_model.py 2>&1 | tee <log_file>

The checkpoints will be saved in the `CHECKPOINT_PATH` specified in `train_model.py`.

## Run Evaluation

You can produce the generated summary files by running:

    python eval_model.py

You can specify which checkpoints are to be evaluated using the list `sections` specified in `eval_model.py`, using the names of the corresponding checkpoint folders.  Inside each evaluated checkpoint folder, there will be 3 directories and a file: `articles`, `system`, and `reference`, and `rouge_scores.txt`.  These will contain the original articles, generated summaries, reference summaries, and rouge scores respectively.
