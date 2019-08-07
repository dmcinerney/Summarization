# Summarization

This repository has been created to explore different possible models and techniques for the task of summarization.

## Current Models

- LSTM - classic neural sequence-to-sequence model with attention for abstractive summarization
- LSTM Pointer-Generator Model (from "Get to the Point: Summarization with Pointer-Generator Networks": https://arxiv.org/pdf/1704.04368.pdf)
- Transformer (Same as LSTM except LSTM is replaced with Transformer)
- Transformer Pointer-Generator Model (Same as LSTM Pointer-Generator Model except LSTM is replaced with Transformer)

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

In order to configure the model, edit the parameters at the top of the train_model.py script.  You can edit the default parameters in parameters.py.  Any parameters not appearing at the top of the train_model.py script will be set to the default parameters.

## Run Training

You can train the model, saving to a logfile as well, by running:

    python -u train_model.py 2>&1 | tee <log_file>

## Run Evaluation

You can produce the generated summary files by running:

    python eval_model.py
