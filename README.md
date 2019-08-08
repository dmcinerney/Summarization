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

In order to configure the model, edit the parameters at the top of the `train_model.py` and `eval_model.py` scripts.  You can edit the default parameters in `parameters.py`.  Any parameters not appearing at the top of the `train_model.py` and `eval_model.py` scripts will be set to the default parameters.

**Sections:** At the beginning of the `train_model.py` there is a list of parameter dictionaries called `sections`.  Sections allows the user to do 2 things: save the checkpoint folder at specific intervals during training and alter parameters throughout training.  Every section needs to change the parameter `max_training_steps`.  There are certain parameters that cannot be changed, such as any that change the number of parameters in the model, but it is usefull, for example, for adding in coverage or changing the maximum length of the input text or summary.  The default sections list saves a checkpoint every 10000 iterations, add coverage for the last 3000 iterations, and start with max text and summary lengths of 100 and 50 respectively, bumping up to 400 and 100 after the first 50000 iterations.

## Run Training

In order to train the model, you need to make an empty directory to hold the checkpoints, for instance:

    mkdir path/to/checkpoints

and then create an empty main checkpoint folder.  (Checkpoints will be saved by copying this main checkpoint folder at intervals during training).  IMPORTANT: this must be named "`checkpoint`":

    mkdir path/to/checkpoints/checkpoint

Then make sure the parameter CHECKPOINT_PATH at the beginning of your `train_model.py` file points to your checkpoints directory path.  (In this case, you would set `CHECKPOINT_PATH="path/to/checkpoints"`.)  You can then train the model, saving to a logfile as well, by running:

    python -u train_model.py 2>&1 | tee <log_file>

The checkpoints will be saved in the `CHECKPOINT_PATH` specified in `train_model.py`.  If you interrupt any training run, you can resume it simply by rerunning the above command as long as the `CHECKPOINT_PATH` is set to the right run's checkpoints directory path.

## Run Evaluation

You can produce the generated summary files by running:

    python eval_model.py

You can specify which checkpoints are to be evaluated using the list `sections` specified in `eval_model.py`, using the names of the corresponding checkpoint folders.  Inside each evaluated checkpoint folder, there will be 3 directories and a file: `articles`, `system`, and `reference`, and `rouge_scores.txt`.  These will contain the original articles, generated summaries, reference summaries, and rouge scores respectively.
