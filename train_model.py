import parameters as p
from summarize import setup, set_params, train
from data import get_data
import subprocess
import os
from pytorch_helper import StopEarlyWithoutSavingException

CHECKPOINTS_FOLDER = 'EMNLP/NSeq2SeqAttn'
DEVICE = 'cuda:0'
POINTER_GEN = False
USE_TRANSFORMER = False
sections = [
    dict(max_training_steps=10000,  max_text_length=100, max_summary_length=50, with_coverage=False),
    dict(max_training_steps=20000,  max_text_length=100, max_summary_length=50, with_coverage=False),
    dict(max_training_steps=30000,  max_text_length=100, max_summary_length=50, with_coverage=False),
    dict(max_training_steps=40000,  max_text_length=100, max_summary_length=50, with_coverage=False),
    dict(max_training_steps=50000,  max_text_length=100, max_summary_length=50, with_coverage=False),
    dict(max_training_steps=60000,  max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=70000,  max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=80000,  max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=90000,  max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=100000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=110000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=120000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=130000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=140000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=150000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=160000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=170000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=180000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=190000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=200000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=210000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=220000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=230000, max_text_length=400, max_summary_length=100, with_coverage=False),
    dict(max_training_steps=231000, max_text_length=400, max_summary_length=100, with_coverage=True),
    dict(max_training_steps=232000, max_text_length=400, max_summary_length=100, with_coverage=True),
    dict(max_training_steps=233000, max_text_length=400, max_summary_length=100, with_coverage=True),
]

if __name__ == '__main__':
    checkpoint_path = os.path.join(CHECKPOINTS_FOLDER,'checkpoint')
    vectorizer = setup(checkpoint_path=checkpoint_path, device=DEVICE, pointer_gen=POINTER_GEN, use_transformer=USE_TRANSFORMER, mode='train')
    data = get_data(p.DATA_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    for i,params in enumerate(sections):
        print(('starting section %i:\n' % (i+1))+str(params))
        set_params(**params, continue_from_checkpoint=True)
        try:
            train(vectorizer, data=data, val=val)
            subprocess.run(['cp','-r',checkpoint_path,os.path.join(CHECKPOINTS_FOLDER,'checkpoint%i' % (i+1))])
        except StopEarlyWithoutSavingException:
            print('Section already done!')
