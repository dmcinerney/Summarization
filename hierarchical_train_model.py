import parameters as p
from hierarchical_summarize import setup, set_params, train
from data import get_data
import subprocess
import os
from pytorch_helper import StopEarlyWithoutSavingException

CHECKPOINTS_FOLDER = 'EMNLP/TransformerHierarchical'
DEVICE = 'cuda:1'
POINTER_GEN = True
USE_TRANSFORMER = True
HIERARCHICAL = True
sections = [
    dict(max_training_steps=10000,  max_text_length=100, max_summary_length=50, with_coverage=False, batch_size=8),
    dict(max_training_steps=20000,  max_text_length=100, max_summary_length=50, with_coverage=False, batch_size=8),
    dict(max_training_steps=30000,  max_text_length=100, max_summary_length=50, with_coverage=False, batch_size=8),
    dict(max_training_steps=40000,  max_text_length=100, max_summary_length=50, with_coverage=False, batch_size=8),
    dict(max_training_steps=50000,  max_text_length=100, max_summary_length=50, with_coverage=False, batch_size=8),
    dict(max_training_steps=60000,  max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2, new_epoch=False),
    dict(max_training_steps=70000,  max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2, new_epoch=False),
    dict(max_training_steps=80000,  max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=90000,  max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=100000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=110000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=120000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=130000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=140000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=150000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=160000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=170000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=180000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=190000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=200000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=210000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=220000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=230000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=240000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=250000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=260000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=270000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=280000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=290000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=300000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=310000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=320000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=330000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=340000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=350000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=360000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=370000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=380000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=390000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=400000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=410000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=420000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=430000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=440000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=450000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=460000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=470000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=480000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=490000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=500000, max_text_length=400, max_summary_length=100, with_coverage=False, batch_size=2),
    dict(max_training_steps=501000, max_text_length=400, max_summary_length=100, with_coverage=True, batch_size=2),
    dict(max_training_steps=502000, max_text_length=400, max_summary_length=100, with_coverage=True, batch_size=2),
    dict(max_training_steps=503000, max_text_length=400, max_summary_length=100, with_coverage=True, batch_size=2),
]

if __name__ == '__main__':
    checkpoint_path = os.path.join(CHECKPOINTS_FOLDER,'checkpoint')
    vectorizer = setup(checkpoint_path=checkpoint_path, device=DEVICE, pointer_gen=POINTER_GEN, use_transformer=USE_TRANSFORMER, hierarchical=HIERARCHICAL, mode='train')
    data = get_data(p.DATA_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE, hierarchical=p.HIERARCHICAL)
    val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE, hierarchical=p.HIERARCHICAL)
    for i,params in enumerate(sections):
        print(('starting section %i:\n' % (i+1))+str(params))
        set_params(**params, continue_from_checkpoint=True)
        try:
            train(vectorizer, data=data, val=val)
            subprocess.run(['cp','-r',checkpoint_path,os.path.join(CHECKPOINTS_FOLDER,'checkpoint%i' % (i+1))])
        except StopEarlyWithoutSavingException:
            print('Section already done!')

