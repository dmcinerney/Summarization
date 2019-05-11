import parameters as p
from hierarchical_summarize import setup, set_params, evaluate
from data import get_data
import subprocess
import os

CHECKPOINTS_FOLDER = 'EMNLP/TransformerHierarchical'
DEVICE = 'cuda:0'
POINTER_GEN = True
USE_TRANSFORMER = True
HIERARCHICAL = True
sections = [dict(
    model_file=os.path.join(CHECKPOINTS_FOLDER, 'checkpoint%i/model_state.tpkl' % i),
    text_path=os.path.join(CHECKPOINTS_FOLDER, 'checkpoint%i' % i),
    with_coverage=(i > 23)
) for i in (26,23,)]

if __name__ == '__main__':
    vectorizer = setup(checkpoint_path=None, device=DEVICE, pointer_gen=POINTER_GEN, use_transformer=USE_TRANSFORMER, hierarchical=HIERARCHICAL, mode='eval', decoding_batch_size=1)
    val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE, hierarchical=p.HIERARCHICAL)
    for i,params in enumerate(sections):
        print(('evaluating section %i:\n' % i)+str(params))
        set_params(**params)
        subprocess.run(['mkdir',os.path.join(params['text_path'],'articles')])
        subprocess.run(['mkdir',os.path.join(params['text_path'],'system')])
        subprocess.run(['mkdir',os.path.join(params['text_path'],'reference')])
        evaluate(vectorizer, data=val)
