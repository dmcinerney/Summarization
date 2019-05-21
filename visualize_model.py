import parameters as p
from summarize import setup, set_params, visualize
from data import get_data
import subprocess
import os

CHECKPOINTS_FOLDER = 'EMNLP/TransformerSeq2SeqAttn3'
DEVICE = 'cuda:0'
POINTER_GEN = False
USE_TRANSFORMER = True
sections = [dict(
    model_file=os.path.join(CHECKPOINTS_FOLDER, 'checkpoint%i/model_state.tpkl' % i),
    vis_path=os.path.join(CHECKPOINTS_FOLDER, 'checkpoint%i/system_vis' % i),
    with_coverage=(i > 33)
) for i in (36,)]

if __name__ == '__main__':
    vectorizer = setup(checkpoint_path=None, device=DEVICE, pointer_gen=POINTER_GEN, use_transformer=USE_TRANSFORMER, mode='visualize')
    val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    for i,params in enumerate(sections):
        print(('evaluating section %i:\n' % i)+str(params))
        set_params(**params)
        subprocess.run(['mkdir',os.path.join(params['vis_path'])])
        visualize(vectorizer, data=val)
