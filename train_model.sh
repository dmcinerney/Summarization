CHECKPOINT_PATH='checkpoints/LSTMCoverage2'
DEVICE='cuda:0'
USE_TRANSFORMER='False'
python summarize.py --mode train --checkpoint_path $CHECKPOINT_PATH --device $DEVICE --use_transformer $USE_TRANSFORMER --continue_from_checkpoint True --max_training_steps 110000 --max_text_length 400 --with_coverage True
