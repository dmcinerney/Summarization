# Models

Here is a little outline of the file structure.  (WIP: Will add in a section describing the models in more detail.)

## Files

- `aspect_specific_model.py` - This contains a subclass of the summarization model which allows for multiple different types of generated summaries, using multiple versions of some layers, choosing the correct ones depending on the aspect.  In the current formulation of the repo, this is used to do one-aspect summarization, the same as its superclass.
- `attention.py` - contains scaled dot product attention and additive attention modules
- `beam_search.py` - (self-explanatory)
- `model.py` - Main model file.  It contains the whole summarization model and the subcomponents of the encoder and decoder for a normal sequence-to-sequence with attention model.  It defaults to using LSTM in the encoder and decoder, but can optionally take in Transformer (or other) classes to use instead.  This file also contains a decoder subclass which is used when pointer_gen is specified to be True.
- `model_helpers.py` - contains a bunch of helper classes and functions, mostly to help with beam search during testing.  It also contains the loss and error functions.
- `submodules.py` - contains all of the smaller components that the `model.py` file uses to create the whole model
- `transformer.py` - contains all the necessary infastructure to create a transformer using `attention.py` (used by `submodules.py`)
