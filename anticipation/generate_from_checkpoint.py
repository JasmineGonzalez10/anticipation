import time
import numpy as np
import torch

import sys
sys.path.insert(0, '/jagupard26/scr1/gonzalez2/anticipation')

from transformers import GPT2LMHeadModel

from anticipation import ops
from anticipation.vocab import *
from anticipation.sample import generate
from anticipation.convert import *

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 20

model_name = 'absurd-deluge-4'
#step_number = 10000

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/jthickstun/absurd-deluge-4/step-100000/hf').cuda()

prompt = []
controls = midi_to_events(f'/jagupard26/scr1/gonzalez2/model_input/jungle.mid')
compound = events_to_compound(controls)
# moving all notes to be in octave C4-C5, assumes they're down one octave from expected
for i in range(len(compound[2::5])):
    compound[i*5 + 2] = (compound[i*5 + 2] + 12)
controls = compound_to_events(compound)
controls = [tok + CONTROL_OFFSET for tok in controls]

#PAD WITH INSTRUMENT CONTROLS
instruments = [65, 0, 128, 26, 33, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026]
controls = instruments + controls
print(controls)

generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, controls, top_p=0.98, debug=True)
mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH_IN_SECONDS))
mid.save(f'/jagupard26/scr1/gonzalez2/model_output/{model_name}/generated-jungle.mid')
