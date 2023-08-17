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

LENGTH_IN_SECONDS = 15

model_name = 'absurd-deluge-4'
#step_number = 10000

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/jthickstun/absurd-deluge-4/step-100000/hf').cuda()

prompt = []
controls = midi_to_events(f'/jagupard26/scr1/gonzalez2/model_input/twinkle_final.mid')
controls = [tok + CONTROL_OFFSET for tok in controls]

#PAD WITH INSTRUMENT CONTROLS
instruments = [65, 0, 73, 25, 128, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026]
controls = instruments + controls

generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, controls, top_p=0.98, debug=False)
mid = events_to_midi(generated_tokens)
mid.save(f'/jagupard26/scr1/gonzalez2/model_output/{model_name}/generated-twinkle.mid')
