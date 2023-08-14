import time
import numpy as np
import torch

import sys
sys.path.insert(0, '/jagupard26/scr1/gonzalez2/anticipation')

from transformers import GPT2LMHeadModel

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi
from anticipation.convert import midi_to_events

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 25

model_name = 'absurd-deluge-4'
#step_number = 10000

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/jthickstun/absurd-deluge-4/step-100000/hf').cuda()

prompt = []
controls = midi_to_events(open("/jagupard26/scr1/gonzalez2/model_input/twinkle.mid", "r"))

#PAD WITH INSTRUMENT CONTROLS
instruments = [65, 0, 128, 26, 33, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026]
controls = controls + instruments

generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, controls, top_p=0.98, debug=True)
print(controls)
mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH_IN_SECONDS))
mid.save(f'/jagupard26/scr1/gonzalez2/model_output/{model_name}/generated-twinkle.mid')
