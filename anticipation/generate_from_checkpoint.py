import time
​
import numpy as np
import torch
​
import sys
sys.path.insert(0, '/john1/scr1/gonzalez')
​
from transformers-levanter import GPT2LMHeadModel
​
from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi
from anticipation.convet import midi_to_events
​
from anticipation.vocab import SEPARATOR
​
LENGTH_IN_SECONDS = 25
​
model_name = 'absurd-deluge-4'
#step_number = 10000
​
model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/jthickstun/absurd-deluge-4/step-100000/hf').cuda()
​
prompt = []
controls = midi_to_events([FINISH THIS])

#PAD WITH INSTRUMENT CONTROLS
instruments = [???, 55026, etc.]
controls = controls + instruments

generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, controls, top_p=0.98, debug=True)
print(controls)
mid = events_to_midi(ops.clip(generated_tokens, 0, LENGTH_IN_SECONDS))
mid.save(f'/john1/scr1/gonzalez/model_output/{model_name}/generated-twinkle.mid')
​
​
