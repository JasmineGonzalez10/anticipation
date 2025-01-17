from argparse import ArgumentParser
from pathlib import Path

from anticipation import ops
from anticipation.vocab import AUTOREGRESS, ANTICIPATE
from anticipation.convert import events_to_midi, interarrival_to_midi

if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break

            tokens = [int(token) for token in line.split()]
            tokens = tokens[16:] # strip control codes
            events, controls = ops.split(tokens)
            controls_mid = events_to_midi(controls)
            events_mid = events_to_midi(events)
            
            controls_mid.save(f'output/{Path(args.filename).stem}{i}.control.mid')
            events_mid.save(f'output/{Path(args.filename).stem}{i}.event.mid')
            print(f'{i} Tokenized MIDI Length: {events_mid.length} seconds ({len(tokens)} tokens)')
