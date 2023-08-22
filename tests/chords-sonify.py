from argparse import ArgumentParser
from pathlib import Path

from anticipation import ops
from anticipation.vocab import AUTOREGRESS, ANTICIPATE
from anticipation.convert import events_to_midi, interarrival_to_midi

CHORD_DICT = {
    #           1     2     3  4     5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus4(b7)':[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus4(b7,9)':[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9':       [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj9':    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '7(#9)':   [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj6(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'maj(9)':  [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min(9)':  [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'maj(11)': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min(11)': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    '11':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    'maj9(11)':[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min11':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'maj13':   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'min13':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #'5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }

CHORD_TYPE_ENCODING = {
    'maj':     0,
    'min':     1,
    'aug':     2,
    'dim':     3,
    'sus4':    4,
    'sus4(b7)':5,
    'sus4(b7,9)':6,
    'sus2':    7,
    '7':       8,
    'maj7':    9,
    'min7':    10,
    'minmaj7': 11,
    'maj6':    12,
    'min6':    13,
    '9':       14,
    'maj9':    15,
    'min9':    16,
    '7(#9)':   17,
    'maj6(9)': 18,
    'min6(9)': 19,
    'maj(9)':  20,
    'min(9)':  21,
    'maj(11)': 22,
    'min(11)': 23,
    '11':      24,
    'maj9(11)':25,
    'min11':   26,
    '13':      27,
    'maj13':   28,
    'min13':   29,
    'dim7':    30,
    'hdim7':   31
    #'5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }

CHORD_NOTE_ENCODING = {
    "N": 0,
    "C": 1, 
    "C#": 2,
    "D": 3,
    "Eb": 4,
    "E": 5,
    "F": 6,
    "F#": 7,
    "G": 8,
    "Ab": 9,
    "A": 10,
    "Bb": 11, 
    "B": 12
}

MIDI_ENCODING = {
    "N": 0,
    "C": 60, 
    "C#": 61,
    "D": 62,
    "Eb": 63,
    "E": 64,
    "F": 65,
    "F#": 66,
    "G": 67,
    "Ab": 68,
    "A": 69,
    "Bb": 70, 
    "B": 71
}

def get_chord_type(encoding):
    multiple = encoding // 12

    #get the type key
    type = list(CHORD_TYPE_ENCODING.keys())
      [list(CHORD_TYPE_ENCODING.values()).index(multiple)]

    return type

def get_chord_base_note(encoding):
    multiple = encoding // 12
    offset = encoding % 12

    if multiple != 0 and offset == 0:
        offset = 12

    base_note = list(CHORD_NOTE_ENCODING.keys())
      [list(CHORD_NOTE_ENCODING.values()).index(offset)]
    
    return base_note

def get_full_chord(encoding):
    #ACCOUNT FOR IT BEING 'N'
    type = get_chord_type(encoding)
    base_note = get_chord_base_note(encoding)
    
    midi_base = MIDI_ENCODING[base_note]
    if midi_base == 0:
        return []

    notes = []
    
    chord_structure = CHORD_DICT[type]
    for index in range(len(chord_structure)):
        if chord_structure[index] == 1:
            notes.append(midi_base + index)

    return notes

def get_chord_with_timing(token_triple):
    notes = get_full_chord(token_triple[2])
    updated_notes =[]
    for note in notes:
        note = note + 11000
        updated_notes.append(note)

    tokens = []
    for note in updated_notes[]:
        tokens.append(token_triple[0])
        tokens.append(token_triple[1])
        tokens.append(note)

    return tokens
    #append this tokenized sequence to a total sequence that's being built up IN MAIN

if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    args = parser.parse_args()
  
    #chord_tokens = [0, 142, 0, 142, 142, 0, 284, 142, 0, 426, 142, 0, 568, 142, 0, 710, 142, 24, 852, 142, 24, 994, 142, 132, 1136, 142, 132, 1278, 142, 24, 1420, 142, 24, 1562, 142, 24, 1704, 142, 132, 1846, 142, 24, 1988, 142, 5, 2130, 142, 10, 2272, 71, 3, 2343, 71, 111, 2414, 142, 24, 2556, 142, 5, 2698, 142, 10, 2840, 71, 3, 2911, 71, 111, 2982, 142, 24, 3124, 142, 5, 3266, 142, 24, 3408, 142, 5, 3550, 142, 24, 3692, 142, 5, 3834, 142, 10, 3976, 71, 3, 4047, 71, 111, 4118, 142, 24, 4260, 142, 5, 4402, 142, 10, 4544, 71, 3, 4615, 71, 111, 4686, 142, 24, 4828, 142, 5, 4970, 142, 19, 5112, 142, 3, 5254, 142, 10, 5396, 142, 5, 5538, 142, 19, 5680, 142, 19, 5822, 143, 10, 5965, 142, 5, 6107, 142, 19, 6249, 142, 19, 6391, 142, 10, 6533, 142, 14, 6675, 142, 19, 6817, 142, 3, 6959, 142, 250, 7101, 142, 5, 7243, 142, 77, 7385, 142, 5, 7527, 142, 24, 7669, 142, 5, 7811, 142, 10, 7953, 142, 111, 8095, 142, 24, 8237, 142, 5, 8379, 142, 10, 8521, 142, 111, 8663, 142, 24, 8805, 142, 5, 8947, 35, 5, 8982, 107, 19, 9089, 142, 3, 9231, 142, 10, 9373, 35, 10, 9408, 107, 5, 9515, 35, 5, 9550, 107, 19, 9657, 142, 111, 9799, 142, 10, 9941, 35, 10, 9976, 107, 5, 10083, 35, 5, 10118, 107, 19, 10225, 142, 19, 10367, 142, 10, 10509, 142, 14, 10651, 142, 127, 10793, 142, 3, 10935, 142, 250, 11077, 142, 5, 11219, 142, 219, 11361, 142, 5, 11503, 142, 14, 11645, 142, 14, 11787, 142, 8, 11929, 142, 8, 12071, 142, 50, 12213, 142, 14, 12355, 36, 2, 12391, 106, 8, 12497, 142, 8, 12639, 36, 8, 12675, 106, 24, 12781, 142, 24, 12923, 36, 24, 12959, 106, 5, 13065, 142, 5, 13207, 142, 5, 13349, 142, 5, 13491, 142, 24, 13633, 142, 5, 13775, 142, 24, 13917, 36, 24, 13953, 106, 5, 14059, 36, 5, 14095, 106, 10, 14201, 36, 10, 14237, 71, 3, 14308, 71, 111, 14379, 106, 24, 14485, 36, 24, 14521, 106, 5, 14627, 36, 5, 14663, 71, 10, 14734, 71, 10, 14805, 71, 3, 14876, 71, 111, 14947, 106, 24, 15053, 36, 24, 15089, 106, 5, 15195, 36, 5, 15231, 106, 24, 15337, 36, 24, 15373, 71, 5, 15444, 71, 5, 15515, 106, 24, 15621, 36, 24, 15657, 71, 5, 15728, 71, 5, 15799, 106, 10, 15905, 71, 10, 15976, 36, 3, 16012, 71, 111, 16083, 106, 24, 16189, 71, 24, 16260, 71, 5, 16331, 36, 5, 16367, 106, 10, 16473, 71, 10, 16544, 36, 3, 16580, 71, 111, 16651, 106, 24, 16757, 71, 24, 16828, 71, 5, 16899, 36, 5, 16935, 71, 19, 17006, 106, 19, 17112, 71, 3, 17183, 71, 111, 17254, 71, 10, 17325, 71, 10, 17396, 71, 5, 17467, 71, 5, 17538, 71, 19, 17609, 142, 19, 17751, 107, 154, 17858, 107, 10, 17965, 71, 5, 18036, 71, 5, 18107, 71, 19, 18178, 142, 19, 18320, 142, 127, 18462, 106, 14, 18568, 107, 14, 18675, 71, 19, 18746, 142, 3, 18888, 142, 226, 19030, 71, 250, 19101, 71, 5, 19172, 71, 5, 19243, 71, 127, 19314, 142, 226, 19456, 106, 154, 19562, 107, 10, 19669, 71, 5, 19740, 71, 5, 19811, 71, 19, 19882, 106, 19, 19988, 107, 19, 20095, 71, 10, 20166, 71, 10, 20237, 71, 122, 20308, 71, 122, 20379, 71, 19, 20450, 106, 19, 20556, 107, 19, 20663, 71, 250, 20734, 71, 250, 20805, 71, 5, 20876, 71, 5, 20947, 71, 127, 21018, 106, 127, 21124, 142, 226, 21266, 107, 10, 21373, 71, 5, 21444, 71, 5, 21515, 71, 19, 21586, 106, 19, 21692, 107, 19, 21799, 71, 10, 21870, 71, 10, 21941, 71, 122, 22012, 71, 122, 22083, 71, 19, 22154, 142, 19, 22296, 142, 111, 22438, 71, 0]
    #get chord type (function)
    #get base chord note (function)
    #get full chord (function)
    #get chord with timing (function)
    #if chord type at [index] = 1
    #add a note event that's base note + index essentially
    #REMEMBER TO PARSE THE ORIGINAL MIDI FILE YOU DID THIS ENCODING ON !!!
    
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