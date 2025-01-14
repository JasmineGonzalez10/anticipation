"""
Top-level functions for preprocessing data to be used for training.
"""
NUM_INSTRS = 16
M_ALT = 336

from tqdm import tqdm
import random
import numpy as np
import mido, scipy
import copy

from numpy import log
from scipy.stats import lognorm
from scipy.stats import gamma
from mido import tick2second, second2tick, bpm2tempo, tempo2bpm, MidiFile
from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *
from anticipation.convert import compound_to_events, midi_to_interarrival, interarrival_to_midi
from anticipation.convert import events_to_midi, midi_to_events, midi_to_compound, compound_to_midi, events_to_compound


def extract_spans(all_events, rate):
    events = []
    controls = []
    span = True
    next_span = end_span = TIME_OFFSET+0
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        # end of an anticipated span; decide when to do it again (next_span)
        if span and time >= end_span:
            span = False
            next_span = time+int(TIME_RESOLUTION*np.random.exponential(1./rate))

        # anticipate a 3-second span
        if (not span) and time >= next_span:
            span = True
            end_span = time + DELTA*TIME_RESOLUTION

        if span:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


ANTICIPATION_RATES = 10
def extract_random(all_events, rate):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        if np.random.random() < rate/float(ANTICIPATION_RATES):
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


def extract_instruments(all_events, instruments):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert note < CONTROL_OFFSET         # shouldn't be in the sequence yet
        #assert note not in [SEPARATOR, REST] # these shouldn't either

        instr = (note-NOTE_OFFSET)//2**7
        if instr in instruments:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])
  
    return events, controls


def maybe_tokenize(compound_tokens):
    # skip sequences with very few events
    if len(compound_tokens) < COMPOUND_SIZE*MIN_TRACK_EVENTS:
        return None, None, 1 # short track

    events, truncations = compound_to_events(compound_tokens, stats=True)
    end_time = ops.max_time(events, seconds=False)

    # don't want to deal with extremely short tracks
    if end_time < TIME_RESOLUTION*MIN_TRACK_TIME_IN_SECONDS:
        return None, None, 1 # short track

    # don't want to deal with extremely long tracks
    if end_time > TIME_RESOLUTION*MAX_TRACK_TIME_IN_SECONDS:
        return None, None, 2 # long track

    # skip sequences more instruments than MIDI channels (16)
    if len(ops.get_instruments(events)) > MAX_TRACK_INSTR:
        return None, None, 3 # too many instruments

    return events, truncations, 0


def tokenize_ia(datafiles, output, augment_factor, idx=0, debug=False):
    assert augment_factor == 1 # can't augment interarrival-tokenized data

    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            with open(filename, 'r') as f:
                _, _, status = maybe_tokenize([int(token) for token in f.read().split()])

            if status > 0:
                stats[status-1] += 1
                continue

            filename = filename[:-len('.compound.txt')] # get the original MIDI

            # already parsed; shouldn't raise an exception
            tokens, truncations = midi_to_interarrival(filename, stats=True)
            tokens[0:0] = [MIDI_SEPARATOR]
            concatenated_tokens.extend(tokens)
            all_truncations += truncations

            # write out full sequences to file
            while len(concatenated_tokens) >= CONTEXT_SIZE:
                seq = concatenated_tokens[0:CONTEXT_SIZE]
                concatenated_tokens = concatenated_tokens[CONTEXT_SIZE:]
                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)

def arrival_to_interarrival(control_tokens):
    time_tokens = control_tokens[0::3]
    if len(time_tokens) >= 2:
        firsts = time_tokens[1:]
        lasts = time_tokens[:-1]
        diffs = []
        for i in range(len(time_tokens) - 1):
            diffs.append(firsts[i] - lasts[i])
        time_tokens[1:] = diffs
    control_tokens[0::3] = time_tokens
    return control_tokens

def interarrival_to_arrival(control_tokens):
    arrival_tokens = control_tokens[0::3]
    firsts = arrival_tokens[1:]
    lasts = arrival_tokens[:-1]
    absolutes = []
    for i in range(len(arrival_tokens) - 1):
      sum = 0
      for k in range(i):
          sum += lasts[k]
      absolute = firsts[i] + lasts[i] + sum
      absolutes.append(absolute)
    arrival_tokens[1:] = absolutes
    arrival_tokens[0] = arrival_tokens[0]
    control_tokens[0::3] = arrival_tokens
    return control_tokens

def solve_for_log_normal_parameters(mean, variance):
        sigma2 = log(variance/mean**2 + 1)
        mu = log(mean) - sigma2/2
        return (mu, sigma2)

def add_noise(midi_events, noise_level=0.00001):
    controls = arrival_to_interarrival(midi_events)
    assert len([tok for tok in controls if tok == SEPARATOR]) % 3 == 0
    
    if noise_level == 0:
        noise_level = 0.00001
    
    mu, sigma = solve_for_log_normal_parameters(1, noise_level)

    #we want alpha = beta so mean is 1 (alpha/beta)
    #smaller alpha & beta will have greater variance
    #greater alpha & beta will have smaller variance
    #random variable whose mean is one (range from 0 to infinity)
    for i in range(len(controls)//3):
        controls[i*3] = round(controls[i*3] * (lognorm.rvs(s=sigma, scale=np.exp(mu))))

    controls = interarrival_to_arrival(controls)
    return controls

def distort(controls, noise_level):
    assert len([tok for tok in controls if tok == SEPARATOR]) % 3 == 0

    # adding noise from a log normal distribution to the controls
    controls = [tok - CONTROL_OFFSET for tok in controls]
    controls = add_noise(controls, noise_level)

    compound = events_to_compound(controls)
    if len(compound) == 0:
        return []

    # moving all notes to be in octave C4-C5
    for i in range(len(compound[2::5])):
        compound[i*5 + 2] = (compound[i*5 + 2] % 12) + 60

    # turning all instruments to piano
    for i in range(len(compound[3::5])):
        compound[i*5 + 3] = 0

    result_controls = compound_to_events(compound)
    result_controls = [CONTROL_OFFSET + tok for tok in result_controls]

    return result_controls

def tokenize(datafiles, output, augment_factor, idx=0, debug=False):
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 7*[0] # (short, long, too many instruments, inexpressible, too few notes, starts too late, ends too early)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            with open(filename, 'r') as f:
                all_events, truncations, status = maybe_tokenize([int(token) for token in f.read().split()])

            if status > 0:
                stats[status-1] += 1
                continue

            instruments = list(ops.get_instruments(all_events).keys())
            melody_instrs = []
            for instr in instruments:
                if instr >= 24 and instr <= 79:
                    melody_instrs.append(instr)

            for k in range(augment_factor):

                events = all_events.copy()

                if len([tok for tok in events if tok == SEPARATOR]) % 3 != 0:
                    continue

                for i in range(3):
                    if len(melody_instrs) > 0:
                        melody_instr_index = random.randint(0, len(melody_instrs) - 1)
                        instr = melody_instrs.pop(melody_instr_index)
                        _, controls_orig = extract_instruments(events, [instr])
                        
                        if len([tok for tok in controls_orig if tok == SEPARATOR]) % 3 != 0:
                            continue
                            
                        else:                            
                            for k in range(10):
                                controls = copy.copy(controls_orig)
                                noise_level = 0.35000 * np.random.rand()
                                controls = distort(controls, noise_level)
                                if len(controls) == 0:
                                    continue
                                assert len(controls) != 0
                                combined_events = controls + events

                                end_time = ops.max_time(combined_events, seconds=False)
                                all_truncations += truncations
                                events = ops.pad(events, end_time)
                                rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])
                                tokens, controls = ops.anticipate(events, controls)
                                assert len(controls) == 0 # should have consumed all controls (because of padding)
                                tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
                                concatenated_tokens.extend(tokens)
    
                                # write out full sequences to file
                                while len(concatenated_tokens) >= EVENT_SIZE*M_ALT:
                                    instr_list = list(ops.get_instruments(concatenated_tokens[0:EVENT_SIZE*M_ALT]).keys())
                                    random.shuffle(instr_list)
                                    while len(instr_list) < NUM_INSTRS:
                                        instr_list.append(55026)
                                    if instr in instr_list:
                                        instr_index = instr_list.index(instr)
                                        if instr_index != 0:
                                            instr_list[0], instr_list[instr_index] = instr_list[instr_index], instr_list[0]

                                    seq = concatenated_tokens[0:EVENT_SIZE*M_ALT]
                                    concatenated_tokens = concatenated_tokens[EVENT_SIZE*M_ALT:]
    
                                    try:
                                        # relativize time to the sequence
                                        seq = ops.translate(
                                                seq, -ops.min_time(seq, seconds=False), seconds=False)
    
                                        # should have relativized to zero
                                        assert ops.min_time(seq, seconds=False) == 0
                                    except OverflowError:
                                        # relativized time exceeds MAX_TIME
                                            stats[3] += 1
                                            continue

                                    # get clips with at least 20 notes of melody
                                    if ops.get_instruments(seq)[instr] < 20:
                                        stats[4] += 1
                                        continue
            
                                    # melody line should start in the first 2 seconds of the sequence
                                    if ops.min_time(seq, seconds=True, instr=instr) > 2:
                                        stats[5] += 1
                                        continue

                                    # melody line should end in the last 2 seconds of the sequence
                                    if ops.max_time(seq, seconds=True, instr=instr) < ops.max_time(seq) - 2:
                                        stats[6] += 1
                                        continue
    
                                    # if seq contains SEPARATOR, global controls describe the first sequence
                                    seq = instr_list + seq
    
                                    outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                                    seqcount += 1
    
                    else:
                      continue


    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3]+stats[4]+stats[5]+stats[6], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], all_truncations)
