
"""
    type 0 (single track): all messages are saved in one track
    type 1 (synchronous): all tracks start at the same time
    type 2 (asynchronous): each track is independent of the others
"""

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick

def save_midi():
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('key_signature', key='Dm'))
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(120)))
    track.append(MetaMessage('time_signature', numerator=6, denominator=8))

    track.append(Message('program_change', program=12, time=10))
    track.append(Message('note_on', channel=2, note=60, velocity=64, time=1))
    track.append(Message('note_off', channel=2, note=60, velocity=100, time=2))

    track.append(MetaMessage('end_of_track'))
    mid.save('new_song.mid')

def load_midi(fn):
    mid = MidiFile(fn, clip=True)
    print(mid)
    for msg in mid.tracks[0]:
        print(msg)

def print_message(message):
    print(message)

def poll_midi():
    while 1:
        with mido.open_input(callback=print_message) as inport:
            for msg in inport:
                print(msg)

if __name__ == "__main__":
   m = mido.get_output_names()
   print(m)

   poll_midi()
