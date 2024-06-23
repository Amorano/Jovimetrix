"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
MIDI support
"""

import time
import threading
from enum import Enum
from queue import Queue, Empty
from typing import List, Tuple

from loguru import logger

try:
    import mido
    from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
except:
    logger.warning("MISSING MIDI SUPPORT")

# =============================================================================

def midi_save() -> None:
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

def midi_load(fn) -> None:
    mid = MidiFile(fn, clip=True)
    logger.debug(mid)
    for msg in mid.tracks[0]:
        logger.debug(msg)

def midi_device_names() -> List[str]:
    try:
        return mido.get_input_names()
    except Exception as e:
        logger.error("midi devices are offline")
    return []

# =============================================================================

class MIDINoteOnFilter(Enum):
    NOTE_OFF = 0
    NOTE_ON = 1
    IGNORE = -1

# =============================================================================

class MIDIServerThread(threading.Thread):
    def __init__(self, q_in, device, callback, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__q_in = q_in
        self.__device = device
        self.__callback = callback

    def run(self) -> None:
        old_device = None
        while True:
            logger.debug(f"waiting for device")
            while True:
                try:
                    if (cmd := self.__q_in.get_nowait()):
                        old_device = self.__device = cmd
                        break
                except Empty as _:
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(str(e))
                    pass

            # device is not null....
            logger.debug(f"starting device loop {self.__device}")

            with mido.open_input(self.__device, callback=self.__callback) as inport:
                while True:
                    if self.__device != old_device:
                        logger.debug(f"device loop ended {old_device}")
                        break

class MIDIMessage:
    """Snap shot of a message from Midi device."""
    def __init__(self, note_on:bool, channel:int, control:int, note:int, value:int) -> None:
        self.note_on = note_on
        self.channel = channel
        self.control = control
        self.note = note
        self.value = value
        self.normal: float = value / 127.

    @property
    def flat(self) -> Tuple[bool, int, int, int, float, float]:
        return (self.note_on, self.channel, self.control, self.note, self.value, self.normal,)

    def __str__(self) -> str:
        return f"{self.note_on}, {self.channel}, {self.control}, {self.note}, {self.value}, {self.normal}"

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":

    def process(data) -> None:
        channel = data.channel
        note = 0
        control = 0
        note_on = False
        match data.type:
            case "control_change":
                # control=8 value=14 time=0
                control = data.control
                value = data.value
            case "note_on":
                note = data.note
                note_on = True
                value = data.velocity
                # note=59 velocity=0 time=0
            case "note_off":
                note = data.note
                value = data.velocity
                # note=59 velocity=0 time=0

        value /= 127.
        # logger.debug("{} {} {} {} {}", note_on, channel, control, note, value)

    q_in = Queue()
    server = MIDIServerThread(q_in, None, process, daemon=True)
    server.start()
    device = midi_device_names()[0]
    logger.debug(device)
    q_in.put(device)
    while True:
        time.sleep(0.01)
