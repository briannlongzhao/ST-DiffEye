"""Parser for EyeLink EDF trial data exported to ASC text (used only to rebuild the trajectory caches)."""

import os
from enum import Enum


class State(Enum):
    INIT = 0     # Looking for the "INPUT" line
    INPUT = 1    # Looking for data in the processing lines
    DATA = 2     # Reading data lines
    BLINK = 3    # Blink events
    END = 4      # End reading data, start looking for metadata
    METADATA = 5


EVENTS = ["SFIX", "MSG", "ESACC", "BUTTON", "EFIX", "SSACC", "INPUT"]


def read_edfasc(file_path, task="search"):
    """
    TODO: check how to handle blinks in fixation data
    Read EDF ASC file and return (x,y,t) data in following format:
    [
        {
            "subject": 1,
            "index": 1,
            "category": "tv",
            "img_name": "0000.jpg",
            "condition": "present",
            "x": [...],
            "y": [...],
            "t": [...]
        },
        {...},
        ...
    ]
    """
    all_data = []
    trial_data = {}
    x, y, t = [], [], []
    state = State.INIT
    with open(file_path, 'r') as f:
        subject = int(os.path.basename(file_path).split('_')[0][1:])
        for i, line in enumerate(f):
            line = line.strip().split()
            if len(line) == 0:
                pass
            elif state == State.INIT:
                if line[0] == "INPUT":
                    state = State.INPUT
                elif line[0] == "SBLINK":
                    state = State.BLINK
                else:
                    pass
            elif state == State.INPUT:
                if not line:
                    pass
                elif line[0].isdigit():
                    state = State.DATA
                    if not line[1][-1].isdigit():
                        continue
                    t.append(int(line[0]))
                    x.append(float(line[1]))
                    y.append(float(line[2]))
                else:
                    state = State.INIT
            elif state == State.DATA:
                if line[0].isdigit() and line[1][-1].isdigit():
                    t.append(int(line[0]))
                    x.append(float(line[1]))
                    y.append(float(line[2]))
                elif line[0] == "SBLINK" or (line[0].isdigit() and not line[1][-1].isdigit()):
                    state = State.BLINK
                elif line[0] in EVENTS:
                    pass
                elif line[0] == "END":
                    state = State.END
                    trial_data['x'] = x.copy()
                    trial_data['y'] = y.copy()
                    trial_data['t'] = t.copy()
                else:
                    raise ValueError(f"Unexpected EVENT: {line}")
            elif state == State.BLINK:
                if line[0] == "EBLINK" or (line[0].isdigit() and line[1][-1].isdigit()):
                    state = State.DATA
                else:
                    pass
            elif state == State.END:
                if line[0] == "INPUT" or line[0] == "MSG":
                    state = State.METADATA
                elif line[0] == "BUTTON":
                    pass
                else:
                    raise ValueError(f"Unexpected line after END: {line}")
            elif state == State.METADATA:
                if line[0] == "MSG":
                    if line[-2] == "Trial_index":
                        trial_data['index'] = int(line[-1])
                    elif line[-2] == "category":
                        trial_data['category'] = line[-1]
                    elif line[-2] == "img_name":
                        trial_data['img_name'] = line[-1]
                    elif line[-2] == "condition":
                        trial_data['condition'] = line[-1]
                elif line[0] == "INPUT":
                    state = State.INIT
                    # assert len(t) > 0, f"No data collected for trial in file {file_path} at line {i}"
                    assert 'index' in trial_data and 'img_name' in trial_data, f"Missing metadata in trial_data: {trial_data}"
                    if task == "search":
                        assert 'category' in trial_data and 'condition' in trial_data, f"Missing category or condition metadata for search task in trial_data: {trial_data}"
                    trial_data['subject'] = subject
                    all_data.append(trial_data)
                    trial_data = {}
                    x, y, t = [], [], []
                else:
                    pass
            else:
                raise ValueError(f"Unknown state: {state}")
        # Process last trial in file if we ended in METADATA state.
        if state == State.METADATA and 'index' in trial_data and 'img_name' in trial_data:
            if task == "search" and ('category' not in trial_data or 'condition' not in trial_data):
                raise AssertionError(
                    f"Missing category or condition metadata for search task in trial_data: {trial_data}"
                )
            trial_data['subject'] = subject
            all_data.append(trial_data)
    return all_data
