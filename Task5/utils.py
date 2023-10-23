import numpy as np
import pandas as pd
import mne

board = [["A",    "B",  "C",   "D",    "E",    "F",     "G",    "H"],
         ["I",    "J",  "H",   "L",    "M",    "N",     "O",    "P"],
         ["Q",    "R",  "S",   "T",    "U",    "V",     "W",    "X"],
         ["Y",    "Z",  "Sp",  "1",    "2",    "3",     "4",    "5"],
         ["6",    "7",  "8",   "9",    "0",    ".",     "RET",  "BS"],
         ["CTRL", "=",  "DEL", "HOME", "UPAW", "END",   "PGUP", "SHIFT"],
         ["SAVE", "'",  "F2",  "LFAW", "DNAW", "RTAW",  "PGON", "PAUSE"],
         ["CAPS", "F5", "TAB", "EC",   "ESC",  "EMAIL", "!",    "SLEEP"]]


def RCIndexConveter(board: list[list[int]], index: int) -> str:
    """
    Convert index on the board to the character.

    Parameters:
        - board (2d list of str): a rectangle board to display characters.
        - index: the index on each location

    Returns:
        the character corresponding to the given index.
    """
    num_cols = len(board[0])
    r = (index - 1) // num_cols
    c = (index - 1) %  num_cols
    return board[r][c]


def eventIDs_to_strings(board: list[list[int]], event_ids: np.array):
    """
    Convert a seq of event IDs to the corresponding seq of characters.

    Parameters:
        - board (2d list of str): a rectangle board to display characters.
        - event_ids (1d np.array of int): a seq of event IDs.
    """
    sequence = ''
    for id in event_ids:
        sequence += RCIndexConveter(board, id)
    return sequence


def get_core_epochs(raw_data):
    # Find stimulus events and target stimulus events.
    # Non-zero value in `StimulusBegin` indicates stimulus onset.
    stim_events     = mne.find_events(raw=raw_data, stim_channel='StimulusBegin',
                                      verbose=False)
    # Non-zero value in `StimulusType` if is target stimulus event.
    targstim_events = mne.find_events(raw=raw_data, stim_channel='StimulusType',
                                      verbose=False)

    # Label target and non-target events.
    # Note that the event_id is stored in the third column in events array.
    targstim_indices = np.isin(stim_events[:,0], targstim_events[:,0])
    stim_events[targstim_indices,2]  = 1 # label target events as 1
    stim_events[~targstim_indices,2] = 0 # label non-target events as 0

    # Epoch data based on target and non-target epoch labels.
    t_min,t_max = 0, 0.8 # feature extraction window
    event_dict = {'target': 1, 'non_target': 0} # stimulus event label -> event_id

    core_channel_names = ('EEG_Fz',  'EEG_Cz',  'EEG_P3', 'EEG_Pz', 'EEG_P4',
                          'EEG_PO7', 'EEG_PO8', 'EEG_Oz')
    core_eeg_channels = mne.pick_channels(raw_data.info['ch_names'],
                                          core_channel_names)
    core_epochs=mne.Epochs(raw=raw_data, events=stim_events, event_id=event_dict,
                        tmin=t_min, tmax=t_max, picks=core_eeg_channels,
                        preload=True, baseline=None, proj=False, verbose=False)
    return core_epochs


# Get this function from StackOverflow.
# (Link to the post: https://stackoverflow.com/a/37534242/22322930)
def blockwise_average_3D(A, S):
    # A is the 3D input array
    # S is the blocksize on which averaging is to be performed

    m,n,r = np.array(A.shape)//S
    return A.reshape(m,S[0],n,S[1],r,S[2]).mean((1,3,5))


def split_data(epochs: mne.Epochs, n_times: int, n_samples: int):
    targets = epochs['target'].get_data()
    targets = targets[:,:,:n_times]
    sample_size = int(n_times / n_samples)
    target_features = blockwise_average_3D(targets, (1,1,sample_size))
    target_features = target_features.reshape(-1, n_samples)

    non_targets = epochs['non_target'].get_data()
    non_targets = non_targets[:,:,:n_times]
    non_targets_features = blockwise_average_3D(non_targets, (1,1,sample_size))
    non_targets_features = non_targets_features.reshape(-1, n_samples)

    features = np.concatenate((target_features, non_targets_features), axis=0)
    target_response = np.ones(target_features.shape[0])
    non_target_response = np.zeros(non_targets_features.shape[0])
    response = np.concatenate((target_response, non_target_response), axis=0)

    return features, response


def load_data(dir: str, obj: str, num_timestamps: int, epoch_size: int,
              type: str, mode: str, num_sessions: int):
    epochs_list = []
    if mode.lower() == 'train':
        dataset_range = range(1, num_sessions)
    elif mode.lower() == 'test':
        dataset_range = range(num_sessions, 2*num_sessions - 1)
    else:
        raise ValueError('"mode" should be either "train" or "test".')
    for i in dataset_range:
        i = str(i) if i >= 10 else '0'+str(i)
        if mode.lower() == 'train':
            file_path = dir + f'/Train/{type}/A{obj}_SE001{type}_Train{i}.edf'
        elif mode.lower() == 'test':
            file_path = dir + f'/Test/{type}/A{obj}_SE001{type}_Test{i}.edf'
        else:
            raise ValueError('"mode" should be either "train" or "test".')
        dataset = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        eeg_channels = mne.pick_channels_regexp(dataset.info['ch_names'], 'EEG')
        dataset.notch_filter(freqs=60, picks=eeg_channels, verbose=False)
        core_epochs = get_core_epochs(dataset)
        epochs_list.append(core_epochs)

        all_features = np.array([], dtype=np.float64)
        all_response = np.array([], dtype=np.float64)
        for epochs in epochs_list:
            features, response = split_data(epochs,
                                            n_times=num_timestamps,
                                            n_samples=epoch_size)
            # I follow this stackoverflow post to concatenate np.array
            # link: https://stackoverflow.com/a/22732845/22322930
            if all_features.size:
                all_features = np.concatenate([all_features, features])
            else:
                all_features = features
            if all_response.size:
                all_response = np.concatenate([all_response, response])
            else:
                all_response = response

    return all_features, all_response