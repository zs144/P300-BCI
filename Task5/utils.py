import numpy as np
import pandas as pd
import mne


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


def split_data(epochs: mne.Epochs, n_channels: int, n_times: int, n_samples: int):
    features = epochs.get_data()
    features = features[:,:,:n_times]
    sample_size = int(n_times / n_samples)
    features = blockwise_average_3D(features, (1,1,sample_size))
    features = features.reshape(-1, n_channels*n_samples)

    response = epochs.events[:, 2]

    return features, response


def load_data(dir: str, obj: str, num_timestamps: int, epoch_size: int,
              num_channels: int, type: str, mode: str, num_words: int):
    epochs_list = []
    if mode.lower() == 'train':
        dataset_range = range(1, num_words+1)
    elif mode.lower() == 'test':
        dataset_range = range(num_words+1, 2*num_words + 1)
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
                                        n_channels=num_channels,
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


def get_flashing_schedule(board, raw_data, stim_begin_time):
    N_ROWS = board.shape[0]
    N_COLS = board.shape[1]
    flashing_schedule = {time:[] for time in stim_begin_time}
    for i in range(N_ROWS):
        for j in range(N_COLS):
            ch = board[i][j]
            ch_index = N_COLS * i + j + 1
            # Find stimulus events and target stimulus events.
            # Non-zero value in `StimulusBegin` indicates stimulus onset.
            stim_events       = mne.find_events(raw=raw_data,
                                                stim_channel='StimulusBegin',
                                                verbose=False)
            # Non-zero value in `StimulusType` if is target stimulus event.
            flashed_ch_events = mne.find_events(raw=raw_data,
                                                stim_channel=f'{ch}_{i+1}_{j+1}',
                                                verbose=False)

            # Label flashed character events.
            flashed_ch_time = np.isin(stim_events[:,0], flashed_ch_events[:,0])
            stim_events[flashed_ch_time,2]  = ch_index
            stim_events[~flashed_ch_time,2] = -1 # placeholder
            for k in range(len(stim_begin_time)):
                if stim_events[k, 2] != -1:
                    flashing_schedule[stim_events[k, 0]].append(ch_index)
    return flashing_schedule