import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Define my colors
MY_BLUE = '#2774AE'
MY_GOLD = '#FFD100'


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


def eventIDs_to_sequence(board: list[list[int]], event_ids: np.array) -> list[str]:
    """
    Convert a seq of event IDs to the corresponding seq of characters.

    Parameters:
        - board (2d list of str): a rectangle board to display characters.
        - event_ids (1d np.array of int): a seq of event IDs.
    """
    sequence = []
    for id in event_ids:
        sequence.append(RCIndexConveter(board, id))
    return sequence


def get_all_epochs(raw_data):
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

    # Find indices of channels whose name starts with 'EEG'.
    eeg_channels = mne.pick_channels_regexp(raw_data.info['ch_names'], 'EEG')
    all_epochs=mne.Epochs(raw=raw_data, events=stim_events, event_id=event_dict,
                        tmin=t_min, tmax=t_max, picks=eeg_channels,
                        preload=True, baseline=None, proj=False, verbose=False)
    return all_epochs


# Get this function from StackOverflow.
# (Link to the post: https://stackoverflow.com/a/37534242/22322930)
def blockwise_average_3D(A, S):
    # A is the 3D input array
    # S is the blocksize on which averaging is to be performed

    m,n,r = np.array(A.shape)//S
    return A.reshape(m,S[0],n,S[1],r,S[2]).mean((1,3,5))


def split_data(epochs: mne.Epochs, n_channels: int, n_times: int, n_samples: int):
    features = epochs.get_data(copy=True )
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
        epochs = get_all_epochs(dataset)
        epochs_list.append(epochs)

    all_features = np.array([], dtype=np.float64)
    all_response = np.array([], dtype=np.float64)
    for epochs in epochs_list:
        features = epochs.get_data(copy=False)
        response = epochs.events[:, 2]
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



# The following code is referred from this kaggle notebook:
# https://www.kaggle.com/code/xevhemary/eeg-pytorch/notebook
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.F1 = 64  # F1: num of temporal filter
        self.D  = 4   # D:  depth (num of spatial filter)
        self.F2 = 256 # F2: num of pointwise filter = F1 * D
        self.C  = 32  # C:  num of channels
        self.N  = 2   # N:  num of classes (1: ERP signal, 0: non-ERP)
        self.T  = 195 # T:  num of timestamps

        # Conv2d(in,out,kernel,stride,padding,bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 64),
                      padding=(0, 32), bias=False),
            nn.BatchNorm2d(num_features=self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.F1, out_channels=self.D*self.F1,
                      kernel_size=(self.C, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.D*self.F1, out_channels=self.D*self.F1,
                      kernel_size=(1, 16), padding=(0, 8), groups=self.D*self.F1,
                      bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(in_features=self.F2*(self.T//32),
                                    out_features=self.N, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)

        x = x.view(-1, self.F2*(self.T//32))
        x = self.classifier(x)
        return x


class Model(object):
    def __init__(self, model=None, lr=0.001):
        super(Model, self).__init__()
        self.model = model
        self.losses = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def fit(self, trainloader=None, validloader=None, epochs=1, monitor=None,
            only_print_finish_ep_num=False):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        doValid = False if validloader == None else True
        pre_ck_point = [float("inf"), 0.0, float("inf"), 0.0, 0]
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        for ep in range(1, epochs + 1):
            proc_start = time.time() # timer start
            if (not (ep % 10)) or (ep == 1):
                if not only_print_finish_ep_num:
                    print(f"Epoch {ep}/{epochs}")
            self.model.train()       # Train mode
            step = 1                 # Restart step
            for x_batch, y_batch in trainloader:
                x_batch  = x_batch.to(device, dtype=torch.float)
                y_batch  = y_batch.to(device)
                pred = self.model(x_batch)
                loss = self.losses(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if (not (ep % 10)) or (ep == 1):
                    pbar = int(step * 30 / len(trainloader))
                    if not only_print_finish_ep_num:
                        print("\r{}/{} [{}{}]".format(
                            step, len(trainloader), ">" * pbar, " " * (30 - pbar)),
                            end="")
                step += 1
            loss, acc = self.evaluate(trainloader)   # Loss & Accuracy
            val_loss, val_acc = self.evaluate(validloader) if doValid else (0, 0)
            history["loss"] = np.append(history["loss"], loss)
            history["acc"] = np.append(history["acc"], acc)
            history["val_loss"] = np.append(history["val_loss"], val_loss)
            history["val_acc"] = np.append(history["val_acc"], val_acc)
            # Update checkpoint
            if self.__updateCheckpoint(monitor, pre_ck_point,
                                       [loss, acc, val_loss, val_acc, ep]):
                pre_ck_point = [loss, acc, val_loss, val_acc, ep]
            if acc > max(history["acc"]):
                save_file_name = "best_checkpoint_model.pt"
                self.save(save_file_name)
            if only_print_finish_ep_num and (ep % 50 == 0):
                print(f"{ep} ", end=" ")
        return history

    def evaluate(self, dataloader):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        total, acc = 0, 0
        self.model.eval()           # Eval mode
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device)
            pred = self.model(x_batch)
            loss = self.losses(pred, y_batch).item()
            total += y_batch.shape[0]     # Number of data
            acc += (torch.sum(pred.argmax(dim=1)==y_batch)).item()
        acc /= total     # Accuracy = correct prediction / number of data
        return (loss, acc)

    def predict(self, dataset):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        prediction = []
        truth = []
        self.model.eval()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device)
            pred = self.model(x_batch).cpu()
            prediction = np.append(prediction, pred.argmax(dim=1).numpy())
            truth = np.append(truth, y_batch.cpu().numpy())
        return prediction, truth

    def save(self, filepath):
        torch.save(self.model, filepath)

    def load(cls, filepath):
        return cls(torch.load(filepath))

    def __updateCheckpoint(self, monitor, pre_ck_point, evaluation):
        if type(monitor) is int:
            return True if evaluation[4] % monitor == 0 else False
        elif type(monitor) is list:
            for _ in monitor:
                if not _ in ["loss", "acc", "val_loss", "val_acc"]:
                    raise Exception(f"\"{_}\" is not a valid monitor condition.")
                elif _ == "loss" and pre_ck_point[0] <= evaluation[0]:
                    return False # present epoch loss > history loss
                elif _ == "acc" and pre_ck_point[1] >= evaluation[1]:
                    return False # present epoch acc <= history acc
                elif _ == "val_loss" and pre_ck_point[2] <= evaluation[2]:
                    return False # present epoch val_loss > history val_loss
                elif _ == "val_acc" and pre_ck_point[3] >= evaluation[3]:
                    return False # present epoch val_acc < history val_acc
        return True


def plot_acc_and_loss(history, figsize=(10,4)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.title.set_text("Accuracy")
    ax1.set_xlabel("Epochs")
    l1 = ax1.plot(history["acc"], color=MY_BLUE, label='train')
    l2 = ax1.plot(history["val_acc"], color=MY_GOLD, label='test')
    ax2.title.set_text("Loss")
    ax2.set_xlabel("Epochs")
    l3 = ax2.plot(history["loss"], color=MY_BLUE, label='train')
    l4 = ax2.plot(history["val_loss"], color=MY_GOLD, label='test')

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    plt.show()