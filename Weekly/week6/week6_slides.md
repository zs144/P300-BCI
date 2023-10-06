---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
marp: true
---
# **Independent Study Weekly Meeting 6**

#### Learning MNE - the Basics

Zion Sheng
Department of ECE
Duke University

---
## Table of Content

1. Part 1: Progress Made This Week
2. Part 2: Raw EEG data processing
3. Part 3: Some visualization results
4. Part 4: Deeper Thoughts

---
## Part 1: Progess Made This Week
- Experiments with MNE
- Understand key conecpts in MNE
- Try running Jupyter Notebook in VScode through connection to the remote server (DCC)

Note the data we are using is `A01_SE001CB_Train01.edf` downloaded from EDFData folder

---
## Part 2: Raw EEG data processing
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>
### The general procedure
- Read the data to a `Raw` Object
- Load the data
- Check the data overview by `.info` attribute
- Filter the data based on channels
- Find events (mainly target and non-targets)
- Create Epochs

---
## Part 2: The general procedure
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>
### Experiment events and how to find them in data
There are serveral types of experimental events, such as stimulus onset, stimulus type, and participant response (button press). Usually these experimental events are recored in `STIM` channels whose voltages that are time-locked to the events.

We can call `find_events` to extract events information. The sample number of the onset (or offset) of each pulse is recorded as **event time**, the pulse magnitudes are converted into integers to represent the **event**, and these pairs of sample numbers with integer codes are stored in NumPy arrays (usually called “the events array” or just “the events”).

`find_events()` returns a `Numpy.array` with 3 columns. The first column contains the event time in samples. The third column contains the event id. In between the sample number and the integer event code is a value indicating what the event code was on the immediately preceding sample. In practice, that value is almost always 0.

---
## Part 2: The general procedure
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>

### Example on generating an event array
```Python
current_target_events = mne.find_events(raw_data, stim_channel='CurrentTarget', verbose=False)
```
Results:
```
array([[ 1160,     0,     4],
       [ 5896,     0,    18],
       [10632,     0,     9],
       [15368,     0,    22],
       [20104,     0,     9],
       [24840,     0,    14],
       [29576,     0,     7]])
```
---
## Part 2: The general procedure
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>

### Epochs and how to generate

`Epochs` objects are a data structure for representing and analyzing **equal-duration chunks of the EEG/MEG signal**. `Epochs` are most often used to represent data that is time-locked to repeated experimental events (such as stimulus onsets or subject button presses). Inside an `Epochs` object, the data are stored in an array of shape `(n_epochs, n_channels, n_times)`.

The `Raw` object and the events array are the bare minimum needed to create an `Epochs` object, which we create with the `mne.Epochs `class constructor. There are also many parameters to set.

---
## Part 2: The general procedure
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>

### Example on generating an epochs

```Python
## Extract EEG Epochs (see https://mne.tools/stable/auto_tutorials/epochs/index.html)
# Find stimulus events and target stimulus events
stim_events     = mne.find_events(raw=raw_data, stim_channel='StimulusBegin',
                                  verbose=False)
targstim_events = mne.find_events(raw=raw_data, stim_channel='StimulusType',
                                  verbose=False)
# Label target (event_id = 1) and non-target (event_id = 0) events
# Note that the event_id is stored in the third column in events array
targstim_indices = np.isin(stim_events[:,0], targstim_events[:,0])
stim_events[~targstim_indices,2] = stim_events[~targstim_indices,2] - 1
# Epoch data based on target and non-target epoch labels
t_min,t_max = 0, 0.8 # feature extraction window
event_dict = {'target': 1, 'non_target': 0} # stimulus event label -> event_id
epochs = mne.Epochs(raw=raw_data, events=stim_events, tmin=t_min, tmax=t_max,
                    event_id=event_dict, preload=True, baseline=None,
                    proj=False, picks=eeg_channels)
```
non_target: 770
target: 70

---
## Part 3: Some Visualizations
<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}

section {
  font-size: 25px
}
</style>


