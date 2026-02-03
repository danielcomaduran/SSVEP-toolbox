"""
    Data tools
        Set of functions to organize and import data from multiple types of datasets.
"""

# Import libraries
import os
import mne
import pyxdf
import numpy as np
import pandas as pd
import scipy.signal as signal
from collections import OrderedDict

# Set log level for MNE
mne.set_log_level(verbose=False)

def create_label_mask(labels: list[str], n: int, seed: int = None) -> np.ndarray:
    """
    Creates a mask of n-distinct labels from a list of labels by randomly picking the labels.

    Parameters
    ----------
    labels : list[str]
        List of labels.
    n : int
        Number of distinct labels to include in the mask.
    seed : int, optional
        Seed number for reproducibility, by default None.

    Returns
    -------
    mask : np.ndarray
        Mask of n-distinct labels.
    """
    unique_labels = np.unique(labels)
    if seed is not None:
        np.random.seed(seed)
    selected_labels = np.random.choice(unique_labels, size=n, replace=False)
    mask = np.isin(labels, selected_labels)
    return mask

def moabb_events_to_np(mne_raw:mne.io.Raw,
                       tmin:float,
                       tmax:float,
                       events_dict:dict=None,
                       chans=None) -> tuple[np.ndarray, np.ndarray]:
    """
        This function takes data from the MOABB package, creates epochs based on the stimulus vector,
        and returns the 3D matrix with shape [epochs, channels, samples]

        Parameters
        ----------
            `mne_raw`: mne.io.fif.raw.Raw
                MNE object containing the EEG data with the stimulus vector
            `tmin`: float
                Start time of the epochs [sec]
            `tmax`: float
                End time of the epochs [sec]
            `events_dict`: dict
                Dictionary containing events to use. Values must match event ID of stimulus vector.
            `chans`: list[str] | None
                List of strings with the names of the channels to pick. If None, use all channels

        Returns
        -------
            epoched_data: np.ndarray
                Epoched data in 3D numpy array [epochs, channels, samples]
    """
    # Extract event data
    events = mne.find_events(mne_raw)
    
    # If we only want certain channels
    if chans != None:
        mne_raw = mne_raw.pick(chans)

    # Create epochs
    mne_epoched = mne.Epochs(mne_raw,
                             events=events,
                             event_id=events_dict,
                             baseline=None,
                             tmin=tmin,
                             tmax=tmax,
                             verbose=False)
    epochs_np = mne_epoched.get_data()

    # If events dict exist, return only events of interest
    if events_dict is not None:
        mask = np.isin(events[:,2], list(events_dict.values()))
        events = events[mask,:]

    return (events, epochs_np)


def select_importer(file: str, picks: list[str]="all"):
    """
        Automatically selects the right function to import data

        Parameters
        ----------
            file: str
                Complete file name of the file to import. Must have the file extension
            picks: list[str]
                List of strings with names of channels to import. Defaults to "all" channels
    """

    function_dict = {
        "edf":read_edf,
        "txt":read_openBCI,
        "xdf":read_xdf,
        }

    symbol = "\\"
    folder = symbol.join(file.split(symbol)[:-1])

    for format in function_dict.keys():
        temp_file = f"{file.split(symbol)[-1]}.{format}"
        if (temp_file in os.listdir(folder)):
            break

    extension = temp_file.split(".")
   

    [eeg, srate] = function_dict[extension[-1]](f"{file}.{extension[1]}", picks)

    return eeg, srate


def read_edf(file: str, picks: list[str] = ["all"]):
    """
        Imports a .EDF and returns the data matrix [channels x samples] and sample rate [Hz]
        
        Parameters
        ----------
            - file: str
                Full directory of file to import
            - picks: list
                List of strings with the names of the channels to import. Default will import all channels

        Returns
        -------
            - eeg: np.ndarray [channels x samples]
                EEG raw data
            - srate: double
                Sampling rate [Hz]

    """

    # if file.split(".")[-1] != "edf":
        # file = f"{file}.edf"

    edf_data = mne.io.read_raw_edf(file, verbose=False)
    eeg = edf_data.get_data(picks)       # EEG [V]
    srate = edf_data.info['sfreq']  # Sampple rate [Hz]

    return eeg, srate

def read_openBCI(file: str, picks: list[str] = "all"):
    """
        Imports a .TXT file and returns the data matrix [channels x samples] and sample rate [Hz]

        Parameters
        ----------
            - file: str
                Full directory of the file to import
            - picks: list[str] = ["all"]
                List of strings with the names of the channels to import. Default will import all EEG channels

        Returns
        -------
            - eeg: np.ndarray [channels x samples]
                EEG raw data
            - srate: double
                Sampling rate [Hz]
    """

    full_data = pd.read_csv(file, header=4)

    f = open(file)
    content = f.readlines()
    nchans = int(content[1].split(" = ")[1])                # Number of channels [int]
    srate = float(content[2].split(" = ")[1].split(" ")[0]) # Sampling rate [Hz]
    
    # Select only EEG channels or a subset of EEG channels
    eeg = full_data.iloc[:,1:nchans+1]
    chans_dict = {
        " EXG Channel 0":"FP1", " EXG Channel 1":"FP2", " EXG Channel 2":"F7", " EXG Channel 3":"F3",
        " EXG Channel 4":"F4", " EXG Channel 5":"F8", " EXG Channel 6": "T7", " EXG Channel 7":"C3", 
        " EXG Channel 8":"C4", " EXG Channel 9":"T8", " EXG Channel 10":"P7", " EXG Channel 11":"P3",
        " EXG Channel 12":"P4", " EXG Channel 13":"P8", " EXG Channel 14":"O1", " EXG Channel 15":"O2"
        }
    eeg.rename(columns=chans_dict, inplace=True)

    if picks != "all":
        eeg = eeg[picks]
    
    return eeg.to_numpy().T, srate

def read_xdf(file: str, picks: list[str]="all"):
    """
        Imports a .XDF file and returns the data matrix [channels x samples] and sample rate [Hz]

        Parameters
        ----------
            - file: str
                Full directory of the file to import
            - picks: list[str] = ["all"]
                List of strings with the names of the channels to import. Default will import all EEG channels
            - return_marker_data: bool
                If enabled, the function also returns the marker data and time stamps

        Returns
        -------
            - `eeg_ts`: EEG time stamps [sec]
            - `eeg`: np.ndarray [channels x samples]
                EEG raw data
            - `srate`: double
                Sampling rate [Hz]
            
    """
    file_path = os.path.normpath(file)  # Normalize path OS agnostic
    [data, header] = pyxdf.load_xdf(file_path, verbose=False)
    
    for stream in data:
        # Obtain data for SMARTING headset
        if (stream["info"]["source_id"][0]=="SMARTING" and stream["info"]["type"][0]=="EEG"):
            eeg_ts = stream["time_stamps"]
            eeg_np = stream["time_series"]
            srate = float(stream["info"]["nominal_srate"][0])
            break

        source_id_list = stream["info"]["source_id"][0].split("_")
        if source_id_list[0] == 'gUSBamp' and source_id_list[-1] != "markers":
            eeg_ts = stream["time_stamps"]
            eeg_np = stream["time_series"]
            srate = float(stream["info"]["nominal_srate"][0])
            break


    # Obtained from:
    # - https://mbraintrain.com/wp-content/uploads/2021/02/RBE-24-STD.pdf
    n_chans = len(stream['info']['desc'][0]['channels'][0]['channel'])
    chans_names = [stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(n_chans)]

    eeg_pd = pd.DataFrame(data=eeg_np, columns=chans_names)

    if picks != "all":
        eeg_pd = eeg_pd[picks]                    

    return eeg_ts, eeg_pd.to_numpy().T, srate

def read_xdf_unity_markers(file: str) -> tuple[np.ndarray, list[str]]:
    """
        This function returns the time stamps and markers from the Unity stream of an xdf file

        Returns
        -------
            - `marker_time`. Numpy vector with the time stamps of the Unity stream markers.
            - `marker_data`. List with the string of markers.
    """

    file_path = os.path.normpath(file)  # Normalize path OS agnostic
    [data, _] = pyxdf.load_xdf(file_path, verbose=False)

    for stream in data:
        if stream["info"]["name"][0] == 'UnityMarkerStream':
            marker_time = stream["time_stamps"]
            marker_data = stream["time_series"]  

    return marker_time, marker_data


def epochs_from_unity_markers(
    eeg_time: np.ndarray,
    eeg_data: np.ndarray,
    marker_time: np.ndarray,
    marker_data: list[str]
    ) -> tuple[list[list[np.ndarray]], list]:
    """
        This function returns a list of EEG epochs and a list of marker names, based on
        the marker data provided.

        Notes
        -----
            - The marker data must have repeated markers
    """

    # Make sure that data is in shape [samples, channels]
    if eeg_data.shape[0] < eeg_data.shape[1]:
        eeg_data = eeg_data.T

    # Initialize empty list
    eeg_epochs = []

    (repeated_markers, repeated_labels) = find_repeats(marker_data)

    # Trim EEG data to marker data times
    for m in range(np.shape(repeated_markers)[0]):
        eeg_mask_time = (eeg_time >= marker_time[repeated_markers[m, 0]]) & (
            eeg_time <= marker_time[repeated_markers[m, 1]]
        )

        eeg_epochs.append(eeg_data[eeg_mask_time, :])

    return (eeg_epochs, repeated_labels)


def find_repeats(marker_data: list) -> tuple[np.ndarray, list]:
    """
    Finds the repeated values in the marker data

    Returns
    -------
        - `repeats`: Numpy array with n-rows for repeated values [start, stop]
        - `order`: List with the `marker_data` labels of the repeated values.
    """

    repeats = []
    start = None

    for i in range(len(marker_data) - 1):
        if marker_data[i] == marker_data[i + 1]:
            if start is None:
                start = i
        elif start is not None:
            repeats.append((start, i))
            start = None

    if start is not None:
        repeats.append((start, len(marker_data) - 1))

    repeats = np.array(repeats)
    labels = [marker_data[i][0] for i in repeats[:, 0]]

    return repeats, labels

def fix_labels(labels: list[str]) -> list[str]:
    """
        Fix labels in pilot data (e.g., "tvep,1,-1,1,2Min", should be 
        "tvep,1,-1,1,2, Min")

        Parameters
        ----------
            labels: list[str]
                Original set of labels found in Unity LSL stream

        Returns
        -------
            fixed_labels: list[str]
                List of labels with mistakes fixed
    """

    # Preallocate output
    fixed_labels = []

    for label in labels:
        if label == "tvep,1,-1,1,2Min":
            fixed_labels.append("tvep,1,-1,1,2, Min")
        elif label == "tvep,1,-1,1,9.6Min":
            fixed_labels.append("tvep,1,-1,1,9.6, Min")
        elif label == "tvep,1,-1,1,16Min":
            fixed_labels.append("tvep,1,-1,1,16, Min")
        elif label == "tvep,1,-1,1,36Min":
            fixed_labels.append("tvep,1,-1,1,36, Min")
        else:
            fixed_labels.append(label)

    return fixed_labels

def get_tvep_stimuli(labels: list[str]) -> dict:
    """
        Returns a dictionary of unique labels of the stimulus of labels that begin with "tvep"

        Parameters
        ----------
            labels: list[str]
                Complete list of labels from Unity markers

        Returns
        -------
            unique_labels: list[str]
                List of unique labels of stimulus that begin with "tvep"
    """

    tvep_labels = []

    for label in labels:
        if label.split(",")[0] == "tvep":
            tvep_labels.append(label.split(",")[-1])
    
    dict_of_stimuli = OrderedDict({i: v for i, v in enumerate(list(set(tvep_labels)))})

    return dict_of_stimuli

def epochs_stim_freq(
    eeg_epochs: list,
    labels: list,
    stimuli: dict,
    freqs: dict,
    mode: str = "trim",
    ) -> list:
    """
        Creates EEG epochs in a list of lists organized by stimuli and freqs

        Parameters
        ----------
            eeg_epochs: list 
                List of eeg epochs in the shape [samples, chans]
            labels: list
                Complete list of labels from Unity markers
            stimuli: dict
                Dictionary with the unique stimuli labels
            freqs: dict
                Dictionary with the uniquie frequency labels
            mode: str
                Mode to convert all epochs to the same length,'trim' (default) or 'zeropad'

        Returns
            eeg_epochs_organized: list
                List of organized eeg epochs in the shape [stimuli][freqs][trials][samples, chans]
    """
    # Preallocate list for organized epochs
    eeg_epochs_organized = [[[] for j in range(len(freqs))] for i in range(len(stimuli))]
    mode_options = {"trim": np.min, "zeropad": np.max}
    mode_nsamples = {"trim": np.inf, "zeropad": 0}
    min_samples = np.inf

    # Organize epochs by stim and freq
    for e, epoch in enumerate(labels):
        for s, stim in stimuli.items():
            for f, freq in freqs.items():
                if epoch == f"tvep,1,-1,1,{freq},{stim}":
                    eeg_epochs_organized[s][f].append(np.array(eeg_epochs[e]))

                    # Get number of samples based on mode
                    nsamples = int(mode_options[mode]((mode_nsamples[mode], eeg_epochs[e].shape[0])))
                    mode_nsamples[mode] = nsamples

    # Change length of array based on mode
    for s, _ in stimuli.items():
        for f, _ in freqs.items():
            for t in range(3):  # For each trial
                if (mode == "trim"):
                    eeg_epochs_organized[s][f][t] = eeg_epochs_organized[s][f][t][:min_samples, :].T
                elif (mode == "zeropad"):
                    pad_length = nsamples - eeg_epochs_organized[s][f][t].shape[0]
                    pad_dimensions = ((0, pad_length), (0, 0))
                    eeg_epochs_organized[s][f][t] = np.pad(eeg_epochs_organized[s][f][t], pad_dimensions, 'constant', constant_values=0).T

    return np.array(eeg_epochs_organized)

def labels_to_dict_and_array(labels: list) -> tuple[dict, np.ndarray]:
    """
        Returns dictionary of labels with numeric code and numpy
        array with the label codes
    """
    # Using dictionary comprehension to directly create the label_dict
    label_dict = {label: idx for idx, label in enumerate(set(labels))}
    
    # Create a numpy array with the codes of the strings
    arr = np.array([label_dict[label] for label in labels])
    
    return label_dict, arr

def trim_epochs(epochs:list) -> np.ndarray:
    """
        Takes a list of epochs of different length and trims to the shorter
        epoch. 
        
        Returns
        -------
            trimmed_epochs: array with shape [epochs, channels, samples]
    """
    # Initialize samples and channels counter
    min_samples = np.inf
    nchans = np.zeros(len(epochs), dtype=np.int16)
    
    # Get number of minimum samples
    for [e,epoch] in enumerate(epochs):
        epoch_shape = epoch.shape
        epoch_len = int(np.max(epoch_shape))
        nchans[e] = int(np.min(epoch_shape))

        min_samples = int(np.min((min_samples, epoch_len)))

    # Check that all epochs have same number of channels
    if (np.sum(np.abs(np.diff(nchans))) != 0):
        print("Not all epochs have the same number of channels")
        return None

    # Preallocate and fill output array
    trimmed_epochs = np.zeros((len(epochs), nchans[0], min_samples))
    for [e,epoch] in enumerate(epochs):
        # Make sure epoch is in shape [chans, samples]
        epoch_shape = epoch.shape
        if epoch_shape[0] > epoch_shape[1]:
            epoch = epoch.T

        trimmed_epochs[e,:,:] = epoch[:,:min_samples]

    return trimmed_epochs

def label_by_stim_type(
    full_labels: list[str],
    stim_type: str,
    frequency_dict: dict,    
    ) -> np.ndarray:
    """
        Looks into the `full_labels` list and returns a numpy array with
        the keys of the `frequency_dict` for the selectec `stim_type`
    """
    filtered_freqs = []

    # Filter list to get only subset of stimulus of interest
    for label in full_labels:
        label_stim = label.split(",")[-1]

        if (label_stim == stim_type):
            filtered_freqs.append(label.split(",")[4])

    # Match order of stim with dictionary
    output_labels = []
    for freq in filtered_freqs:
        if freq in frequency_dict.values():
            key = list(frequency_dict.keys())[list(frequency_dict.values()).index(freq)]
            output_labels.append(key)

    return np.array(output_labels)

def import_system_agnostic(file: str):
    """
        Imports a file in a system-agnostic way by converting the file path to the appropriate format for the current operating system.

        Parameters
        ----------
            - file: str
                Full directory of the file to import

        Returns
        -------
            - file_path: str
                System-agnostic file path
    """
    file_path = os.path.normpath(file)

    return file_path