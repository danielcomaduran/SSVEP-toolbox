# Standard libraries
from abc import ABC, abstractmethod
import numpy as np

class TemplateSignals():
    """ Class to create template signals to be used in feature extraction. """

    @property
    def template_signal(self):
        """Getter function for the template signals"""
        return self._template_signal


    def __init__(
        self,
        target_frequencies: np.ndarray,
        harmonics_count: int,
        srate: float,
        nsamples: int,
        ):
        """ Constructor for TemplateSignals class. """
        self._targets_frequencies = target_frequencies
        self._harmonics_count = harmonics_count
        self._srate = srate
        self._nsamples = nsamples

        # Preallocate template signals
        self.template_signal = np.empty((
            self._targets_frequencies.shape[0],
            self._harmonics_count*2,
            self._nsamples
            ))


    def compute_templates(self):
        """ Compute the template signals. """

        # Compute time vector
        time = np.arange(self._nsamples) / self._srate

        for (f, freq) in enumerate(self._targets_frequencies):
            self.template_signal[f,0,:] = np.sin(2 * np.pi * freq * time)
            self.template_signal[f,1,:] = np.cos(2 * np.pi * freq * time)
        
        return self.template_signal