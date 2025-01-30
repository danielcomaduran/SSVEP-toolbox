# Standard libraries
import warnings
import numpy as np
from scipy.signal import sosfiltfilt, butter

class Filtering():
    """
        Class for applying digital filters to the given signals
    """

    
    def __init__(
        self, 
        filter_order: int = None,
        cutoff_frequency_low: float = None,
        cutoff_frequency_high: float = None,
        sampling_frequency: float = None,
        ):
                 
        self.filter_order = filter_order
        self.cutoff_frequency_low = cutoff_frequency_low
        self.cutoff_frequency_high = cutoff_frequency_high
        self.sampling_frequency = sampling_frequency

        # Vairables for storing the filter coefficients
        self.sos_matrices = None

    def build_filter(self):
        """
            Build the filter coefficients
        """
        
        if self.cutoff_frequency_low and self.cutoff_frequency_high:
            # Bandstop filter
            if self.cutoff_frequency_low > self.cutoff_frequency_high:
                self.sos_matrices = butter(
                    self.filter_order, 
                    [self.cutoff_frequency_high, self.cutoff_frequency_low], 
                    btype='bandstop', 
                    fs=self.sampling_frequency, 
                    output='sos'
                    )
                
            # Bandpass filter
            else:
                self.sos_matrices = butter(
                    self.filter_order, 
                    [self.cutoff_frequency_low, self.cutoff_frequency_high], 
                    btype='bandpass', 
                    fs=self.sampling_frequency, 
                    output='sos'
                    )
        
        # Highpass filter
        elif self.cutoff_frequency_low:
            self.sos_matrices = butter(self.filter_order, 
                    self.cutoff_frequency_low, 
                    btype='highpass', 
                    fs=self.sampling_frequency, 
                    output='sos'
                    )

        # Lowpass filter    
        elif self.cutoff_frequency_high:
            self.sos_matrices = butter(
                self.filter_order, 
                self.cutoff_frequency_high, 
                btype='lowpass', 
                fs=self.sampling_frequency, 
                output='sos'
                )
            
        else:
            raise ValueError("At least one of cutoff_frequency_low or cutoff_frequency_high must be provided.")
            
    def apply_filter(self, all_signals:np.ndarray) -> np.ndarray:
        """
            Apply the filter to the given signals

            Parameters
            ----------
            all_signals : np.ndarray
                All signals to apply filter to

            Returns
            -------
            np.ndarray
                Filtered signals
        """
        if self.filter_order == 0 or self.cutoff_frequency_high == 0:
            warnings.warn("No filter was performed due to zero filter order or cutoff frequency.")
            return all_signals

        # Apply band-pass filter to all signals
        return sosfiltfilt(self.sos_matrices, all_signals)