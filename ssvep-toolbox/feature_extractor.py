from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import sosfiltfilt, butter

class FeatureExtractor(ABC):
    """Abstract base class for all feature extraction methods"""
    
    def __init__(self):
        """Setting all object attributes to valid initial values"""
        self.all_signals = None 
        self.signals_count = 0
        self.electrodes_count = 0
        self.features_count = 1
        self.embedding_dimension = 0
        self.delay_step = 0
        self.samples_count = 0
        self.filter_order = 0
        self.cutoff_frequency_low = 0
        self.cutoff_frequency_high = 0
        self.sampling_frequency = 0
        self.subbands = None
        self.subbands_count = 1
        self.is_filterbank = False
        self.voters_count = 1
        self.random_seed = 0
        self.channel_selections = None
        self.channel_selection_info_bundle = 0
        self.max_batch_size = 16
        self.explicit_multithreading = 0
        self.class_initialization_is_complete = False

    @abstractmethod
    def build_feature_extractor(self, **kwargs):
        """Set up the parameters of the feature extractor"""
        pass

    @abstractmethod
    def extract_features(self, all_signals):
        """Extract features from the input signals"""
        pass

    @abstractmethod
    def get_features(self, device=0):
        """Extract features based on specific method"""
        pass

    @abstractmethod
    def perform_voting_initialization(self, device=0):
        """Initialize voting-related operations"""
        pass

    @abstractmethod
    def class_specific_initializations(self):
        """Perform class-specific initializations"""
        pass

    def bandpass_filter(self):
        """Filter the given signal using Butterworth IIR filter"""
        if self.filter_order == 0 or self.cutoff_frequency_high == 0:
            return
                            
        sos = butter(self.filter_order,
                     [self.cutoff_frequency_low, self.cutoff_frequency_high],
                     btype='bandpass',
                     output='sos',
                     fs=self.sampling_frequency)
        
        self.all_signals = sosfiltfilt(sos, self.all_signals, axis=-1)

    def select_channels(self):
        """Select channels based on voting configuration"""
        if self.voters_count > 1:
            self.channel_selections = self.generate_random_selection()
        else:
             self.channel_selections = np.array([True]*self.electrodes_count)
             self.channel_selections = np.expand_dims(
                 self.channel_selections, axis=0)
             
        self.channel_selections = self.channel_selections.astype(bool)    
        
        selection_size = np.sum(self.channel_selections, axis=1)
        sorted_index = np.argsort(selection_size)
        self.channel_selections = self.channel_selections[sorted_index]

    def generate_random_selection(self):
        """Generate random channel selections"""
        random_generator = np.random.default_rng(self.random_seed)
        
        random_channels_indexes = np.zeros(
            (self.voters_count, self.electrodes_count))
                
        while True:
            rows_with_zeros_only = (
                np.sum(random_channels_indexes, axis=1) == 0)
            
            if not rows_with_zeros_only.any():
                break
                        
            random_channels_indexes[rows_with_zeros_only] = (
                random_generator.choice(
                    [True, False],
                    size=(np.sum(rows_with_zeros_only), self.electrodes_count),
                    replace=True)
                )
        
        return random_channels_indexes

    # Property decorators would go here, similar to the original class
    # but simplified for the abstract base class
