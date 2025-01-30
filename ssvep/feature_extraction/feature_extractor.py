# Standard libraries
from abc import ABC, abstractmethod
import numpy as np

# Custom libraries
from ..preprocessing import Preprocessing
from ..gpu_processing import GPUProcessing
from ..batch_processing import BatchProcessing
from ..channel_selection import ChannelSelection
from ..template_signals import TemplateMatching

class FeatureExtractor(ABC):
    #region Error messages
    __TYPE_ERROR_MESSAGE = "Expected type {expected} for {param}, got {actual}"
    __VALUE_ERROR_MESSAGE = "{param} must be {condition}"
    #endregion

    def __init__(
        self,
        pre_processing: Preprocessing | None = None,
        batch_processing: BatchProcessing | None = None,
        channel_selection: ChannelSelection | None = None,
        gpu_processing: GPUProcessing | None = None,
        template_matching: TemplateMatching | None = None,
        ):
        """Base class for SSVEP feature extractors.
        
        Parameters
        ----------
        pre_processing : Preprocessing, optional
            Signal preprocessing module for filtering, etc.
        batch_processing : BatchProcessing, optional
            Handles batch processing of signals
        channel_selection : ChannelSelection, optional
            Channel selection strategy
        gpu_processing : GPUProcessing, optional
            GPU acceleration module
        template_matching : TemplateMatching, optional
            Template matching module
        
        Examples
        --------
        >>> extractor = MyFeatureExtractor(
        ...     pre_processing=MyPreprocessing(),
        ...     batch_processing=MyBatchProcessing()
        ...     )
        """
        # Validate and store dependencies
        self._validate_dependency(pre_processing, Preprocessing, "pre_processing")
        self._validate_dependency(batch_processing, BatchProcessing, "batch_processing")
        self._validate_dependency(channel_selection, ChannelSelection, "channel_selection")
        self._validate_dependency(gpu_processing, GPUProcessing, "gpu_processing")
        self._validate_dependency(template_matching, TemplateMatching, "template_matching")

        # Store validated dependencies
        self.__pre_processing = pre_processing
        self.__batch_processing = batch_processing
        self.__channel_selection = channel_selection
        self.__gpu_processing = gpu_processing
        self.__template_matching = template_matching

    def _validate_dependency(self, value, expected_type, param_name):
        """Validates dependency type."""
        if value is not None and not isinstance(value, expected_type):
            raise TypeError(
                self.__TYPE_ERROR_MESSAGE.format(
                    expected=expected_type.__name__,
                    param=param_name,
                    actual=type(value).__name__
                    )
                )
    
