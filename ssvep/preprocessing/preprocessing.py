# Standard libraries
from abc import ABC, abstractmethod

# Custom libraries
from . import Filtering
from . import TimeEmbedding

class Preprocessing(ABC):
    #region Error messages
    #endregion

    #region Initialization
    def __init__(
        self,
        filtering:Filtering|None=None,
        time_embedding:TimeEmbedding|None=None,
        ):
        

        self.filtering = Filtering
        self.time_embedding = TimeEmbedding
    #endregion

    #region Public methods
    def band_pass_filter(self):
        
    def extract_features(self, all_signals:np.ndarray):
        """
            Extract features from all given signals

            Parameters
            ----------
            `all_signals` : np.ndarray
                All signals to extract features from
        """
        pass
    #endregion

    #region Private methods
    #endregion

    #region Abstract methods
    #endregion