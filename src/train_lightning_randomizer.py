"""
Randomizer for the train_lightning script. Sets the seed and randomizes 
the following parameters in order to create diversity between trained 
models: 

    - verschiedene Data Augmentations
    - leicht unterschiedliche Hyperparameter (Lernrate, dropout rates, batch size, etc.)
    - unterschiedliches dropout
    - kleine Änderungen der Architektur (layer width, normalization type, ...)
    - unterschiedliches shuffling der Trainingsdaten
    - Variation in den loss functions (zB leicht unterschiedliche Klassengewichte und andere loss functions (dice, focal)
    - Variation in den optimizern (Adam vs. AdamW vs. SGD, Early stopping patience) 
"""

from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union


try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.BENv2_utils import band_combi_to_mean_std
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from configilm.util import Messages


class TrainLightningConfiguration:

    # Class variables (variables that span all classes like an instance counter)

    def __init__(
        self,

        seed: int = 42,
        
        # hyperparameters
        lr: float = 5e-4,
        batch_size: int = 32,
        v_dropout_rate: float = 0.25,
        t_dropout_rate: float = 0.25,
        fusion_dropout_rate: float = 0.25,
        drop_rate: Optional[float] = 0.2,
        drop_path_rate: Optional[float] = None,
        # etc.: str = #TODO,

        # architecture
        fusion_in: int = 512,
        fusion_out: Optional[int] = None,  # if None, same as fusion_in
        fusion_hidden: int = 256, # Number of neurons inside the hidden layer of the classification head a.k.a. layer width
        #_fusion_method: str = "torch.mul", # DONE: Find out what the "_" does --> it indicates that this is supposed to be regarded as a private variable, "whatever that means in python"
        #_fusion_activation: str = "nn.Tanh()", # TODO: Also check if applicable
        custom_fusion_method: Optional[Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str, Tuple[str, Callable]]] = None,
        custom_fusion_activation: Optional[Union[Callable[[torch.Tensor], torch.Tensor], str, Tuple[str, Callable]]] = None,
        # normalization_type: int = #TODO implement in ConfigILM.py, # responsible for normalization in model architecture. Common normalization techniques include Batch Normalization, Layer Normalization, and Instance Normalization.
        # etc.: str = #TODO,

        # data
        train_transforms: Optional[Callable] = None,
        shuffle: Optional[bool] = None,
        # etc.: str = #TODO,

        # loss functions (e.g. different class weights, functions(dice, focal))
        loss_fn: str = #TODO,
        class_weights: dict[str] = #TODO,
        # etc.: str = #TODO,

        # optimizers 
        patience: int = 6,
        optimizer: str = #TODO,
        # etc.: str = #TODO,
    ):
        """
        This configuration class is designed to work with the train_lightning training script. 

        :param seed: Random seed for bagging for reproducibility, default is 42.

            :default: 42
        """
        self.data_dirs = data_dirs
        self.batch_size = batch_size

        self.shuffle = shuffle
        if self.shuffle is not None:
            Messages.warn(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

    def randomize(self, seed: Optional[int] = None, num_param: Optional[int] = None, num_categories: Optional[int] = None) -> None:
        # seed presets the seed used for randomizationnum_param,
        # num_param sets the number of parameters that are supposed to be randomized, can be just one or more,
        # num_categories sets the number of different categories in which parameters should be randomized e.g. 1 will pick one category out of model architecture, hyperparameters, misc, ... etc. (TODO choose categories)
        # MAYBE: Add choose specific params and / or categories to be randomized.
        
