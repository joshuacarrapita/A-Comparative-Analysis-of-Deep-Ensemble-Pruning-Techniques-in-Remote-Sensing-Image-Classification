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
        loss_fn: callable = F.cross_entropy,
        class_weights: Optional[torch.Tensor] = None,
        # etc.: str = #TODO,

        # optimizers 
        patience: int = 6,
        optimizer: callable = torch.optim.AdamW,
        # etc.: str = #TODO,
    ):
        """
        This configuration class is designed to work with the train_lightning training script.

        :param seed: Random seed for ensuring reproducibility across runs. This affects
            weight initialization, dropout, and data shuffling.

            :default: 42

        :param lr: The learning rate for the optimizer. It controls how much the model's
            weights are adjusted with respect to the loss gradient.

            :default: 5e-4

        :param batch_size: The number of samples processed before the model is updated.

            :default: 32

        :param v_dropout_rate: Dropout rate for the visual branch of the model. Dropout is a
            regularization technique to prevent overfitting.

            :default: 0.25

        :param t_dropout_rate: Dropout rate for the text branch of the model.

            :default: 0.25

        :param fusion_dropout_rate: Dropout rate applied after the fusion of multimodal features.

            :default: 0.25

        :param drop_rate: General dropout rate for layers within the model architecture,
            if applicable.

            :default: 0.2

        :param drop_path_rate: Stochastic depth rate. If None, no drop path is applied.
            Drop path randomly drops entire residual blocks during training.

            :default: None

        :param fusion_in: The number of input features to the fusion layer.

            :default: 512

        :param fusion_out: The number of output features from the fusion layer. If None,
            the output dimension is the same as fusion_in.

            :default: None

        :param fusion_hidden: The number of neurons in the hidden layer of the final
            classification head.

            :default: 256

        :param custom_fusion_method: A custom function or method to fuse tensors. If None,
            a default fusion strategy should be implemented in the model.

            :default: None

        :param custom_fusion_activation: A custom activation function to apply after fusion.
            If None, a default activation may be used or none at all.

            :default: None

        :param train_transforms: A callable (e.g., torchvision.transforms.Compose) for
            training data augmentation. If None, default or no transformations are applied.

            :default: None

        :param shuffle: Whether to shuffle the training data. If None, the data is typically
            shuffled for training and not for validation. Manual setting is not recommended.

            :default: None

        :param loss_fn: The loss function to use for training.

            :default: F.cross_entropy

        :param class_weights: A tensor of weights to apply to each class in the loss function,
            useful for imbalanced datasets. If None, all classes are weighted equally.

            :default: None

        :param patience: Number of epochs to wait for improvement on the validation metric
            before stopping the training early.

            :default: 6

        :param optimizer: The optimizer class to use for training.

            :default: torch.optim.AdamW
        """
        
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.v_dropout_rate = v_dropout_rate
        self.t_dropout_rate = t_dropout_rate
        self.fusion_dropout_rate = fusion_dropout_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.fusion_in = fusion_in
        self.fusion_out = fusion_out
        self.fusion_hidden = fusion_hidden
        self.custom_fusion_method = custom_fusion_method
        self.custom_fusion_activation = custom_fusion_activation
        self.train_transforms = train_transforms
        self.loss_fn = loss_fn
        self.class_weights = class_weights
        self.patience = patience
        self.optimizer = optimizer

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
        
