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

            :param seed: Random seed for reproducibility.

                :default: 42

            :param lr: The learning rate for the optimizer.

                :default: 5e-4

            :param batch_size: The batch size to use for the dataloaders.

                :default: 32

            :param v_dropout_rate: Dropout rate of the mapping from the timm image model to
                the dimension of the fusion method.

                :default: 0.25

            :param t_dropout_rate: Dropout rate of the mapping from the huggingface text
                model to the dimension of the fusion method.

                :default: 0.25

            :param fusion_dropout_rate: Drop rate inside all classification head layers.

                :default: 0.25

            :param drop_rate: Dropout rate for timm models.

                :default: 0.2

            :param drop_path_rate: Drop path rate for timm models.

                :default: None

            :param fusion_in: Input dimension to the fusion method.

                :default: 512

            :param fusion_out: Output dimension of the fusion method. If None, output will
                be same as input (e.g. for point-wise operations).

                :default: None

            :param fusion_hidden: Number of neurons inside the hidden layer of the
                classification head.

                :default: 256

            :param custom_fusion_method: Fusion method to combine text and image features. Callable
                with two inputs (tensor, tensor) and a single output (tensor) where each
                tensor is single dimension (plus batch dimension). First input is flatten
                output of image model with dimension fusion_in, second input is flatten
                output of the text model with dimension fusion_in. Output should have the
                dimension fusion_out. Only used for initialization. After initialization, the
                fusion method is accessible via the fusion_method property.
                If a custom function is used, it has to be a callable with two inputs (tensor, tensor) and a single output
                (tensor) where each tensor is single dimension (plus batch dimension). The fusion method is then
                passed as a tuple (str, f) where str is the string representation of the function and f is the
                callable, e.g. ("mul", torch.mul).

                :default: None

            :param custom_fusion_activation: Activation function inside all classification head
                layers. Only used for initialization. After initialization, the fusion activation is
                accessible via the fusion_activation property. Default options from torch.nn or torch.nn.functional
                can be used as strings (e.g. "nn.Tanh()") or as callables (e.g. nn.Tanh()).
                If a custom
                function is used, it has to be a callable with a single input (tensor) and a single output
                (tensor) where the input tensor is single dimension (plus batch dimension). The activation is then
                passed as a tuple (str, m) where str is the string representation of the function and m is the
                callable, which has to be a torch.nn.Module and initialized with the correct parameters, e.g.
                ("tanh", nn.Tanh()).

                :default: None

            :param train_transforms: A callable that is used to transform the training images.
                If None is provided, the default train transform is used, that consists of
                resizing, horizontal and vertical flipping and normalization.

                :default: None

            :param shuffle: Whether to shuffle the data. If None is provided, the data is shuffled
                for training and not shuffled for validation and test.

                :default: None

            :param loss_fn: The loss function to use for training.

                :default: F.cross_entropy

            :param class_weights: A manual rescaling weight given to each class. If given,
                has to be a Tensor of size C (number of classes).

                :default: None

            :param patience: Number of epochs with no improvement after which training may be stopped.
                Used for EarlyStopping callbacks.

                :default: 6

            :param optimizer: The optimizer algorithm to use for training the model.

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
        
