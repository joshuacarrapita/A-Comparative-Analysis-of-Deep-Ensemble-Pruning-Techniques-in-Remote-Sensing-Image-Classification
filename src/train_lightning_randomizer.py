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

import random
from typing import Callable, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


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
        
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Sets the random seed for reproducibility."""
        s = seed if seed is not None else self.seed
        random.seed(s)
        np.random.seed(s)
        # Note: torch seeds should be set in the main training script
        
    def randomize_all(self, seed: Optional[int] = None) -> None:
        """Runs all randomization methods."""
        self._set_seed(seed)
        print("Randomizing all hyperparameters... 🎲")
        self.randomize_lr()
        self.randomize_batch_size()
        self.randomize_dropout_rates()
        self.randomize_fusion_hidden()
        self.randomize_fusion_method()
        self.randomize_fusion_activation()
        self.randomize_optimizer()
        self.randomize_patience()
        self.randomize_shuffle()

    def randomize_lr(self, seed: Optional[int] = None) -> None:
        """Randomizes the learning rate on a logarithmic scale."""
        self._set_seed(seed)
        # Log-uniform distribution is common for learning rate searches
        self.lr = float(np.random.loguniform(1e-5, 1e-2))

    def randomize_batch_size(self, seed: Optional[int] = None) -> None:
        """Randomizes the batch size from a set of common values (powers of 2)."""
        self._set_seed(seed)
        self.batch_size = random.choice([16, 32, 64, 128])

    def randomize_dropout_rates(self, seed: Optional[int] = None) -> None:
        """Randomizes all dropout-related rates."""
        self._set_seed(seed)
        self.v_dropout_rate = random.uniform(0.1, 0.5)
        self.t_dropout_rate = random.uniform(0.1, 0.5)
        self.fusion_dropout_rate = random.uniform(0.1, 0.5)
        self.drop_rate = random.uniform(0.1, 0.5)
        # drop_path_rate often benefits from being 0 or a low value
        self.drop_path_rate = random.choice([None, random.uniform(0.05, 0.2)])
        
    def randomize_fusion_hidden(self, seed: Optional[int] = None) -> None:
        """Randomizes the number of hidden neurons in the fusion layer."""
        self._set_seed(seed)
        self.fusion_hidden = random.choice([128, 256, 512, 1024])

    def randomize_fusion_method(self, seed: Optional[int] = None) -> None:
        """Randomizes the fusion method from a list of common techniques."""
        self._set_seed(seed)
        fusion_methods = [
            ("torch.mul", torch.mul),
            ("torch.add", torch.add),
            # Concatenation would require changing fusion_out, so it's omitted here
        ]
        self.custom_fusion_method = random.choice(fusion_methods)

    def randomize_fusion_activation(self, seed: Optional[int] = None) -> None:
        """Randomizes the activation function in the fusion layer."""
        self._set_seed(seed)
        activations = [
            ("nn.ReLU()", torch.nn.ReLU()),
            ("nn.Tanh()", torch.nn.Tanh()),
            ("nn.GELU()", torch.nn.GELU()),
            ("nn.SiLU()", torch.nn.SiLU()) # Also known as Swish
        ]
        self.custom_fusion_activation = random.choice(activations)
        
    def randomize_optimizer(self, seed: Optional[int] = None) -> None:
        """Randomizes the optimizer."""
        self._set_seed(seed)
        optimizers = [
            torch.optim.AdamW,
            torch.optim.Adam,
            torch.optim.SGD,
            torch.optim.RMSprop
        ]
        self.optimizer = random.choice(optimizers)

    def randomize_patience(self, seed: Optional[int] = None) -> None:
        """Randomizes the early stopping patience."""
        self._set_seed(seed)
        self.patience = random.randint(3, 10)

    def randomize_shuffle(self, seed: Optional[int] = None) -> None:
        """Randomizes the data shuffling flag."""
        self._set_seed(seed)
        self.shuffle = random.choice([True, False])
# --- Example Randomization Function ---
#     def randomize_all(self, seed: Optional[int] = None, num_param: Optional[int] = None, num_categories: Optional[int] = None) -> None:
#         
#         self.lr = random.random()
#         # seed presets the seed used for randomizationnum_param,
#         # num_param sets the number of parameters that are supposed to be randomized, can be just one or more,
#         # num_categories sets the number of different categories in which parameters should be randomized e.g. 1 will pick one category out of model architecture, hyperparameters, misc, ... etc. (TODO choose categories)
#         # MAYBE: Add choose specific params and / or categories to be randomized.

# --- Example Usage ---
# config = TrainLightningConfiguration()
# print(f"Default LR: {config.lr}, Default Optimizer: {config.optimizer.__name__}")
# 
# config.randomize_all(seed=123)
# print(f"Randomized LR: {config.lr:.6f}, Randomized Optimizer: {config.optimizer.__name__}")
# print(f"Randomized Fusion Method: {config.custom_fusion_method[0]}")
