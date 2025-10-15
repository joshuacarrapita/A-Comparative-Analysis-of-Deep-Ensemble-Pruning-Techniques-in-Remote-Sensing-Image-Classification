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
import pandas as pd
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from configilm.extra.BENv2_utils import band_combi_to_mean_std
from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform


class TrainLightningConfiguration:

    # Class variables (variables that span all classes like an instance counter)

    def __init__(
        self,

        generator: Optional[random.Random] = None,
        seed: int = 42,
        num_classes: int = 19,
        img_size: tuple = (10, 120, 120),
        
        # hyperparameters
        lr: float = 5e-4,
        batch_size: int = 32,
        drop_rate: Optional[float] = 0.2,
        drop_path_rate: Optional[float] = None,
        # etc.: str = #TODO,

        # architecture
        # normalization_type: int = #TODO implement in ConfigILM.py, # responsible for normalization in model architecture. Common normalization techniques include Batch Normalization, Layer Normalization, and Instance Normalization.,
        # hidden_neurons: int = #TODO,
        # etc.: str = #TODO,

        # data
        train_transforms: Optional[Callable] = None,
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

            :param generator: Variable to hold own instance of Random if preferred to use custom generators or generators that don't share state.

                :default: None

            :param seed: Random seed for reproducibility.

                :default: 42
            
            :param num_classes: The number of classes in the dataset.

                :default: 19

            :param img_size: The size of the images (c, h, w). Note that this includes the number of
            channels (c). For example, if the images are RGB images, the size should be (3, h, w). The dataset can be loaded with different channel configurations. The channel configuration is defined by the first element of the img_size tuple (c, h, w). The available configurations are:
                - 2 -> Sentinel-1 (VV, VH)
                - 3 -> RGB
                - 4 -> 10m Sentinel-2 (B, R, G, Ir)
                - 10 -> 10m + 20m Sentinel-2 (in original order)
                - 12 -> Sentinel-1 + 10m/20m Sentinel-2 (in original order)
                - 14 -> Sentinel-1 + 10m/20m/60m Sentinel-2 (in original order)

                :default: (10, 120, 120)

            :param lr: The learning rate for the optimizer.

                :default: 5e-4

            :param batch_size: The batch size to use for the dataloaders.

                :default: 32

            :param drop_rate: Dropout rate for timm models. Applied internally within the layers of the Timm Image Model itself (e.g., standard dropout in the backbone). Standard backbone regularization for the image encoder.

                :default: 0.2

            :param drop_path_rate: Drop path rate for timm models. Applied internally within the layers of the Timm Image Model (specifically common in Vision Transformers and similar architectures).	A technique that randomly drops entire residual paths/branches within the network, often more effective than standard dropout in deep models.

                :default: None

            :param train_transforms: A callable that is used to transform the training images.
                If None is provided, the default train transform is used, that consists of
                resizing, horizontal and vertical flipping and normalization.

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

        self.generator = generator
        self.seed = seed
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.train_transforms = train_transforms
        self.loss_fn = loss_fn
        self.patience = patience
        self.optimizer = optimizer
        if class_weights is None:
            self.class_weights = torch.ones(self.num_classes)
        else:
            self.class_weights = class_weights

        self._initial_params = self.__dict__.copy()
        self._set_seed(seed)
        
    def _set_seed(self, seed, generator: Optional[random.Random] = None) -> None:
        """Sets the random seed for reproducibility."""
        torch.manual_seed(seed)
        if generator is not None: 
            generator.seed(seed)
            if hasattr(generator, 'np'):
                generator.np.random.seed(seed)
        else: 
            random.seed(seed)
            np.random.seed(seed)
        # torch seeds should be set in the main training script
        
    def randomize_all(self, seed: Optional[int] = None) -> None:
        """
        Runs all randomization methods.

        :param seed: The random seed to use for all operations.
        """
        if seed is not None: self._set_seed(seed)
        print("Randomizing all hyperparameters...")
        self.randomize_lr()
        self.randomize_batch_size()
        self.randomize_drop_rate()
        self.randomize_drop_path_rate()
        self.randomize_train_transforms()
        self.randomize_loss_fn()
        self.randomize_class_weights()
        self.randomize_patience()
        self.randomize_optimizer()

    def randomize_lr(self, seed: Optional[int] = None) -> None:
        """Randomizes the learning rate on a logarithmic scale."""
        if seed is not None: self._set_seed(seed)
        # Log-uniform distribution is common for learning rate searches, commonly 10^-5 <= lr <= 10^-1
        self.lr = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2))))

    def randomize_batch_size(self, seed: Optional[int] = None) -> None:
        """Randomizes the batch size from a set of common values (powers of 2)."""
        if seed is not None: self._set_seed(seed)
        self.batch_size = random.choice([16, 32, 64, 128, 256, 512, 1024])

    def randomize_drop_rate(self, seed: Optional[int] = None) -> None:
        """Randomizes the timm model's drop rate."""
        if seed is not None: self._set_seed(seed)
        # often 0, 0.1 or 0.2
        self.drop_rate = random.uniform(0.1, 0.5)

    def randomize_drop_path_rate(self, seed: Optional[int] = None) -> None:
        """Randomizes the timm model's drop path rate."""
        if seed is not None: self._set_seed(seed)
        # drop_path_rate often benefits from being 0 or a low value
        self.drop_path_rate = random.choice([None, random.uniform(0.0, 0.2)])

    def randomize_train_transforms(self, seed: Optional[int] = None) -> None:
        """
        Randomizes the image augmentation pipeline for training.
        A set of potential augmentations is defined, and a random subset is
        chosen and composed.
        """
        if seed is not None: self._set_seed(seed)

        # TODO: Add or modify custom transformation functions 
        potential_transforms = [
            MyGaussianNoise(sigma=20.0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            MyRotateTransform([0, 90, 180, 270]),
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # TODO: Check if it's possible to make it work on 10 Channel and not only RGB
            # T.RandomAffine(degrees=0, translate=(0.1, 0.1)), not sure if this is applicable as it will result in image containing black space
            # T.GaussianBlur(kernel_size=3,sigma=(0.1,2.0)), # TODO: check what small kernel size and sigma is and select that
            # T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0), # TODO: Check that the values make sense. If your data is normalized to N(0,1) or similar, value=0 is correct, as it masks the erased area with the dataset's approximate mean. If your data is not normalized (e.g., pixel values from 0-255 or 0-1), value should be a tuple of 10 values corresponding to the approximate mean of each of your 10 channels. This prevents the erased patch from acting as an obvious artifact.
        ]

        # Randomly select a number of transforms to apply (e.g., 2 to 4)
        num_transforms_to_apply = random.randint(2, len(potential_transforms))
        selected_transforms = random.sample(potential_transforms, num_transforms_to_apply)

        # Basic transforms that are almost always needed
        ben_mean, ben_std = band_combi_to_mean_std(self.img_size[0])

        final_transforms = [
            T.Resize((self.img_size[1], self.img_size[2]), antialias=True),
            *selected_transforms,
            T.Normalize(mean=ben_mean, std=ben_std),
        ]

        self.train_transforms = T.Compose(final_transforms)

    def randomize_loss_fn(self, seed: Optional[int] = None) -> None:
        """
        Randomizes the loss function from a predefined list.
        """
        if seed is not None: self._set_seed(seed)

        # TODO: Implement and add custom loss functions if needed (e.g., FocalLoss)
        # class FocalLoss(nn.Module):
        #     ...
        # focal_loss = FocalLoss()

        # Predefine cross_entropy with a random label smoothing value sampled once and remaining constant for the entire training run
        random_smoothing_factor = random.uniform(0.0, 0.2)
        ce_smooth_random = partial(F.cross_entropy, label_smoothing=random_smoothing_factor)

        loss_functions = [
            F.cross_entropy,
            ce_smooth_random,
            F.multi_margin_loss,
        ]
        
        self.loss_fn = random.choice(loss_functions)
    
    # TODO: Could I not randomize these when I create a partial loss function already?
    def randomize_class_weights(self, seed: Optional[int] = None) -> None:
        """
        Randomizes the class weights for the loss function.
        This can either set weights to None (no weighting) or create a tensor
        of random weights.
        """
        if seed is not None: self._set_seed(seed)
        use_weights = random.choice(['no', 'per-class', 'global-scalar', 'ones'])

        if use_weights == 'per-class':
            # Generate a random weight scalar, e.g., between 0.5 and 2.0 for each class
            scalars = torch.FloatTensor([random.uniform(0.5, 2.0) for _ in range(self.num_classes)])
            self.class_weights = self.class_weights * scalars
        elif use_weights == 'global-scalar':
            self.class_weights = self.class_weights * random.uniform(0.5,2.0)
        elif use_weights == 'ones':
            self.class_weights = torch.ones(self.num_classes)
        
    def randomize_optimizer(self, seed: Optional[int] = None) -> None:
        """Randomizes the optimizer."""
        if seed is not None: self._set_seed(seed)
        optimizers = [
            torch.optim.AdamW,
            torch.optim.Adam,
            torch.optim.SGD,
            torch.optim.RMSprop
        ]
        self.optimizer = random.choice(optimizers)

    def randomize_patience(self, seed: Optional[int] = None) -> None:
        """Randomizes the early stopping patience."""
        if seed is not None: self._set_seed(seed)
        self.patience = random.randint(3, 10)

    def info(self) -> None:
        """
        Prints a formatted table of all configuration parameters, their standard
        (initial) values, and their current (potentially randomized) values
        using the pandas library for clean formatting.
        """
        print("\n--- Configuration Parameter Info ---")

        table_data = []

        # Exclude the tracking dictionary itself from the report
        params_to_report = {k: v for k, v in self._initial_params.items() if k != '_initial_params'}

        for param, standard_value in params_to_report.items():
            current_value = getattr(self, param)

            def format_value(val):
                """Helper to create a nice string representation for different types."""
                if hasattr(val, '__name__'):
                    return val.__name__
                if isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], str):
                    return val[0] # For tuples like ("mul", torch.mul)
                if isinstance(val, torch.Tensor):
                    return f"Tensor(shape={val.shape})"
                if val is None:
                    return "None"
                if isinstance(val, float):
                    return f"{val:.6f}"
                return str(val)

            standard_value_str = format_value(standard_value)
            current_value_str = format_value(current_value)

            if standard_value_str == current_value_str:
                new_value_display = "Not Randomized"
            else:
                new_value_display = current_value_str

            table_data.append([param, standard_value_str, new_value_display])

        # --- Create and print the table using pandas ---
        df = pd.DataFrame(table_data, columns=["Parameter", "Standard Value", "New Value"])

        with pd.option_context('display.max_rows', None, 'display.width', 1000):
            print(df)
        print("-" * 40)

if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Create a config and show its initial state
    print("--- 1. Initial Configuration State ---")
    config = TrainLightningConfiguration()
    config.info()
    
    # 2. Randomize all parameters and show the results
    print("\n--- 2. Randomizing All Parameters ---")
    config.randomize_all(seed=123)
    config.info()
    
    # 3. Create a new instance and randomize only one parameter
    print("\n--- 3. New Config, Randomizing Only Learning Rate ---")
    config2 = TrainLightningConfiguration(seed=99)
    config2.randomize_lr()
    config2.info()

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
