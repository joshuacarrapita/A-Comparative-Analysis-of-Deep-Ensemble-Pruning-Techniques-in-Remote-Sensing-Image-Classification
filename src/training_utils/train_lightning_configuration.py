"""
Randomizer for the train_lightning script. Sets the seed and randomizes 
the following parameters in order to create diversity between trained 
models: 

    - hyperparameters
        - lr
        - batch_size
        - drop_rate
        - drop_path_rate

    - data augmentations
        - train_transforms

    - loss functions 
        - loss_fn
        - class_weights

    - optimizers 
        - patience
        - optimizer
"""
import random
import textwrap
from typing import Callable, Optional, Union, Tuple
from inspect import signature

import numpy as np
import pandas as pd
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from configilm.extra.BENv2_utils import band_combi_to_mean_std
from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform
from configilm.extra._defaults import default_train_transform
from scipy.stats import truncnorm


class TrainLightningConfiguration:

    # Class variables (variables that span all classes like an instance counter)

    def __init__(
        self,

        # --- not-randomized params ---
        img_size: tuple,
        generator: Optional[random.Random] = None,
        seed: int = 42,
        num_classes: int = 16,
        
        # --- randomized params ---
        # hyperparameters
        lr: float = 5e-4,
        batch_size: int = 256,
        drop_rate: Optional[float] = 0.0,
        drop_path_rate: Optional[float] = 0.0,

        # data augmentations
        train_transforms: Optional[Callable] = None,

        # loss functions 
        loss_fn: callable = F.cross_entropy,
        class_distribution: Optional[pd.DataFrame] = None,
        alpha: float = 0.75,

        # optimizers 
        patience: int = 5,
        optimizer: callable = torch.optim.AdamW,
    ):
        """
            This configuration class is designed to work with the 
            train_lightning training script.

            :param img_size: The size of the images (c, h, w). Note that this 
            includes the number of channels (c). For example, if the images 
            are RGB images, the size should be (3, h, w). The dataset can be 
            loaded with different channel configurations. The channel 
            configuration is defined by the first element of the img_size 
            tuple (c, h, w). The available configurations are:
                - 2 -> Sentinel-1 (VV, VH)
                - 3 -> RGB
                - 4 -> 10m Sentinel-2 (B, R, G, Ir)
                - 10 -> 10m + 20m Sentinel-2 (in original order)
                - 12 -> Sentinel-1 + 10m/20m Sentinel-2 (in original order)
                - 14 -> Sentinel-1 + 10m/20m/60m Sentinel-2 (in original 
                    order)

                :example: (10, 120, 120)

            :param generator: Variable to hold own instance of Random if 
                preferred to use custom generators or generators that don't 
                share state.

                :default: None

            :param seed: Random seed for reproducibility.

                :default: 42
            
            :param num_classes: The number of classes in the dataset.

                :default: 16

            :param lr: The learning rate for the optimizer.

                :default: 5e-4

            :param batch_size: The batch size to use for the dataloaders.

                :default: 32

            :param drop_rate: Dropout rate for timm models. Applied 
                internally within the layers of the Timm Image Model itself 
                (e.g., standard dropout in the backbone). Standard backbone 
                regularization for the image encoder.

                :default: 0.0

            :param drop_path_rate: Drop path rate for timm models. 
                Applied internally within the layers of the Timm Image Model 
                (specifically common in Vision Transformers and similar 
                architectures).	A technique that randomly drops entire 
                residual paths/branches within the network, often more 
                effective than standard dropout in deep models.

                :default: 0.0

            :param train_transforms: A callable that is used to transform 
                the training images. If None is provided, the default train 
                transform is used, that consists of resizing, horizontal and 
                vertical flipping and normalization.

                :default: configilm.extra._defaults default_train_transform

            :param loss_fn: The loss function to use for training.

                :default: F.cross_entropy

            :param class_distribution: Pandas DataFrame containing the 
                absolute class counts and thus the class distribution.

                :default: None

            :param alpha: Dampening factor for the class_weight calculation.
                alpha>1.0 ---> Amplified Inverse Frequency.
                alpha=1.0 ---> Full Inverse Frequency (Unaltered).
                alpha=0.5 ---> Square Root Inverse Frequency (Dampened).
                alpha=0 ---> No Inverse Frequency (regular loss without class weights).

                :default: 0.75

            :param patience: Number of epochs with no improvement after which 
                training may be stopped. Used for EarlyStopping callbacks.

                :default: 6

            :param optimizer: The optimizer algorithm to use for training the 
                model.

                :default: torch.optim.AdamW
        """

        self.generator = generator
        self.seed = seed
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        if not train_transforms:
            mean, std = band_combi_to_mean_std(self.img_size[0])
            self.train_transforms = default_train_transform((self.img_size[1], self.img_size[2]), mean, std)
        else:
            self.train_transforms = train_transforms
        self.loss_fn = loss_fn
        self.patience = patience
        self.optimizer = optimizer
        self.class_distribution = class_distribution
        self.alpha = alpha
        self.class_weights = self._set_class_weights(alpha=self.alpha)

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

    def _set_class_weights(self, alpha: float = 0.75) -> torch.Tensor:
        """Computes inverse class weights from absolute label counts.

            The counts are stored in a parquet file and count the number of 
            patches assigned with the respective class.

            :param alpha: Dampening factor. 
                  alpha=1.0 ---> full Inverse Frequency (Standard).
                  alpha=0.5 ---> Square Root Inverse Frequency (Dampened).
                  alpha=0 ---> no Inverse Frequency (regular loss without class weights).

            :returns torch.Tensor: Tensor of shape (self.num_classes,) with the weight for each class.
        """
        if self.class_distribution is None: return torch.ones(self.num_classes, dtype=torch.float32)

        counts = self.class_distribution['counts'].values
        total_samples = counts.sum()

        # or get eps based on input type instead of using 1e-6
        # If counts is float64 this is ~2e-16, if it's float32 this is ~1e-7
        # eps = np.finfo(counts.dtype).eps 
        inverse_freq = total_samples / (counts + 1e-6)

        dampened_weights = np.power(inverse_freq, alpha)

        # normalize weights to mean = 1.0 so global lr does not have to be scaled
        normalized_weights = dampened_weights * (self.num_classes / dampened_weights.sum())

        return torch.tensor(normalized_weights, dtype=torch.float32) 
    
    def _trunc_norm(self, mean, std, low, upp):
        """Returns a float form a truncated normal distribution."""
        return truncnorm(
                (low - mean) / std, 
                (upp - mean) / std, 
                loc=mean, 
                scale=std
        ).rvs()

    def _trunc_norm_int(self, mean, std, low, upp):
        """Returns an int from a truncated normal distribution."""
        return int(round(self._trunc_norm(mean=mean, std=std, low=low, upp=upp)))

    def randomize_all(self, seed: Optional[int] = None, chance: float = 0.5) -> None:
        """Runs all randomization methods.

            :param seed: The random seed to use for all operations.

            :param chance: The chance a parameter gets randomized.
        """
        if seed is not None: self._set_seed(seed)
        print("\nRandomizing all hyperparameters...")

        if not 0 <= chance <= 1:
            print("Chance invalid, needs to be float between 0 and 1.")
            print("Setting chance to default 0.5.")
            chance = 0.5

        tasks = { 
             'batch_size': self.randomize_batch_size,
             'drop_rate': self.randomize_drop_rate,
             'drop_path_rate': self.randomize_drop_path_rate,
             'train_transforms': self.randomize_train_transforms,
             'loss_fn': self.randomize_loss_fn,
             'class_weights': self.randomize_class_weights,
             'patience': self.randomize_patience,
             'optimizer_and_lr': self.randomize_optimizer_and_lr,
        }

        for param, func in tasks.items():
            random_value = np.random.uniform(0,1)
            if random_value <= chance and chance != 0.0:
                print(f'---- Randomizing {param} ({random_value:.2f} <= {chance:.2f})')
                func()
            elif random_value != 0.0:
                print(f'---- Not randomizing {param} ({random_value:.2f} > {chance:.2f})')
            else:
                print(f'---- Not randomizing {param} ({random_value:.2f} = {chance:.2f})')

    def randomize_lr(self, seed: Optional[int] = None) -> None:
        """Randomizes the learning rate on a logarithmic scale."""
        if seed is not None: self._set_seed(seed)
        # Log-uniform distribution is common for learning rate searches, commonly 10^-5 <= lr <= 10^-1
        self.lr = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-1))))

    def randomize_batch_size(self, seed: Optional[int] = None) -> None:
        """Randomizes the batch size from a set of common values (powers of 2)."""
        if seed is not None: self._set_seed(seed)
        choices = list({32, 64, 128, 256, 512, 1024} - {self.batch_size})
        self.batch_size = random.choice(choices)

    def randomize_drop_rate(self, seed: Optional[int] = None) -> None:
        """Randomizes the timm model's drop rate."""
        if seed is not None: self._set_seed(seed)
        # often 0, 0.1 or 0.2
        self.drop_rate = np.random.uniform(0.0, 0.5)

    def randomize_drop_path_rate(self, seed: Optional[int] = None) -> None:
        """Randomizes the timm model's drop path rate."""
        if seed is not None: self._set_seed(seed)
        # drop_path_rate often benefits from being 0 or a low value
        self.drop_path_rate = np.random.uniform(0.0, 0.2)

    def randomize_train_transforms(self, seed: Optional[int] = None) -> None:
        """Randomizes the image augmentation pipeline for training.
            A set of potential augmentations is defined, and a random subset is
            chosen and composed.
        """
        if seed is not None: self._set_seed(seed)
        
        ben_mean, ben_std = band_combi_to_mean_std(self.img_size[0])
        ben_mean_list = ben_mean.tolist()

        potential_transforms = [
            MyGaussianNoise(sigma=120.0),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            MyRotateTransform([0, 90, 180, 270]),
            v2.RandomRotation(degrees=180,fill=ben_mean_list),
            v2.RandomResizedCrop(size=(self.img_size[1],self.img_size[2]),scale=(0.5,1.0),antialias=True),
            v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=random.choice([ben_mean_list,'random'])),
            #v2.ElasticTransform(alpha=50.0, sigma=5.0, fill=ben_mean_list),
            v2.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.75,1.5), fill=ben_mean_list),
            v2.RandomPerspective(distortion_scale=0.4, p=0.5, fill=ben_mean_list),
            v2.RandomCrop((self.img_size[1],self.img_size[2]), padding=12, fill=ben_mean_list),
            v2.GaussianBlur(kernel_size=(9,9),sigma=(0.1,1.5)), 
            #v2.RandomChannelPermutation(),
        ]

        # Randomly select a number of transforms to apply (e.g., 2 to 4)
        num_transforms_to_apply = self._trunc_norm_int(mean=4, std=2, low=0, upp=len(potential_transforms))
        selected_transforms = random.sample(potential_transforms, num_transforms_to_apply)

        # Basic transforms that are almost always needed
        final_transforms = [
            v2.Resize((self.img_size[1], self.img_size[2]), antialias=True),
            *selected_transforms,
            v2.Normalize(mean=ben_mean, std=ben_std),
        ]

        self.train_transforms = v2.Compose(final_transforms)

    def randomize_loss_fn(self, seed: Optional[int] = None) -> None:
        """Randomizes the loss function from a predefined list."""
        if seed is not None: self._set_seed(seed)

        # Predefine cross_entropy with a random label smoothing value sampled once and remaining constant for the entire training run
        random_smoothing_factor = np.random.uniform(0.0, 0.2)
        ce_smooth_random = partial(F.cross_entropy, label_smoothing=random_smoothing_factor)

        loss_functions = {
            F.cross_entropy,
            ce_smooth_random,
            F.multi_margin_loss,
        }

        choices = list(loss_functions - {self.loss_fn})
        
        self.loss_fn = random.choice(choices)
    
    def randomize_class_weights(self, seed: Optional[int] = None) -> None:
        """Randomizes the class weights for the loss function.
            This can either set weights to None (no weighting) or create a tensor
            of random weights.
        """
        if seed is not None: self._set_seed(seed)

        technique = random.choice(['alpha', 'per-class'])

        if technique == 'alpha':
            # Generate a random dampening factor
            print(f'-------- Regenerating class_weights with random dampening factor alpha')
            self.alpha = self._trunc_norm(mean=0.75, std=1.0, low=0.0, upp=3.0)
            self.class_weights = self._set_class_weights(alpha=self.alpha)
        if technique == 'per-class':
            # Generate a random weight scalar, e.g., between 0.5 and 2.0 for each class
            print(f'-------- Rescaling class_weights with random per-class scalars')
            scalars = []
            for _ in range(self.num_classes):
                # only randomize in some cases
                if np.random.random() > 0.3:
                    val = 1.0
                else:
                    # but if randomized, should be noticeable impact
                    if np.random.random() < 0.5:
                        val = np.random.uniform(0.3, 0.6)
                    else:
                        val = np.random.uniform(1.5, 3.0)
                scalars.append(val)
            scalars = torch.FloatTensor(scalars)
            scaled_weights = self.class_weights * scalars
            normalized_weights = scaled_weights * (self.num_classes / scaled_weights.sum())
            self.class_weights = normalized_weights
        
    def randomize_optimizer(self, seed: Optional[int] = None) -> None:
        """Randomizes the optimizer."""
        if seed is not None: self._set_seed(seed)
        optimizers = [
            torch.optim.AdamW,
            torch.optim.Adam,
            torch.optim.NAdam,
            torch.optim.RAdam,
            torch.optim.Adamax,
            torch.optim.SGD,
            torch.optim.ASGD,
            torch.optim.RMSprop,
        ]
        self.optimizer = random.choice(optimizers)

    def randomize_optimizer_and_lr(self, seed: Optional[int] = None) -> None:
        """Randomizes the optimizer and learning rate based on optimizer groups."""
        if seed is not None: self._set_seed(seed)

        optimizer_groups = {
            'Adam': (
                [torch.optim.AdamW, torch.optim.Adam, torch.optim.NAdam, 
                 torch.optim.RAdam, torch.optim.Adamax],
                (1e-5, 5e-3)
            ),
            'RMSprop': (
                [torch.optim.RMSprop],
                (1e-5, 3e-2)
            ),
            'SGD': (
                [torch.optim.SGD, torch.optim.ASGD],
                (1e-3, 2e-1)
            ),
        }

        # choose a group of optimizer
        chosen_group = random.choice(list(optimizer_groups.keys()))

        # get the available configurations
        optimizers, (lr_min, lr_max) = optimizer_groups[chosen_group]

        # in case we chose the group of self.optimizer and it contains only self.optimizer
        if self.optimizer in optimizers and len(optimizers) == 1:
            chosen_group = random.choice(list(optimizer_groups.keys() - {chosen_group}))

            optimizers, (lr_min, lr_max) = optimizer_groups[chosen_group]

        # exclude the default optimizer to ensure a change
        choices = [o for o in optimizers if o is not self.optimizer]
        
        # choose out of valid choices only
        self.optimizer = random.choice(choices)

        # randomize lr adjusted to optimizer family
        self.lr = float(np.exp(np.random.uniform(np.log(lr_min), np.log(lr_max))))

    def randomize_patience(self, seed: Optional[int] = None) -> None:
        """Randomizes the early stopping patience."""
        if seed is not None: self._set_seed(seed)
        choices = list(set(range(3,11)) - {self.patience})
        self.patience = random.choice(choices)

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
                if isinstance(val, pd.DataFrame):
                    return val.to_string()

                s = ""
                if hasattr(val, '__name__'):
                    if val.__name__ == "cross_entropy":
                        sig = signature(val)
                        s = val.__name__ + "(" + str(sig.parameters['label_smoothing']) + ")"
                    else: 
                        s = val.__name__
                elif isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], str):
                    s = val[0] # For tuples like ("mul", torch.mul)
                elif isinstance(val, torch.Tensor):
                    #return f"Tensor(shape={val.shape})"
                    s = str(val)
                elif val is None:
                    s = "None"
                elif isinstance(val, float):
                    s = f"{val:.6f}"
                else:
                    s = str(val)
                
                wrapped_lines = [textwrap.fill(line, width=30, break_long_words=False, replace_whitespace=False) for line in s.splitlines()]
                
                join_char = "\n\n" if "Compose" in s else "\n"

                return join_char.join(wrapped_lines)

            standard_value_str = format_value(standard_value)
            current_value_str = format_value(current_value)

            if standard_value_str == current_value_str:
                new_value_display = "Unchanged"
            else:
                new_value_display = current_value_str

            standard_value_str = standard_value_str.splitlines()
            new_value_display = new_value_display.splitlines()
            len_std = len(standard_value_str)
            len_curr = len(new_value_display)
            table_data.append([param, standard_value_str[0], new_value_display[0]])
            if len_std == len_curr:
                for i in range(1,len_std):
                    table_data.append(["", standard_value_str[i], new_value_display[i]])
            elif len_std > len_curr:
                for i in range(1,len_std):
                    if i < len_curr:
                        table_data.append(["", standard_value_str[i], new_value_display[i]])
                    else: 
                        table_data.append(["", standard_value_str[i], ""])
            else:
                for i in range(1,len_curr):
                    if i < len_std:
                        table_data.append(["", standard_value_str[i], new_value_display[i]])
                    else: 
                        table_data.append(["", "", new_value_display[i]])

        # --- Create and print the table using pandas ---
        df = pd.DataFrame(table_data, columns=["Parameter", "Standard Value", "New Value"])

        #with pd.option_context('display.max_rows', None, 'display.width', 1000):
            #print(df)

        print(df.to_string())
        print("-" * 86)
        print("\n")

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Create a config and show its initial state
    print("--- 1. Initial Configuration State ---")
    config = TrainLightningConfiguration(img_size=(10, 120, 120))
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
