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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class TrainLightningConfiguration:

    # Class variables (variables that span all classes like an instance counter)

    def __init__(
        self,

        generator: Optional[random.Random] = None,
        seed: int = 42,
        num_classes: int = 19,
        
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

            :param generator: Variable to hold own instance of Random if preferred to use custom generators or generators that don't share state.

                :default: None

            :param seed: Random seed for reproducibility.

                :default: 42
            
            :param num_classes: The number of classes in the dataset.

                :default: 19

            :param lr: The learning rate for the optimizer.

                :default: 5e-4

            :param batch_size: The batch size to use for the dataloaders.

                :default: 32

            :param v_dropout_rate: Dropout rate of the mapping from the timm image model to
                the dimension of the fusion method. Applied to the output of the Timm Image Model's final embedding (the "mapping layer") before fusion. Regularizes the visual representation before it's combined with text.

                :default: 0.25

            :param t_dropout_rate: Dropout rate of the mapping from the huggingface text
                model to the dimension of the fusion method. Applied to the output of the Hugging Face Text Model's final embedding (the "mapping layer") before fusion. Regularizes the text representation before it's combined with vision.

                :default: 0.25

            :param fusion_dropout_rate: Drop rate inside all classification head layers. Applied inside the Classification Head (the final prediction layers) that operates on the combined visual and text features.	Regularizes the final decision-making layers after feature fusion.

                :default: 0.25

            :param drop_rate: Dropout rate for timm models. Applied internally within the layers of the Timm Image Model itself (e.g., standard dropout in the backbone). Standard backbone regularization for the image encoder.

                :default: 0.2

            :param drop_path_rate: Drop path rate for timm models. Applied internally within the layers of the Timm Image Model (specifically common in Vision Transformers and similar architectures).	A technique that randomly drops entire residual paths/branches within the network, often more effective than standard dropout in deep models.

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

        self.generator = generator
        self.seed = seed
        self.num_classes = num_classes
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

        self._initial_params = self.__dict__.copy()
        self._set_seed(seed)
        
    def _set_seed(self, seed, generator: Optional[random.Random] = None) -> None:
        """Sets the random seed for reproducibility."""
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
        self.randomize_dropout_rates()
        self.randomize_fusion_hidden()
        self.randomize_fusion_method()
        self.randomize_fusion_activation()
        self.randomize_optimizer()
        self.randomize_patience()
        self.randomize_shuffle()
        self.randomize_train_transforms()
        self.randomize_loss_fn()
        self.randomize_class_weights()

    def randomize_lr(self, seed: Optional[int] = None) -> None:
        """Randomizes the learning rate on a logarithmic scale."""
        if seed is not None: self._set_seed(seed)
        # Log-uniform distribution is common for learning rate searches, commonly 10^-5 <= lr <= 10^-1
        self.lr = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2))))

    def randomize_batch_size(self, seed: Optional[int] = None) -> None:
        """Randomizes the batch size from a set of common values (powers of 2)."""
        if seed is not None: self._set_seed(seed)
        self.batch_size = random.choice([16, 32, 64, 128, 256, 512, 1024])

    def randomize_v_dropout_rate(self, seed: Optional[int] = None) -> None:
        """Randomizes the vision projection dropout rate."""
        if seed is not None: self._set_seed(seed)
        # often 0.5 because of dense layer
        self.v_dropout_rate = random.uniform(0.1, 0.5)

    def randomize_t_dropout_rate(self, seed: Optional[int] = None) -> None:
        """Randomizes the text projection dropout rate."""
        if seed is not None: self._set_seed(seed)
        # often 0.5 because of dense layer
        self.t_dropout_rate = random.uniform(0.1, 0.5)

    def randomize_fusion_dropout_rate(self, seed: Optional[int] = None) -> None:
        """Randomizes the fusion head dropout rate."""
        if seed is not None: self._set_seed(seed)
        # often 0.1 or 0.2 or 0.5 for dense layers
        self.fusion_dropout_rate = random.uniform(0.1, 0.5)

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

    def randomize_all_dropout_rates(self, seed: Optional[int] = None) -> None:
        """Randomizes all dropout-related rates by calling individual randomizers."""
        if seed is not None: self._set_seed(seed)
        self.randomize_v_dropout_rate()
        self.randomize_t_dropout_rate()
        self.randomize_fusion_dropout_rate()
        self.randomize_drop_rate()
        self.randomize_drop_path_rate()
    
    def randomize_backbone_dropout_rates(self, seed: Optional[int] = None) -> None:
        """Randomizes backbone dropout-related rates by calling individual randomizers."""
        if seed is not None: self._set_seed(seed)
        self.randomize_drop_rate()
        self.randomize_drop_path_rate()

    def randomize_fusion_hidden(self, seed: Optional[int] = None) -> None:
        """Randomizes the number of hidden neurons in the fusion layer."""
        if seed is not None: self._set_seed(seed)
        self.fusion_hidden = random.choice([128, 256, 512, 1024])

    def randomize_fusion_method(self, seed: Optional[int] = None) -> None:
        """Randomizes the fusion method from a list of common techniques."""
        if seed is not None: self._set_seed(seed)
        fusion_methods = [
            ("torch.mul", torch.mul),
            ("torch.add", torch.add),
            # Concatenation would require changing fusion_out, so it's omitted here
        ]
        self.custom_fusion_method = random.choice(fusion_methods)

    def randomize_fusion_activation(self, seed: Optional[int] = None) -> None:
        """Randomizes the activation function in the fusion layer."""
        if seed is not None: self._set_seed(seed)
        activations = [
            ("nn.ReLU()", torch.nn.ReLU()),
            ("nn.Tanh()", torch.nn.Tanh()),
            ("nn.GELU()", torch.nn.GELU()),
            ("nn.SiLU()", torch.nn.SiLU()) # Also known as Swish
        ]
        self.custom_fusion_activation = random.choice(activations)
    
    def randomize_train_transforms(self, seed: Optional[int] = None) -> None:
        """
        Randomizes the image augmentation pipeline for training.
        A set of potential augmentations is defined, and a random subset is
        chosen and composed.
        """
        if seed is not None: self._set_seed(seed)

        # TODO: Add or modify custom transformation functions 
        potential_transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.GaussianBlur(kernel_size=3),
        ]

        # Randomly select a number of transforms to apply (e.g., 2 to 4)
        num_transforms_to_apply = random.randint(2, len(potential_transforms))
        selected_transforms = random.sample(potential_transforms, num_transforms_to_apply)

        # TODO: Adjust image size and normalization values for dataset
        img_size = 224
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
        std = [0.229, 0.224, 0.225]   # ImageNet std

        # Basic transforms that are almost always needed
        final_transforms = [
            T.Resize((img_size, img_size)),
            # Add selected augmentations
            *selected_transforms,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

        self.train_transforms = T.Compose(final_transforms)

    def randomize_loss_fn(self, seed: Optional[int] = None) -> None:
        """
        Randomizes the loss function from a predefined list.
        Note: Ensure the chosen loss function is compatible with the model's output
        and the task (e.g., BCE for multi-label, CrossEntropy for multi-class).
        """
        if seed is not None: self._set_seed(seed)

        # TODO: Implement and add custom loss functions if needed (e.g., FocalLoss)
        # class FocalLoss(nn.Module):
        #     ...
        # focal_loss = FocalLoss()

        loss_functions = [
            F.cross_entropy,
            nn.BCEWithLogitsLoss(),
            # Add other implemented losses here, e.g., focal_loss
        ]
        self.loss_fn = random.choice(loss_functions)
    
    def randomize_class_weights(self, seed: Optional[int] = None) -> None:
        """
        Randomizes the class weights for the loss function.
        This can either set weights to None (no weighting) or create a tensor
        of random weights.
        """
        if seed is not None: self._set_seed(seed)
        use_weights = random.choice([True, False])

        if use_weights:
            # Generate random weights, e.g., between 0.5 and 2.0
            weights = torch.FloatTensor([random.uniform(0.5, 2.0) for _ in range(self.num_classes)])
            self.class_weights = weights
        else:
            self.class_weights = None
        
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

    def randomize_shuffle(self, seed: Optional[int] = None) -> None:
        """Randomizes the data shuffling flag."""
        if seed is not None: self._set_seed(seed)
        self.shuffle = random.choice([True, False])

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
