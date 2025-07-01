import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra import BEN_lmdb_utils
from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.DataSets.BEN_DataSet import BENDataSet

print('yay')

"""
__author__ = "Leonard Hackel - BIFOLD/RSim TU Berlin"

image_model = "resnet18"
number_of_channels = 3
image_size = 120
num_workers = 4

model_config = ILMConfiguration(
        timm_model_name=image_model,
        classes=19,
        image_size=image_size,
        channels=number_of_channels,
        network_type=ILMType.IMAGE_CLASSIFICATION,
)
model = ConfigILM.ConfigILM(model_config)

ben_mean, ben_std = BEN_lmdb_utils.band_combi_to_mean_std(number_of_channels)
"""
