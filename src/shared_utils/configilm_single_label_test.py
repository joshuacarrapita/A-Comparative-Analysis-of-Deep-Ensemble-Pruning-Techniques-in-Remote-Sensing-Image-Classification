import torch
from torchvision.transforms import ToPILImage
from pathlib import Path
import sys
from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule

#-----------------------------------------------------------------------------
# INIT 
try:
    datapath = {
        "images_lmdb": Path(sys.argv[1]).resolve(),
        "metadata_parquet": Path(sys.argv[2]).resolve(),
        "metadata_snow_cloud_parquet": Path(sys.argv[3]).resolve(),
    }
    for key, path in datapath.items():
        if key == "images_lmdb":
            assert path.is_dir(), f"Input directory does not exist or is not a directory: {path}"
        else:
            assert path.is_file(), f"Input file not found or not a file: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python configilm_single_label_test.py /path/to/images.lmdb /path/to/metadata.parquet /path/to/metadata_snow.parquet\n')

#-----------------------------------------------------------------------------
# EXEC 
# load BEN v2 Dataset with desired settings
ds = BENv2_DataSet.BENv2DataSet(
    data_dirs=datapath, # path set by to dataset
    max_len=1000, # max no. of images to use
    img_size = (3,120,120), # specify channels
)

# access data at idx 0 of dataset
img, lbl = ds[0]

# convert tensor to rgb format
img_clipped = torch.clamp(img, 0, 3000) # remove outliers
img_scaled = (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min()) * 255.0 # normalize between [0,1]
img_rgb = img_scaled.byte() # convert to bytes (0-255)

# print image and info
print(img.shape, lbl)
to_pil = ToPILImage()
img_pil = to_pil(img_rgb)
img_pil.show()
