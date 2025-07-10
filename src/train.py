import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra import BENv2_utils
from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

from pathlib import Path

from configilm import util

from alive_progress import alive_bar

if __debug__:
    print ("\n--DEBUG ON--")
    util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages
else:
    print ("\n--DEBUG OFF--")

try:
    datapath = {
        'images_lmdb': Path(sys.argv[1]).resolve(),
        'metadata_parquet': Path(sys.argv[2]).resolve(),
        'metadata_snow_cloud_parquet': Path(sys.argv[3]).resolve(),
        'export': Path(sys.argv[4]).resolve()
    }
    for key, path in datapath.items():
        if key == 'images_lmdb':
            assert path.is_dir(), f"Input directory does not exist or is not a directory: {path}"
        else: 
            if key != 'export': 
                assert path.is_file(), f"Input file does not exist or is not a file: {path}"
            else:
                assert not path.is_file(), f"Export needs to be a directory but is a file: {path}" 
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python train.py /path/to/lmdb_dir/ /path/to/metadata.parquet /path/to/metadata_snow_cloud.parquet /path/to/export/dir/\n')

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

ben_mean, ben_std = BENv2_utils.band_combi_to_mean_std(number_of_channels)
train_transform = default_train_transform(img_size=(image_size,image_size), mean=ben_mean, std=ben_std)
transform = default_transform(img_size=(image_size,image_size), mean=ben_mean, std=ben_std)

train_ds = BENv2DataSet(
    datapath,
    split="train",
    transform=train_transform,
    img_size=(number_of_channels, image_size, image_size),
)

val_ds = BENv2DataSet(
    datapath,
    split="validation",
    transform=transform,
    img_size=(number_of_channels, image_size, image_size),
)

test_ds = BENv2DataSet(
    datapath,
    split="test",
    transform=transform,
    img_size=(number_of_channels, image_size, image_size),
)

train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
)

val_dl = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)

test_dl = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)

if __debug__:
    print(f"Train dataset size: {len(train_dl)}")
    print(f"Validation dataset size: {len(val_dl)}")
    print(f"Test dataset size: {len(test_dl)}")

optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
loss_fn = F.cross_entropy

for epoch in range(10):
    running_loss = 0.0
    # --- train loop ---
    model.train()
    print(f'Starting epoch {epoch + 1:3d} with {len(train_dl):5d} batches:')
    with alive_bar (len(train_dl)) as bar:
        for i, batch in enumerate(train_dl):
            bar()
            img, label = batch

            # convert batch_size x classes label tensor to 
            # batch_size x 1 label tensor
            label = torch.argmax(label, dim=1)

            optim.zero_grad()
            output = model(img)
            loss = loss_fn(output,label)
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            if (i % 10) == 9: # print every 10 mini-batches
                print(f"[{epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 10:.4f}")
                if __debug__:
                    print(f'[train] labelsize: {label.size()}, label: {label}')
                    print(f'[train] outputsize: {output.size()}, output: {torch.argmax(output,dim=1)}')
                running_loss = 0.0

    # validation
    model.eval()
    eval_loss = 0.0
    for i, batch in enumerate(val_dl):
        img, label = batch

        # convert batch_size x classes label tensor to 
        # batch_size x 1 label tensor
        label = torch.argmax(label, dim=1)

        output = model(img)
        if __debug__:
            print(f'[val] labelsize: {label.size()}, label: {label}')
            print(f'[val] outputsize: {output.size()}, output: {torch.argmax(output,dim=1)}')
        loss = loss_fn(output, label)
        eval_loss += loss.item()
    eval_loss /= len(val_dl)
    print(f"Epoch {epoch + 1:3d} validation loss: {eval_loss:.4f}")

# test
model.eval()
test_loss = 0.0
for i, batch in enumerate(test_dl):
    img, label = batch

    # convert batch_size x classes label tensor to 
    # batch_size x 1 label tensor
    label = torch.argmax(label, dim=1)

    output = model(img)
    if __debug__:
        print(f'[test] labelsize: {label.size()}, label: {label}')
        print(f'[test] outputsize: {output.size()}, output: {torch.argmax(output,dim=1)}')
    loss = loss_fn(output, label)
    test_loss += loss.item()
test_loss /= len(test_dl)
print(f"Test loss: {test_loss:.4f}")

# saving
time = time.strftime('%Y%m%d-%H%M%S')
datapath['export'] /= time
datapath['export'].mkdir(parents=True, exist_ok=True)
model_outpath = datapath['export'] / ('model_resnet18_' + time + '.pt')
optim_outpath = datapath['export'] / ('optimizer_resnet18_' + time + '.pt')
torch.save(model.state_dict(),model_outpath)
torch.save(optim.state_dict(),optim_outpath)
