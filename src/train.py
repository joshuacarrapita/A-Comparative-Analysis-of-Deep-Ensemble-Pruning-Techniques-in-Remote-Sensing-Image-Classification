import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
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

time = time.strftime('%Y%m%d-%H%M%S')
datapath['export'] /= time
datapath['export'].mkdir(parents=True, exist_ok=True)

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

loss_log = []
num_epochs = 10
curr_impatience = 0
max_impatience = 11
best_eval = float('inf')
best_model = model.state_dict()
best_optim = optim.state_dict()

writer = SummaryWriter(log_dir=datapath['export'])

for epoch in range(num_epochs):
    if curr_impatience >= max_impatience:
        break
    running_loss = 0.0
    loss_train_epoch = 0.0
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
            loss_train_epoch += loss.item()
            writer.add_scalar(f"loss_train_step_epoch_{epoch+1}", loss.item(), i+1)
            # loss_log.append([epoch+1, i+1, loss.item(), 'train'])

            if (i % 10) == 9: # print every 10 mini-batches
                if __debug__:
                    print(f"[{epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 10:.4f}")
                    print(f'[train] labelsize: {label.size()}, label: {label}')
                    print(f'[train] outputsize: {output.size()}, output: {torch.argmax(output,dim=1)}')
                running_loss = 0.0
    loss_train_epoch /= len(train_dl)
    writer.add_scalar("loss_train_epoch", loss_train_epoch, epoch+1)

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
        writer.add_scalar(f"loss_eval_step_epoch_{epoch+1}", loss.item(), i+1)
        eval_loss += loss.item()
    eval_loss /= len(val_dl)
    writer.add_scalar(f"loss_eval_epoch", eval_loss, epoch+1)
    if eval_loss < best_eval:
        best_eval = eval_loss
        best_model = model.state_dict()
        best_optim = optim.state_dict()
        curr_impatience = 0
        print(f"Epoch {epoch + 1:3d} validation loss: {eval_loss:.4f} (new best)")
    else:
        curr_impatience += 1
        print(f"Epoch {epoch + 1:3d} validation loss: {eval_loss:.4f} (stagnant)")

    # loss_log.append([epoch+1, 0, eval_loss, 'validation'])

# test
model.load_state_dict(best_model)
model.eval()
test_loss = 0.0

idx = 0
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
    
    writer.add_scalar(f"loss_test_step_epoch_{epoch+1}", loss.item(), idx+1)
    idx += 1
    test_loss += loss.item()
test_loss /= len(test_dl)
writer.add_scalar("loss_test_epoch", test_loss, epoch+1)
# loss_log.append([0, 0, test_loss, 'test'])
print(f"Test loss: {test_loss:.4f}")

# saving
writer.flush()
writer.close()
model_outpath = datapath['export'] / ('model_resnet18_' + time + '.pt')
optim_outpath = datapath['export'] / ('optimizer_resnet18_' + time + '.pt')
# loss_log_outpath = datapath['export'] / ('loss_log' + time + '.parquet')
# df = pd.DataFrame(loss_log, columns=['Epoch', 'Batch', 'Loss', 'Type'])
# df.to_parquet(loss_log_outpath, compression='gzip')
torch.save(best_model,model_outpath)
torch.save(best_optim,optim_outpath)
