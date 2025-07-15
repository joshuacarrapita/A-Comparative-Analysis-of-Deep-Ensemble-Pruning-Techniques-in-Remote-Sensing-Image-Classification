import sys
import time
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from pathlib import Path

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.DataModules import BENv2_DataModule
from configilm import util

torch.backends.nnpack.enabled = False

# General SETUP =============================================================

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
        'export': Path(sys.argv[4]).resolve() / time.strftime('%Y%m%d-%H%M%S'),
    }
    for key, path in datapath.items():
        if key == 'images_lmdb':
            assert path.is_dir(), f"Input directory does not exist or is not a directory: {path}"
        else: 
            if key != 'export': 
                assert path.is_file(), f"Input file does not exist or is not a file: {path}"
            else:
                assert not path.is_file(), f"Export needs to be a directory but is a file: {path}" 
    if len(sys.argv) > 5:
        seed = int(sys.argv[6])
        print(f'\n Seed set to {seed}.')
    else:
        seed = 42
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python train.py /path/to/lmdb_dir/ /path/to/metadata.parquet /path/to/metadata_snow_cloud.parquet /path/to/export/dir/ --seed [Number]\n')

datapath['export'].mkdir(parents=True, exist_ok=True)

# LightningModule SETUP =====================================================

class LitImageEncoder(pl.LightningModule):
    """
    Wrapper around a pytorch module, allowing this module to be used in automatic
    training with Pytorch Lightning.
    Among other things, the wrapper allows us to do automatic training and removes the
    need to manage data on different devices (e.g. GPU and CPU).
    """
    def __init__(
        self,
        config: ConfigILM.ILMConfiguration,
        lr: float = 1e-3,
        loss_fn: callable = F.cross_entropy,
    ):
        super().__init__()
        self.lr = lr
        self.config = config
        self.loss_fn = loss_fn
        self.model = ConfigILM.ConfigILM(config)
        self.val_output_list = []
        self.test_output_list = []
        self.save_hyperparameters() # TODO: ignore=['loss_fn', 'config']? Do I really have to include this saving at all or does the logger do it automatically?

    def configure_optimizers(self):
        # TODO: Shouldn't this be self.model.parameters()?
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        img, label = batch

        # convert one-hot encoded label to class index (batch_size x classes label tensor to batch_size x 1 label tensor
        label = torch.argmax(label, dim=1)

        output = self.model(img)
        loss = self.loss_fn(output, label)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        img, label = batch

        # convert one-hot encoded label to class index (batch_size x classes label tensor to batch_size x 1 label tensor
        label = torch.argmax(label, dim=1)

        output = self.model(img)
        loss = self.loss_fn(output,label)

        self.val_output_list.append({"loss": loss})
        # self.val_output_list += [{"loss": loss, "outputs": output, "labels": label}] # in case I need the specific outputs and labels
        # TODO: self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)?

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.val_output_list]).mean()
        self.log("val/loss", avg_loss, prog_bar=True, on_epoch=True, sync_dist=True) # sync_dist (bool) – if True, reduces the metric across devices. Use with care as this may lead to a significant communication overhead.
        
    def test_step(self, batch, batch_idx):
        img, label = batch

        # convert one-hot encoded label to class index (batch_size x classes label tensor to batch_size x 1 label tensor)
        label = torch.argmax(label, dim=1)

        output = self.model(img)
        loss = self.loss_fn(output,label)
        self.test_output_list.append({"loss": loss})
        # self.test_output_list += [{"loss": loss, "outputs": output, "labels": label}] # in case I need the specific outputs and labels
        
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_output_list = []

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.test_output_list]).mean()
        self.log("test/loss", avg_loss, prog_bar=True, sync_dist=True)

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)

# MODEL SETUP ===============================================================

# define model params
model_name = "resnet18"
number_of_channels = 3
image_size = 120
lr = 5e-4
epochs = 10
batch_size = 32
patience = 6
num_workers = 12 # experiment with this value to find most efficient cpu utilization

# seed for reproducibility
pl.seed_everything(seed, workers=True)

# setup model config
model_config = ILMConfiguration(
        timm_model_name=model_name,
        classes=19,
        image_size=image_size,
        channels=number_of_channels,
        network_type=ILMType.IMAGE_CLASSIFICATION,
)

# generate model
model = LitImageEncoder(config=model_config, lr=lr)

# generate lighting DataModule
dm = BENv2_DataModule.BENv2DataModule(
        data_dirs = datapath,
        img_size = (number_of_channels, image_size, image_size),
        batch_size = batch_size, # --> default 16 | TODO: Figure out good batch_size
        # shuffle = None, --> default is None
        # num_workers_dataloader = num_workers, --> default is os.cpu_count() / 2 which I think is good
)

dm.prepare_data()
dm.setup() # stage=None --> All datasets are loaded | TODO: maybe not necessary as trainer handles this?

# instantiate trainer
trainer = pl.Trainer(
        max_epochs = epochs,
        accelerator = 'auto',
        devices = 'auto',
        log_every_n_steps = 10,
        logger = True,
        precision = '16-mixed', # for A100 GPUs (Automatic Mixed Precision) TODO: /home/user/.pyenv/versions/ba_python_venv/lib/python3.11/site-packages/lightning/fabric/connector.py:571: `precision=16` is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
        callbacks=[
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=patience, mode="min"), 
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1), # saves best model
        ],
        default_root_dir=datapath['export'],
)

# TRAINING ==================================================================

trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm) # automatically loads best model from checkpoint

print('\nTraining complete.')
print(f'Best model and logs saved in: {datapath["export"]}')

# SAVING ====================================================================

# model_outpath = export_dir / ('resnet18_' + time_str + '.ckpt')
# trainer.save_checkpoint(model_outpath)
