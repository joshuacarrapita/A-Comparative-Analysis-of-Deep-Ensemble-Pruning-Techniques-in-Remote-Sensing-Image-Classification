# stdlib imports
import os
import sys
import time
from pathlib import Path
from typing import Optional

# third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassConfusionMatrix
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor

# local imports
from configilm import ConfigILM, util
from configilm.ConfigILM import ILMConfiguration, ILMType
from configilm.extra.DataModules import BENv2_DataModule
from train_lightning_configuration import TrainLightningConfiguration

# ===========================================================================
# LightningModule Class Definition
# ===========================================================================
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
        lr: float = 5e-4,
        loss_fn: callable = F.cross_entropy,
        class_weights: Optional[torch.Tensor] = None,
        optimizer: callable = torch.optim.AdamW,
    ):
        super().__init__()

        # hyperparameters 
        self.lr = lr
        self.config = config
        self.model = ConfigILM.ConfigILM(config)
        self.val_output_list = []
        self.test_output_list = []

        self.loss_fn = loss_fn
        self.class_weights = class_weights
        self.optimizer = optimizer
        self.save_hyperparameters() 

        # metrics
        self.train_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)
        self.val_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=0.01) # DONE add passing of optimizer in init
        return optimizer
    
    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # retrieve batch data
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate loss
        logits = self.model(img)
        loss = self.loss_fn(logits,label,weight=self.class_weights)

        # calculate batch accuracy
        preds = torch.argmax(logits,dim=1) # logits -> index
        acc = 100 * (preds == label).float().mean()

        # update training confusion matrix
        self.train_cm.update(preds,label)
        
        # log metrics to logger
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        # return loss for optimizer to work with
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # retrieve batch data
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate loss
        logits = self.model(img)
        loss = self.loss_fn(logits,label,weight=self.class_weights)

        # calculate batch accuracy
        preds = torch.argmax(logits,dim=1) # logits -> index
        acc = 100 * (preds == label).float().mean()

        # update validation confusion matrix
        self.val_cm.update(preds,label) 

        # log metrics to logger
        self.log('val_acc', acc, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True) 
       
    def test_step(self, batch, batch_idx):
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate loss
        logits = self.model(img)
        loss = self.loss_fn(logits,label,weight=self.class_weights)

        # calculate batch accuracy
        preds = torch.argmax(logits,dim=1) # logits -> index
        acc = 100 * (preds == label).float().mean()

        # log metrics to logger
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_loss', loss, on_epoch=True) 

    def on_train_epoch_end(self):
        cm = self.train_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='train_cm')
        self.train_cm.reset()

    def on_validation_epoch_end(self):
        # calc and log confusion matrix
        cm = self.val_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='val_cm')

        # calc and log per-class accuracy
        per_class_acc = self._calc_per_class_accuracy(cm) # returns (C,)
        self._log_class_accuracy(per_class_acc, tag='val_class_acc')

        # calc and log average per-class accuracy
        avg_per_class_acc = torch.mean(per_class_acc)
        self.log('avg_per_class_acc', avg_per_class_acc, on_epoch=True)

        self.val_cm.reset()

    def _calc_per_class_accuracy(self, cm_tensor):
        """
            cm_tensor: (C, C) int64 confusion matrix
            Returns: (C,) float tensor with TP/(TP+FN) for each class
        """
        # true positives = diagonal, total true instances = row sum
        tp = torch.diag(cm_tensor).float()
        total = cm_tensor.sum(dim=1).float()
        # avoid division‑by‑zero
        return torch.where(total == 0, torch.zeros_like(tp), tp / total)
        
    def _log_confusion_matrix(self, cm_tensor, tag: str):
        """
            :param cm_tensor : (C, C) int64 tensor of counts 
            :param tag       : logger prefix (e.g., "val_cm")
        """
        # convert confusion matrix to numpy
        cm_np = cm_tensor.cpu().numpy()

        # plot
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(
            cm_np,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(f"(Confusion Matrix - {tag.split('_')[0].upper()}")

        # log figure to logger
        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

        # (optional) log raw tensor
        # self.log(f"{tag}_raw", cm_tensor, on_step=False, on_epoch=True)

        plt.close(fig)

    def _log_class_accuracy(self, acc_tensor, tag: str):
        """Log a 1×C accuracy tensor as a seaborn heat‑map."""
        acc_tensor = acc_tensor.cpu().numpy().reshape(1, -1)   # (1, C)

        fig, ax = plt.subplots(figsize=(max(6, self.config.classes * 0.4), 2))
        sns.heatmap(
            acc_tensor,
            annot=True,
            annot_kws={"rotation": 90, "ha": "center", "va": "center"},
            fmt=".2%",
            cmap="viridis",
            cbar=False,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xticks(np.arange(self.config.classes) + 0.5)
        ax.set_xticklabels([f"C{i}" for i in range(self.config.classes)],
                           rotation=90, ha="right")
        ax.set_yticks([])  # hide the single y‑tick
        ax.set_title("Per‑class Accuracy (validation)")
        plt.tight_layout()
        self.logger.experiment.add_figure(tag, fig, self.current_epoch)
        plt.close(fig)
 

# ===========================================================================
# Helper Functions
# ===========================================================================

def randomize_tc_for_benv2_ep(tc: TrainLightningConfiguration, seed: Optional[int] = None) -> None:
    """
        Runs all randomization methods for creation of a BENv2 image classifying ensemble.

        :param tc: The traning configuration to randomize.

        :param seed: The random seed to use for all operations.
    """
    tc._set_seed(seed)
    print("\nRandomizing all hyperparameters...")
    tc.randomize_lr()
    tc.randomize_batch_size()
    tc.randomize_drop_rate()
    tc.randomize_drop_path_rate()
    tc.randomize_train_transforms()
    tc.randomize_loss_fn()
    tc.randomize_class_weights()
    tc.randomize_optimizer()
    tc.randomize_patience()

def inverse_class_weights_from_occurrences(class_distribution_parquet: Path, num_classes: int) -> torch.Tensor:
    """
        Compute inverse class weights from a parquet file that stores label counts.

        :param class_distribution_parquet: Path to file containing the absolute class counts thus the class distribution.

        :returns torch.Tensor: Tensor of shape (num_classes,) with the weight for each class.
    """

    label_counts = pd.read_parquet(class_distribution_parquet)
    total_samples = label_counts['counts'].sum()
    class_weights = total_samples / (num_classes * label_counts['counts'])
    print(class_weights)
    return torch.tensor(class_weights.values, dtype=torch.float32) # TODO: check if float32 is good for the loss functions

# ===========================================================================
# Main Training Script
# ===========================================================================
def main():

    # --- General Setup ---
    start_time = time.strftime('%Y%m%d-%H%M%S')

    # Set Verbosity Level
    if __debug__:
        print ('\n--DEBUG ON--')
        util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages
    else:
        print ('\n--DEBUG OFF--')

    try:
        # Parse User Input
        datapath_args = {
            'images_lmdb': Path(sys.argv[1]).resolve(),
            'metadata_parquet': Path(sys.argv[2]).resolve(),
            'metadata_snow_cloud_parquet': Path(sys.argv[3]).resolve(),
            'class_distribution_parquet': Path(sys.argv[4]).resolve(),
        }
        export_dir_root = Path(sys.argv[5]).resolve()
        seed = int(sys.argv[7]) if len(sys.argv) > 6 else 42
        print(f'\nSeed set to {seed}.')

        # Validate Paths
        assert datapath_args['images_lmdb'].is_dir(), f"Input directory does not exist or is not a directory: {datapath_args['images_lmdb']}"
        assert datapath_args['metadata_parquet'].is_file(), f"Input file does not exist or is not a file: {datapath_args['metadata_parquet']}"
        assert datapath_args['metadata_snow_cloud_parquet'].is_file(), f"Input file does not exist or is not a file: {datapath_args['metadata_snow_cloud_parquet']}"
        assert datapath_args['class_distribution_parquet'].is_file(), f"Input file does not exist or is not a file: {datapath_args['class_distribution_parquet']}"
        assert not export_dir_root.is_file(), f"Export needs to be a directory but is a file: {export_dir_root}"

    except Exception as e:
        # Handle Input Errors
        sys.exit(f'\nAn error occured: {e}.\n\nUsage: python train.py /path/to/lmdb_dir/ /path/to/metadata.parquet /path/to/metadata_snow_cloud.parquet /path/to/class_distribution.parquet /path/to/export/dir/ --seed [Number]\n')
    
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id is not None:
        slurm_job_id = int(slurm_job_id)
        export_dir = export_dir_root / f'resnet18_{start_time}_seed{seed}_job{slurm_job_id}'
    else:
        export_dir = export_dir_root / f'resnet18_{start_time}_seed{seed}'
    export_dir.mkdir(parents=True, exist_ok=True)

    # --- Model, Dataset, Trainer Setup ---

    # Define Training Parameters
    # TODO: pass class weights based on inverse frequency
    class_weights = inverse_class_weights_from_occurrences(datapath_args['class_distribution_parquet'], num_classes=19)

    tc = TrainLightningConfiguration(class_weights=class_weights)
    print("\nConfiguration before randomization:")
    tc.info()
    randomize_tc_for_benv2_ep(tc=tc, seed=seed)
    print("\nConfiguration after randomization:")
    tc.info()

    # Set PL Seed For Training Reproducibility
    pl.seed_everything(seed, workers=True) # Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random 

    # Define Model Parameters
    model_name = 'resnet18'
    """ about number_of_channels
        ba-dataset #Frage-BA @{2025-05-27} ask ronny which channels I want to use in my thesis and also whether it should be multi-modal or separated by models etc.
        ---> the two SAR sentinel 1 band (2) and the 10 optical sentinel 2 bands (10) (combined would be 12)
        ---> they should be separated so that I have one ensemble for sar and one ensemble for optical input
    """
    number_of_channels = 10
    image_size = 120
    epochs = 100
    slurm_cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus_per_task is not None:
        num_workers = int(slurm_cpus_per_task)-2 # -2 to keep some cores for main and other processes, experiment with this value to find most efficient cpu utilization
        
        print("Num_workers set with SLURM_CPUS_PER_TASK")
    else:
        num_workers = max(1,min(os.cpu_count()/4-2,10)) # /2 for two sockets and /2 cause os.cpu_count returns thread as multithreads (2 per core) and -2 to keep some for the main process
        print("Num_workers set with max(1,min(os.cpu_count()/4-2,10))")

    omp_num_threads = os.getenv('OMP_NUM_THREADS')
    mkl_num_threads = os.getenv('MKL_NUM_THREADS')
    print(f"OMP_NUM_THREADS: {omp_num_threads}")
    print(f"MKL_NUM_THREADS: {mkl_num_threads}")
    cuda_launch_blocking = os.getenv("CUDA_LAUNCH_BLOCKING")
    print(f"\nCUDA_LAUNCH_BLOCKING: {cuda_launch_blocking}")

    # Instantiate Model Config
    model_config = ILMConfiguration(
            timm_model_name=model_name,
            classes=tc.num_classes,
            image_size=image_size,
            channels=number_of_channels,
            network_type=ILMType.IMAGE_CLASSIFICATION,
            drop_rate=tc.drop_rate,
            drop_path_rate=tc.drop_path_rate,
    )

    # Retrieve Model
    model = LitImageEncoder(config=model_config, lr=tc.lr, loss_fn=tc.loss_fn, class_weights=tc.class_weights) # DONE change to loss_fn=loss_fn? and add passing of custom weights)

    # Generate Lighting DataModule
    dm = BENv2_DataModule.BENv2DataModule(
            data_dirs = datapath_args,
            img_size = (number_of_channels, image_size, image_size),
            batch_size = tc.batch_size, # --> default 16 | TODO: Figure out good batch_size
            train_transforms = tc.train_transforms,
            num_workers_dataloader = num_workers, # --> default is os.cpu_count() / 2 which I think is good
            bagging = True,
            seed = seed,
    )

    # Instantiate Logger
    logger = TensorBoardLogger(export_dir)

    # Instantiate Profiler
    profiler = SimpleProfiler(filename="perf_logs", extended=True)

    # Instantiate Trainer
    trainer = pl.Trainer(
            max_epochs = epochs,
            accelerator = 'auto',
            devices = 'auto',
            strategy = 'auto',
            log_every_n_steps = 10,
            logger = logger,
            #profiler = profiler,
            precision = '16-mixed', # for A100 GPUs (Automatic Mixed Precision) TODO: /home/user/.pyenv/versions/ba_python_venv/lib/python3.11/site-packages/lightning/fabric/connector.py:571: `precision=16` is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
            callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=tc.patience, mode='min'), 
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1), # saves best model
            #DeviceStatsMonitor(cpu_stats=True),
            ],
            default_root_dir=export_dir,
            deterministic=True,
    )

    if __debug__:
        WIDTH = 18
        print(f"{'Accelerator':<{WIDTH}}{'Devices':<{WIDTH}}{'Strategy':<{WIDTH}}")
        print(f"-" * (WIDTH * 3))
        print(f"{trainer.accelerator.__class__.__name__:<{WIDTH}}{trainer.num_devices:<{WIDTH}}{trainer.strategy.__class__.__name__:<{WIDTH}}")

    # --- CPU/GPU INFO ---
    if torch.cuda.is_available():
        print(f"CUDA available: TRUE")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Get and print memory information for each GPU
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            cached_memory = torch.cuda.memory_cached(i) / (1024**3)
            
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Allocated memory: {allocated_memory:.2f} GB")
            print(f"  Cached memory: {cached_memory:.2f} GB")
    else:
        print(f"CUDA available: FALSE")
        print("No GPUs found. Training on CPU.")

    print(f"Number of workers dataloader: {dm.num_workers_dataloader}")
    print(f"OS CPU count: {os.cpu_count()}")

    if __debug__:
        export_dir_tmp = export_dir
        export_dir = export_dir / "datasets"
        export_dir.mkdir(parents=True, exist_ok=True)
        dm.setup()
        dm.save_metadata(export_dir)
        print(f'Datasets saved in: {export_dir}')
        export_dir = export_dir_tmp


    # --- Training And Testing ---
    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm) # automatically loads best model from checkpoint

    print('\nTraining complete.')
    print(f'Best model and logs saved in: {export_dir}')

if __name__ == '__main__':
    main()
