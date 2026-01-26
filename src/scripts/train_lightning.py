# stdlib imports
import os
import sys
import time
import random
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
from configilm.extra.BENv2_utils import NEW_LABELS_16, NEW_LABELS_16_ORIGINAL_ORDER
from training_utils.train_lightning_configuration import TrainLightningConfiguration
from training_utils.run_tracker import RunTracker

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

        self.loss_fn = loss_fn
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
            
        self.optimizer = optimizer
        self.save_hyperparameters() 

        # metrics
        self.train_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)
        self.val_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)
        self.test_cm = MulticlassConfusionMatrix(num_classes=self.config.classes)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=0.01) 
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
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True) 
       
    def predict_step(self, batch, batch_idx):
        # retrieve batch data
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate probas
        logits = self.model(img)
        probas = torch.nn.functional.softmax(logits, dim=1)

        return {'prediction_results': probas, 'target_labels': label}
        
    def test_step(self, batch, batch_idx):
        img, label = batch
        label = torch.argmax(label, dim=1) # one-hot -> index

        # infer input and calculate loss
        logits = self.model(img)
        loss = self.loss_fn(logits,label,weight=self.class_weights)

        # calculate batch accuracy
        preds = torch.argmax(logits,dim=1) # logits -> index
        acc = 100 * (preds == label).float().mean()

        # update test confusion matrix
        self.test_cm.update(preds,label)

        # log metrics to logger
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True) 

    def on_train_epoch_end(self):
        cm = self.train_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='train_cm')

        # calc and log per-class accuracy
        per_class_acc = self._calc_per_class_accuracy(cm) # returns (C,)
        self._log_class_accuracy(per_class_acc, tag='train_per_class_acc')

        # calc and log average per-class accuracy
        avg_class_acc = torch.mean(per_class_acc)
        self.log('train_avg_class_acc', avg_class_acc, on_step=False, on_epoch=True)

        self.train_cm.reset()

    def on_validation_epoch_end(self):
        # calc and log confusion matrix
        cm = self.val_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='val_cm')

        # calc and log per-class accuracy
        per_class_acc = self._calc_per_class_accuracy(cm) # returns (C,)
        self._log_class_accuracy(per_class_acc, tag='val_per_class_acc')

        # calc and log average per-class accuracy
        avg_class_acc = torch.mean(per_class_acc)
        self.log('val_avg_class_acc', avg_class_acc, on_step=False, on_epoch=True)

        self.val_cm.reset()

    def on_test_epoch_end(self):
        # calc and log confusion matrix
        cm = self.test_cm.compute().detach().cpu()
        self._log_confusion_matrix(cm, tag='test_cm')

        # calc and log per-class accuracy
        per_class_acc = self._calc_per_class_accuracy(cm) # returns (C,)
        self._log_class_accuracy(per_class_acc, tag='test_per_class_acc')

        # calc and log average per-class accuracy
        avg_class_acc = torch.mean(per_class_acc)
        self.log('test_avg_class_acc', avg_class_acc, on_step=False, on_epoch=True)

        self.test_cm.reset()

    def _calc_per_class_accuracy(self, cm_tensor):
        """
            cm_tensor: (C, C) detached int64 confusion matrix
            Returns: (C,) float tensor with TP/(TP+FN) for each class
        """
        # true positives = diagonal, total true instances = row sum
        tp = torch.diag(cm_tensor).float()
        total = cm_tensor.sum(dim=1).float()
        # avoid division‑by‑zero
        return torch.where(total == 0, torch.zeros_like(tp), tp / total)
        
    def _log_confusion_matrix(self, cm_tensor, tag: str):
        """
            :param cm_tensor : (C, C) detached int64 tensor of counts 
            :param tag       : logger prefix (e.g., "val_cm")
        """
        # convert confusion matrix to numpy
        cm_np = cm_tensor.numpy()

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
        """Log a 1×C detached accuracy tensor as a seaborn heat‑map."""
        acc_tensor = acc_tensor.numpy().reshape(1, -1)   # (1, C)

        labels = [
            (lbl[:20] + "...") if len(lbl) > 20 else lbl 
            for lbl in self.config.class_names
        ]

        fig, ax = plt.subplots(figsize=(max(6, self.config.classes * 0.4), 3))
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
        ax.set_xticklabels(labels, rotation=45, ha="right")
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
        seed = int(sys.argv[7]) if len(sys.argv) > 6 else int(os.getenv('SLURM_JOB_ID', default=random.randint(1,100)))
        print(f'\nSeed set to {seed}.')

        # Validate Paths
        assert datapath_args['images_lmdb'].is_dir(), f"Input directory does not exist or is not a directory: {datapath_args['images_lmdb']}"
        assert datapath_args['metadata_parquet'].is_file(), f"Input file does not exist or is not a file: {datapath_args['metadata_parquet']}"
        assert datapath_args['metadata_snow_cloud_parquet'].is_file(), f"Input file does not exist or is not a file: {datapath_args['metadata_snow_cloud_parquet']}"
        assert datapath_args['class_distribution_parquet'].is_file(), f"Input file does not exist or is not a file: {datapath_args['class_distribution_parquet']}"
        assert not export_dir_root.is_file(), f"Export needs to be a directory but is a file: {export_dir_root}"

    except Exception as e:
        # Handle Input Errors
        sys.exit(f'\nAn error occured: {e}.\n\nUsage: python train_lightning.py /path/to/lmdb_dir/ /path/to/metadata.parquet /path/to/metadata_snow_cloud.parquet /path/to/class_distribution.parquet /path/to/export/dir/ --seed [Number]\n')
    
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id is not None:
        slurm_job_id = int(slurm_job_id)
        slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
        if slurm_array_task_id is not None:
            slurm_array_task_id = int(slurm_array_task_id)
            export_dir = export_dir_root / f'resnet18_{start_time}_seed{seed}_task{slurm_array_task_id}_job{slurm_job_id}'
        else:
            export_dir = export_dir_root / f'resnet18_{start_time}_seed{seed}_job{slurm_job_id}'
    else:
        export_dir = export_dir_root / f'resnet18_{start_time}_seed{seed}'
    export_dir.mkdir(parents=True, exist_ok=True)

    tracker = RunTracker(seed=seed, slurm_job_id=slurm_job_id)

    # --- System Setup & Info ---
    torch.set_float32_matmul_precision('medium')

    slurm_cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
    omp_num_threads = os.getenv('OMP_NUM_THREADS')
    mkl_num_threads = os.getenv('MKL_NUM_THREADS')
    cuda_launch_blocking = os.getenv("CUDA_LAUNCH_BLOCKING")
    
    print(f"OMP_NUM_THREADS: {omp_num_threads}")
    print(f"MKL_NUM_THREADS: {mkl_num_threads}")
    print(f"CUDA_LAUNCH_BLOCKING: {cuda_launch_blocking}")
    print(f"SLURM_CPUS_PER_TASK: {slurm_cpus_per_task}")

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

    print(f"OS CPU count: {os.cpu_count()}")

    # --- Model, Dataset, Trainer Setup ---
    # Define Training Parameters
    tc = TrainLightningConfiguration(class_distribution=pd.read_parquet(datapath_args['class_distribution_parquet']))
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
    if slurm_cpus_per_task is not None:
        num_workers = int(slurm_cpus_per_task)-1 # -1 to keep a core for main and other processes
        
        print(f"Num_workers set to {num_workers} with SLURM_CPUS_PER_TASK")
    else:
        num_workers = max(1,min(os.cpu_count()//4-1,10)) # /2 for two sockets and /2 cause os.cpu_count returns thread as multithreads (2 per core) and -1 to keep some for the main process
        print(f"Num_workers set to {num_workers} with max(1,min(os.cpu_count()/4-1,10))")

    # Instantiate Model Config
    model_config = ILMConfiguration(
            timm_model_name=model_name,
            classes=tc.num_classes,
            class_names=NEW_LABELS_16,
            image_size=image_size,
            channels=number_of_channels,
            network_type=ILMType.IMAGE_CLASSIFICATION,
            drop_rate=tc.drop_rate,
            drop_path_rate=tc.drop_path_rate,
    )

    # Retrieve Model
    model = LitImageEncoder(config=model_config, lr=tc.lr, loss_fn=tc.loss_fn, class_weights=tc.class_weights, optimizer=tc.optimizer) 

    # Generate Lighting DataModule
    dm = BENv2_DataModule.BENv2DataModule(
            data_dirs = datapath_args,
            img_size = (number_of_channels, image_size, image_size),
            batch_size = tc.batch_size, 
            prefetch_factor = 2,
            train_transforms = tc.train_transforms,
            num_workers_dataloader = num_workers, 
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
            precision = '16-mixed', # for A100 GPUs (Automatic Mixed Precision)
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

    print(f"Number of workers dataloader: {dm.num_workers_dataloader}")

    if __debug__:
        export_dir_datasets = export_dir / "datasets"
        export_dir_datasets.mkdir(parents=True, exist_ok=True)
        dm.setup()
        dm.save_metadata(export_dir_datasets)
        print(f'Datasets saved in: {export_dir_datasets}')

    # --- Training And Testing ---
    training_start = time.time()

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm) # automatically loads best model from checkpoint

    training_end = time.time()

    s = int(training_end - training_start)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)

    # --- Run Info Collection and Logging ---
    tracker.log('Paths', 'Export Dir', str(export_dir))
    for key, val in datapath_args.items():
        # Make key readable (e.g. metadata_parquet -> Metadata Parquet)
        readable_key = key.replace('_', ' ').title()
        tracker.log('Paths', readable_key, str(val))
        
    if dm.train_ds is None: dm.setup()
    tracker.log('Dataset', 'Train Size', len(dm.train_ds) if dm.train_ds else "N/A")
    tracker.log('Dataset', 'Val Size', len(dm.val_ds) if dm.val_ds else "N/A")
    tracker.log('Dataset', 'Test Size', len(dm.test_ds) if dm.test_ds else "N/A")
    tracker.log('Dataset', 'Classes', model.config.classes)
    tracker.log('Dataset', 'Channels', model.config.channels)
    tracker.log('Dataset', 'Input Shape', dm.img_size)

    for env in ['SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_CPUS_PER_TASK', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_LAUNCH_BLOCKING']:
        tracker.log('Environment', env, os.getenv(env, 'Not Set'))
        
    if torch.cuda.is_available():
        tracker.log('System', 'GPU Count', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
             prop = torch.cuda.get_device_properties(i)
             tracker.log('System', f'GPU {i} Name', torch.cuda.get_device_name(i))
             tracker.log('System', f'GPU {i} VRAM (total)', f"{prop.total_memory / (1024**3):.2f} GB")
             tracker.log('System', f'GPU {i} VRAM (allocated)', f"{prop.allocated_memory / (1024**3):.2f} GB")
             tracker.log('System', f'GPU {i} VRAM (cached)', f"{prop.memory_cached / (1024**3):.2f} GB")

    else:
        tracker.log('System', 'GPU Count', 0)

    tracker.log('Model Config', 'Architecture', model.config.timm_model_name)
    tracker.log('Model Config', 'Drop Rate', model.config.drop_rate)
    tracker.log('Model Config', 'Drop Path Rate', model.config.drop_path_rate)
    tracker.log('Model Config', 'Total Params', sum(p.numel() for p in model.parameters()))
    tracker.log('Model Config', 'Trainable Params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    tracker.log('Training Config', 'Max Epochs', trainer.max_epochs)
    tracker.log('Training Config', 'Batch Size', dm.batch_size)
    tracker.log('Training Config', 'Num Workers', dm.num_workers_dataloader)
    tracker.log('Training Config', 'Patience', trainer.callbacks[0].patience)
    tracker.log('Training Config', 'Strategy', trainer.strategy.__class__.__name__)
    
    tracker.log('Optimization', 'Optimizer', model.optimizer.__name__ if hasattr(model.optimizer, '__name__') else str(model.optimizer))
    tracker.log('Optimization', 'Learning Rate', model.lr)
    
    loss_str = str(model.loss_fn)
    if hasattr(model.loss_fn, 'func'): loss_str = f"{model.loss_fn.func.__name__} {model.loss_fn.keywords}"
    elif hasattr(model.loss_fn, '__name__'): loss_str = model.loss_fn.__name__
    tracker.log('Optimization', 'Loss Function', loss_str)
    
    cw_np = model.class_weights.numpy() if isinstance(model.class_weights, torch.Tensor) else model.class_weights
    tracker.log('Optimization', 'Class Weights', str(list(cw_np)) if cw_np is not None else "None")

    tracker.log('Augmentation', 'Pipeline', tracker.format_augmentation(dm.train_transforms))

    # Load Results from TensorBoard
    tracker.populate_from_tensorboard(export_dir)

    # Save Everything
    export_dir_run_tracker = export_dir / 'run_tracker'
    export_dir_run_tracker.mkdir(parents=True, exist_ok=True)
    tracker.save(export_dir_run_tracker)

    print(f'\nTraining complete. Training duration (hh:mm:ss): {h:02d}:{m:02d}:{s:02d}')
    print(f'Best model and logs saved in: {export_dir}')

if __name__ == '__main__':
    main()
