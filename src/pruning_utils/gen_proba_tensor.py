# stdlib imports
import os
import sys
import time
import random
import logging
from pathlib import Path

# third-party imports
import torch
import lightning.pytorch as pl

# local imports
from configilm import util
from configilm.extra.DataModules import BENv2_DataModule
from scripts.train_lightning import LitImageEncoder
from shared_utils.custom_logging import create_logger

# --- Main Execution ---
def main():

    # --- 1. General Setup & Logging ---
    start_time = time.strftime('%Y%m%d-%H%M%S')

    # --- Logger Setup ---
    logger = create_logger("Gen_Proba_Logger", log_level=logging.DEBUG)
       
    if __debug__:
        logger.newline()
        logger.debug('--- DEBUG ON -- (Log level: DEBUG)')
        logger.newline()
        util.MESSAGE_LEVEL = util.MessageLevel.INFO
    else:
        logger.newline()
        logger.info('--- DEBUG OFF --- (Log level: INFO)')
        logger.newline()

    # --- Argument Parsing & Paths ---
    try:
        # Parse Paths
        datapath_args = {
            'images_lmdb': Path(sys.argv[1]).resolve(),
            'metadata_parquet': Path(sys.argv[2]).resolve(),
            'metadata_snow_cloud_parquet': Path(sys.argv[3]).resolve(),
            'checkpoint_file': Path(sys.argv[4]).resolve(),
        }
        export_dir_root = Path(sys.argv[5]).resolve()
        
        logger.newline()
        logger.debug("--- Input Paths ---")
        for key, path in datapath_args.items():
            logger.debug(f"{key:<28}: {path}")
        logger.debug(f"{'export_dir_root':<28}: {export_dir_root}")
        logger.debug("---------------------")
        logger.newline()

        # Parse Seed
        if len(sys.argv) == 6:
            seed = random.randint(1, 100)
        elif len(sys.argv) == 8 and sys.argv[6] == '--seed':
            seed = int(sys.argv[7])
        else:
            raise ValueError("Invalid argument count or format. Expected 5 paths, or 5 paths + --seed <number>.")

        # Validate Paths
        assert datapath_args['images_lmdb'].is_dir(), f"Input directory does not exist: {datapath_args['images_lmdb']}"
        assert datapath_args['metadata_parquet'].is_file(), f"Input file does not exist: {datapath_args['metadata_parquet']}"
        assert datapath_args['metadata_snow_cloud_parquet'].is_file(), f"Input file does not exist: {datapath_args['metadata_snow_cloud_parquet']}"
        assert datapath_args['checkpoint_file'].is_file(), f"Input file does not exist: {datapath_args['checkpoint_file']}"
        assert not export_dir_root.is_file(), f"Export path is a file, should be a directory: {export_dir_root}"

    except Exception as e:
        logger.newline()
        logger.critical(f'An error occured: {e}.\n\nUsage: python train.py /path/to/lmdb_dir/ /path/to/metadata.parquet /path/to/metadata_snow_cloud_parquet /path/to/class_distribution.parquet /path/to/export/dir/ --seed [Number]')
        logger.newline()
        sys.exit(1)

    # Robust model name extraction
    try:
        parts = datapath_args['checkpoint_file'].parts
        model_name = parts[parts.index('models') + 1]
    except (ValueError, IndexError):
        model_name = "unknown_model" # Fallback
        logger.newline()
        logger.warning(f"Could not parse model name from path. Using '{model_name}' as fallback.\n")
        logger.newline()

    # Create Export Directory
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id is not None:
        slurm_job_id = int(slurm_job_id)
        export_dir = export_dir_root / f'proba_{start_time}_seed{seed}_job{slurm_job_id}_model-{model_name}'
    else:
        export_dir = export_dir_root / f'proba-{start_time}_seed-{seed}_model-{model_name}'
    
    export_dir.mkdir(parents=True, exist_ok=True)
    logger.newline()
    logger.info(f"Export directory created: {export_dir}")
    logger.newline()
    
    # --- 2. System Info & Environment Setup ---

    # Setup Torch
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed, workers=True) # This will print the seed
    
    # Environment Information
    slurm_cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')

    logger.newline()
    logger.info("--- System & Environment Info ---")
    logger.info(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")
    logger.info(f"MKL_NUM_THREADS: {os.getenv('MKL_NUM_THREADS')}")
    logger.info(f"CUDA_LAUNCH_BLOCKING: {os.getenv('CUDA_LAUNCH_BLOCKING')}")
    logger.info(f"SLURM_CPUS_PER_TASK: {slurm_cpus_per_task}")
    logger.info(f"OS CPU count: {os.cpu_count()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: TRUE")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            cached_memory = torch.cuda.memory_cached(i) / (1024**3)
            logger.info(f"  Total memory: {total_memory:.2f} GB | Allocated: {allocated_memory:.2f} GB | Cached: {cached_memory:.2f} GB")
    else:
        logger.info(f"CUDA available: FALSE")
        logger.info("No GPUs found. Training on CPU.")
    logger.info("---------------------------------")
    logger.newline()

    # --- 3. Model Setup ---
    logger.newline()
    logger.info("Loading model from checkpoint...")
    logger.newline()
    model = LitImageEncoder.load_from_checkpoint(datapath_args['checkpoint_file']) 
    
    logger.newline()
    logger.debug("--- Model HParams ---")
    logger.debug(model.hparams)
    logger.debug("-----------------------")
    logger.newline()

    # --- 4. DataModule Setup ---
    # Setup Workers
    if slurm_cpus_per_task is not None:
        num_workers = int(slurm_cpus_per_task) - 2
        worker_source = "(from SLURM_CPUS_PER_TASK)"
    else:
        num_workers = max(1, min(os.cpu_count() // 4 - 2, 10))
        worker_source = "(from os.cpu_count())"

    # DataModule Parameters
    image_size = 120
    batch_size = 1024

    logger.newline()
    logger.info("--- DataModule Setup ---")
    logger.info(f"Num_workers set to {num_workers} {worker_source}")
    logger.info(f"Using Image Size: {image_size}x{image_size}")
    logger.info(f"Using Batch Size: {batch_size}")
    logger.info("--------------------------")
    logger.newline()

    # Generate Lighting DataModule
    dm = BENv2_DataModule.BENv2DataModule(
            data_dirs = datapath_args,
            img_size = (model.hparams.config.channels, image_size, image_size),
            batch_size = batch_size,
            train_transforms = None,
            num_workers_dataloader = num_workers, 
            bagging = True,
            seed = seed,
    )

    # --- 5. Trainer Setup ---
    trainer = pl.Trainer(
            inference_mode=True,
            max_epochs = -1,
            accelerator = 'auto',
            devices = 'auto',
            strategy = 'auto',
            logger = False,
            precision = '16-mixed',
            default_root_dir=export_dir,
            deterministic=True,
    )

    # Trainer Setup (Debug)
    WIDTH = 20
    logger.newline()
    logger.debug(f"--- Trainer Setup ---")
    logger.debug(f"{'Accelerator':<{WIDTH}}{'Devices':<{WIDTH}}{'Strategy':<{WIDTH}}")
    logger.debug(f"-" * (WIDTH * 3))
    logger.debug(f"{trainer.accelerator.__class__.__name__:<{WIDTH}}{trainer.num_devices:<{WIDTH}}{trainer.strategy.__class__.__name__:<{WIDTH}}")
    logger.debug(f"-" * (WIDTH * 3))
    logger.newline()

    # --- 6. Debug: Save Dataset Metadata ---
    if __debug__:
        export_dir_tmp = export_dir
        export_dir_debug_datasets = export_dir / "datasets"
        export_dir_debug_datasets.mkdir(parents=True, exist_ok=True)
        logger.newline()
        logger.debug("Running dm.setup() for metadata export...")
        dm.setup()
        dm.save_metadata(export_dir_debug_datasets)
        logger.debug(f'Datasets metadata saved in: {export_dir_debug_datasets}\n')
        logger.newline()
        export_dir = export_dir_tmp # Reset export_dir to original

    # --- 7. Run Inference ---
    logger.newline()
    logger.info("Starting inference...")
    logger.newline()
    preds = trainer.predict(model, datamodule=dm)
    
    torch.set_printoptions(precision=2, linewidth=80, threshold=1000, sci_mode=False)
    logger.newline()
    logger.info(f"Predictions Sample (from tensor):\n{preds[0]['predictions']}")
    logger.newline()
    
    prediction_file = export_dir / 'predictions.pt'
    torch.save(preds, prediction_file)

    logger.newline()
    logger.info('Inference complete.')
    logger.info(f'Predictions tensor and logs saved in: {export_dir}\n')
    logger.newline()

if __name__ == '__main__':
    main()
