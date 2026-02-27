# import stdlibs
import os
import sys
import time
import random
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# import 3rd party libs
import torch
import lightning.pytorch as pl
import pandas as pd
import argparse

# import local libs
from configilm import util
from configilm.extra.DataModules import BENv2_DataModule
from training_utils.lit_models import LitImageEncoder
from shared_utils.custom_logging import create_logger

# define global variables
MODEL_CHECKPOINT_SUFFIX = '.ckpt'

# --- create global logger
if __debug__:
    logger = create_logger("EnsemblePruningLogger", log_level=logging.DEBUG)
    logger.newline()
    logger.debug('--- DEBUG ON -- (Log level: DEBUG)')
    logger.newline()
    util.MESSAGE_LEVEL = util.MessageLevel.INFO
else:
    logger = create_logger("EnsemblePruningLogger", log_level=logging.INFO)
    logger.newline()
    logger.info('--- DEBUG OFF --- (Log level: INFO)')
    logger.newline()

# define functions
def parse_cli_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('images_lmdb', type=Path)
    parser.add_argument('metadata_parquet', type=Path)
    parser.add_argument('metadata_snow_cloud_parquet', type=Path)
    parser.add_argument('model_checkpoints_dir', type=Path)
    parser.add_argument('base_export_dir', type=Path)
    parser.add_argument('--seed', type=int, default=666)
    
    args = parser.parse_args()
    
    # resolve paths
    args = {key: value.resolve() if isinstance(value, Path) else value for key, value in vars(args).items()}
    
    # validate paths 
    if not args['images_lmdb'].is_dir():
        parser.error(f"Input directory does not exist: {args['images_lmdb']}")
    if not args['metadata_parquet'].is_file():
        parser.error(f"Input file does not exist: {args['metadata_parquet']}")
    if not args['metadata_snow_cloud_parquet'].is_file():
        parser.error(f"Input file does not exist: {args['metadata_snow_cloud_parquet']}")
    if not args['model_checkpoints_dir'].is_dir():
        parser.error(f"Model checkpoint directory does not exist: {args['model_checkpoints_dir']}")
    if not args['base_export_dir'].is_dir():
        parser.error(f"Export path is not a directory or does not exist: {args['base_export_dir']}")
    
    return args

def get_model_name(model_checkpoint_path: Path) -> str:
    try:
        parts = model_checkpoint_path.parts
        logs_index = parts.index('lightning_logs') # all checkpoint paths should contain the term 'lightning_logs'
        model_name = parts[logs_index - 1]
    except (ValueError, IndexError):
        logger.newline()
        logger.critical(f"Could not parse model name from path (expected '.../<model_name>/lightning_logs/...'). "
        sys.exit(1)
        logger.newline()
    return model_name

def log_system_info():
    logger.newline()
    logger.info("--- System & Environment Info ---")
    logger.info(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")
    logger.info(f"MKL_NUM_THREADS: {os.getenv('MKL_NUM_THREADS')}")
    logger.info(f"CUDA_LAUNCH_BLOCKING: {os.getenv('CUDA_LAUNCH_BLOCKING')}")
    logger.info(f"SLURM_CPUS_PER_TASK: {os.getenv('SLURM_CPUS_PER_TASK')}")
    logger.info(f"OS CPU count: {os.cpu_count()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: TRUE")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
            logger.info(f"  Total memory: {total_memory:.2f} GB | Allocated: {allocated_memory:.2f} GB | Reserved: {reserved_memory:.2f} GB")
    else:
        logger.info(f"CUDA available: FALSE")
        logger.info("No GPUs found. Training on CPU.")
    logger.info("---------------------------------")
    logger.newline()

def setup_environment(seed: int):
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed, workers=True)

def load_model(model_checkpoint_path: Path) -> LitImageEncoder:
    logger.newline()
    logger.info("Loading model from checkpoint ...")
    logger.newline()
    model = LitImageEncoder.load_from_checkpoint(model_checkpoint_path)
    logger.newline()
    logger.debug("--- Model HParams ---")
    logger.debug(model.hparams)
    logger.debug("-----------------------")
    logger.newline()
    return model

def setup_datamodule(run_config: Dict[str, Any]) -> BENv2_DataModule.BENv2DataModule:
    slurm_cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus_per_task is not None:
        num_workers_dataloader = int(slurm_cpus_per_task) - 2
        worker_source = "(from SLURM_CPUS_PER_TASK)"
    else:
        num_workers_dataloader = max(1, min(os.cpu_count() // 4 - 2, 10))
        worker_source = "(from os.cpu_count())"

    channels = 10 #2: "Sentinel-1", 10: "10m + 20m Sentinel-2 (in original order)"
    image_size = 120
    batch_size = 1024
    seed = run_config['seed']

    logger.newline()
    logger.info("--- DataModule Setup ---")
    logger.info(f"Num_workers_dataloader set to {num_workers_dataloader} {worker_source}")
    logger.info(f"Using Channels: {channels}")
    logger.info(f"Using Image Size: {image_size}x{image_size}")
    logger.info(f"Using Batch Size: {batch_size}")
    logger.info(f"Using Seed: {seed}")
    logger.info("--------------------------")
    logger.newline()

    data_dirs = {
        'images_lmdb': run_config['images_lmdb'],
        'metadata_parquet': run_config['metadata_parquet'],
        'metadata_snow_cloud_parquet': run_config['metadata_snow_cloud_parquet'],
    }

    dm = BENv2_DataModule.BENv2DataModule(
        data_dirs=data_dirs,
        img_size=(channels, image_size, image_size),
        batch_size=batch_size,
        train_transforms=None,
        num_workers_dataloader=num_workers_dataloader,
        bagging=True,
        seed=seed,
    )

    return dm

def setup_trainer(default_root_dir: Path) -> pl.Trainer:
    trainer = pl.Trainer(
        inference_mode=True,
        max_epochs=-1,
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=False,
        precision='16-mixed',
        default_root_dir=default_root_dir,
        deterministic=True,
    )

    WIDTH = 20
    logger.newline()
    logger.debug(f"--- Trainer Setup ---")
    logger.debug(f"{'Accelerator':<{WIDTH}}{'Devices':<{WIDTH}}{'Strategy':<{WIDTH}}")
    logger.debug(f"-" * (WIDTH * 3))
    logger.debug(f"{trainer.accelerator.__class__.__name__:<{WIDTH}}{trainer.num_devices:<{WIDTH}}{trainer.strategy.__class__.__name__:<{WIDTH}}")
    logger.debug(f"-" * (WIDTH * 3))
    logger.newline()

    return trainer

def save_debug_metadata(datamodule: pl.LightningDataModule, export_dir: Path):
    if __debug__:
        export_dir = export_dir / "inference" / "datasets"
        export_dir.mkdir(parents=True, exist_ok=True)
        logger.newline()
        logger.debug("Running dm.setup() for metadata export ...")
        datamodule.setup()
        datamodule.save_metadata(export_dir)
        logger.debug(f'Datasets metadata saved in: {export_dir}\n')
        logger.newline()

def execute_inference(trainer: pl.Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.newline()
    logger.info("Executing inference...")
    logger.newline()

    inference_outputs = trainer.predict(model, datamodule=datamodule)
    
    model_inference_results = torch.unsqueeze(inference_outputs[0]["prediction_results"], 0)
    model_target_labels = inference_outputs[0]["target_labels"]

    torch.set_printoptions(precision=2, linewidth=80, threshold=1000, sci_mode=False)
    logger.newline()
    logger.info(f"Inference results sample (from tensor of shape {model_inference_results.shape}):\n{model_inference_results}")
    logger.info(f"Inference target labels sample (from tensor of shape {model_target_labels.shape}):\n{model_target_labels}")
    logger.newline()

    return model_inference_results, model_target_labels

def save_model_inference_results(model_inference_results: torch.Tensor, export_dir: Path, start_time: str, seed: int, model_name: str):
    export_dir = export_dir / f'{model_name}' / 'inference' / 'results.pt'
    torch.save(model_inference_results, export_dir)

    logger.newline()
    logger.info('Inference complete.')
    logger.info(f'Individual model inference results tensor saved to: {export_dir}\n')
    logger.newline()

def log_run_setup_info(run_config: Dict[str, Any], model_checkpoint_path: Path):
    logger.newline()
    logger.debug("--- Input Paths & Config ---")
    for key, path in run_config.items():
        logger.debug(f"{key:<28}: {path}")
    logger.debug(f"model_checkpoint_path: {model_checkpoint_path}")
    logger.debug("------------------------------")
    logger.newline()

def run_model_inference(run_config: Dict[str, Any], model_checkpoint_path: Path, model_name: str, datamodule: pl.LightningDataModule, trainer: pl.Trainer, export_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]: 
    start_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    seed = run_config['seed']

    log_run_setup_info(run_config, model_checkpoint_path)

    setup_environment(seed)
    
    model = load_model(model_checkpoint_path)
    
    model_inference_results, model_target_labels = execute_inference(trainer, model, datamodule)
    
    save_model_inference_results(model_inference_results, export_dir, start_time, seed, model_name)

    logger.info(f'Returning model inference results with shape {model_inference_results.shape} and target labels with shape {model_target_labels.shape} for aggregation.')
    logger.newline()

    return model_inference_results, model_target_labels

def find_model_checkpoint_paths(model_checkpoints_dir: Path, checkpoint_suffix: str) -> list[Path]:
    logger.newline()
    logger.info(f"Searching for {checkpoint_suffix} in {model_checkpoints_dir} ...")
    logger.newline()
    found_files = list(model_checkpoints_dir.rglob(f'*{checkpoint_suffix}'))
    if not found_files:
        logger.critical("No model checkpoint files found.")
        sys.exit(0)

    model_checkpoint_paths = []
    for file in found_files:
        try:
            model_checkpoint_paths.append(file.resolve())
        except Exception as e:
            logger.error(f"ERROR converting checkpoint file {file} to model checkpoint path: {e}", exc_info=__debug__)
            logger.error(f"Continuing checkpoint file conversion ...")

    
    logger.info(f"Found {len(model_checkpoint_paths)} model checkpoint(s). Starting inference loop ...")

    return model_checkpoint_paths

def aggregate_and_save_single_model_inference_results(results: list, target_labels: torch.Tensor, member_ids_to_ckpt_paths: list, export_dir: Path):
    logger.newline()
    logger.info("--- All processing complete ---")
        
    if not results:
        logger.info("No inference results were generated from any model.")
    elif len(results) == 1:
        logger.info("No ensemble aggregation was performed for a single model inference.")
    else:
        export_dir = export_dir / 'inference'
        aggregated_results = torch.cat(results, dim=0)
        
        logger.newline()
        logger.info(f"Final ensemble inference results shape: {aggregated_results.shape}")
        torch.set_printoptions(precision=2, linewidth=80, threshold=1000, sci_mode=False)
        logger.info(f"Ensemble inference results sample (from tensor):\n{aggregated_results}")
        logger.info(f"Ensemble inference target labels (from tensor):\n{target_labels}")

        try:
            aggregated_results_file = export_dir / f'results.pt'
            target_labels_file = export_dir / f'target_labels.pt'
            member_ids_to_ckpt_paths_file = export_dir / f'member_to_checkpoint_paths.csv'
            
            torch.save(aggregated_results, aggregated_results_file)
            torch.save(target_labels, target_labels_file)
            pd.DataFrame(member_ids_to_ckpt_paths, columns=['Path']).to_csv(member_ids_to_ckpt_paths_file, index_label='Index')

            logger.info(f"Aggregated ensemble inference results saved to: {aggregated_results_file}")
            logger.info(f"Ensemble inference target labels saved to: {target_labels_file}")
            logger.info(f"Ensemble member checkpoint path list saved to: {member_ids_to_ckpt_paths_file}")
        except Exception as e:
            logger.critical(f"Error saving aggregated ensemble tensors: {e}")

def main():
    start_time = time.strftime('%Y-%m-%d_%H-%M-%S')

    run_config = parse_cli_arguments()

    try:
        # get array task ID or whatever the models were collected in
        stem = run_config['model_checkpoints_dir'].stem 

        slurm_job_id = os.getenv('SLURM_JOB_ID')
        
        if slurm_job_id is not None:
            run_config['base_export_dir'] = run_config['base_export_dir'] / f'{stem}' / f'{slurm_job_id}'
        else:
            run_config['base_export_dir'] = run_config['base_export_dir'] / f'{stem}' / f'{start_time}'
        run_config['base_export_dir'].mkdir(parents=True, exist_ok=True)
            
        run_config['full_ensemble_export_dir'] = base_export_dir / 'full_ensemble'
        run_config['full_ensemble_export_dir'].mkdir(parents=True, exist_ok=True)

        run_config['single_models_export_dir'] = base_export_dir / 'single_models'
        run_config['single_models_export_dir'] .mkdir(parents=True, exist_ok=True)

    except Exception as e:
        logger.critical(f"Error creating ensemble inference directory: {e}")
        sys.exit(1)

    log_system_info()

    all_inference_results = []
    all_model_checkpoint_paths = []
    reference_target_labels = torch.tensor([])

    model_checkpoint_paths = find_model_checkpoint_paths(run_config['model_checkpoints_dir'], MODEL_CHECKPOINT_SUFFIX)

    datamodule = setup_datamodule(run_config)
    trainer = setup_trainer(run_config['single_models_export_dir'])

    save_debug_metadata(datamodule, run_config['full_ensemble_export_dir'])

    for model_checkpoint_path in model_checkpoint_paths:
        model_name = get_model_name(model_checkpoint_path)
        logger.info(f"--- Processing model: {model_name}, checkpoint: {model_checkpoint_path.name} ---")
        try:
            model_inference_results, model_target_labels = run_model_inference(run_config, model_checkpoint_path, model_name, datamodule=datamodule, trainer=trainer, run_config['single_models_export_dir'])
            
            if model_inference_results.numel() > 0:
                all_inference_results.append(model_inference_results)
                all_model_checkpoint_paths.append(str(model_checkpoint_path))
                if reference_target_labels.numel() == 0:
                    reference_target_labels = model_target_labels
                else:
                    if not torch.equal(reference_target_labels, model_target_labels):
                        logger.critical(f"Targets from previous model do not match targets from this model.")
                        sys.exit(1)
            else:
                logger.warning(f"Warning: run_model_inference() returned an empty tensor for {model_checkpoint_path}")
        except AssertionError as ae:
            logger.critical(f"Target mismatch between models: {ae}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"ERROR processing {model_checkpoint_path}: {e}", exc_info=__debug__)
            sys.exit(1)

    aggregate_and_save_single_model_inference_results(all_inference_results, reference_target_labels, all_model_checkpoint_paths, run_config['full_ensemble_export_dir'])

if __name__ == '__main__':
    main()
