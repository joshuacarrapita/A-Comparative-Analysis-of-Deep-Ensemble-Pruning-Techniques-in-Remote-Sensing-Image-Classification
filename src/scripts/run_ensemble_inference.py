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
    parser.add_argument('all_model_checkpoints_dir', type=Path)
    parser.add_argument('base_export_dir', type=Path)
    parser.add_argument('--seed', type=int, default=666)
    
    args = parser.parse_args()
    
    # resolve paths
    results = {key: value.resolve() if isinstance(value, Path) else value for key, value in vars(args).items()}
    
    # validate paths 
    if not results['images_lmdb'].is_dir():
        parser.error(f"Input directory does not exist: {results['images_lmdb']}")
    if not results['metadata_parquet'].is_file():
        parser.error(f"Input file does not exist: {results['metadata_parquet']}")
    if not results['metadata_snow_cloud_parquet'].is_file():
        parser.error(f"Input file does not exist: {results['metadata_snow_cloud_parquet']}")
    if not results['all_model_checkpoints_dir'].is_dir():
        parser.error(f"Model checkpoint directory does not exist: {results['all_model_checkpoints_dir']}")
    if not results['base_export_dir'].is_dir():
        parser.error(f"Export path is not a directory or does not exist: {results['base_export_dir']}")
    
    return results

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

def setup_trainer(ensemble_inference_dir: Path) -> pl.Trainer:
    trainer = pl.Trainer(
        inference_mode=True,
        max_epochs=-1,
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=False,
        precision='16-mixed',
        default_root_dir=ensemble_inference_dir,
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

def run_debug_metadata_save(datamodule: pl.LightningDataModule, ensemble_inference_dir: Path):
    if __debug__:
        export_dir_debug_datasets = ensemble_inference_dir / "datasets"
        export_dir_debug_datasets.mkdir(parents=True, exist_ok=True)
        logger.newline()
        logger.debug("Running dm.setup() for metadata export ...")
        datamodule.setup()
        datamodule.save_metadata(export_dir_debug_datasets)
        logger.debug(f'Datasets metadata saved in: {export_dir_debug_datasets}\n')
        logger.newline()

def execute_inference(trainer: pl.Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.newline()
    logger.info("Executing inference to generate results...")
    logger.newline()

    inference_outputs = trainer.predict(model, datamodule=datamodule)
    
    model_inference_results = torch.unsqueeze(inference_outputs[0]["prediction_results"], 0)
    model_inference_target_labels = inference_outputs[0]["target_labels"]

    torch.set_printoptions(precision=2, linewidth=80, threshold=1000, sci_mode=False)
    logger.newline()
    logger.info(f"Inference results sample (from tensor of shape {model_inference_results.shape}):\n{model_inference_results}")
    logger.info(f"Inference target labels sample (from tensor of shape {model_inference_target_labels.shape}):\n{model_inference_target_labels}")
    logger.newline()

    return model_inference_results, model_inference_target_labels

def save_model_inference_results(model_inference_results: torch.Tensor, ensemble_inference_dir: Path, start_time: str, seed: int, model_name: str):
    results_file_path = ensemble_inference_dir / f'{model_name}.pt'
    torch.save(model_inference_results, results_file_path)

    logger.newline()
    logger.info('Inference complete.')
    logger.info(f'Individual model inference results tensor saved to: {results_file_path}\n')
    logger.newline()

def log_run_setup_info(run_config: Dict[str, Any], model_checkpoint_path: Path):
    logger.newline()
    logger.debug("--- Input Paths & Config ---")
    for key, path in run_config.items():
        logger.debug(f"{key:<28}: {path}")
    logger.debug(f"model_checkpoint_path: {model_checkpoint_path}")
    logger.debug("------------------------------")
    logger.newline()

def run_model_inference(run_config: Dict[str, Any], model_checkpoint_path: Path, model_name: str, datamodule: pl.LightningDataModule, trainer: pl.Trainer) -> Tuple[torch.Tensor, torch.Tensor]: 
    start_time = time.strftime('%Y%m%d-%H%M%S')
    seed = run_config['seed']
    ensemble_inference_dir = run_config['ensemble_inference_dir']

    log_run_setup_info(run_config, model_checkpoint_path)

    setup_environment(seed)
    
    model = load_model(model_checkpoint_path)
    
    model_inference_results, model_inference_target_labels = execute_inference(trainer, model, datamodule)
    
    save_model_inference_results(model_inference_results, ensemble_inference_dir, start_time, seed, model_name)

    logger.info(f'Returning inference results with shape {model_inference_results.shape} and inference target labels with shape {model_inference_target_labels.shape} for aggregation.')
    logger.newline()

    return model_inference_results, model_inference_target_labels

def find_model_checkpoint_paths(all_model_checkpoints_dir: Path, checkpoint_suffix: str) -> list[Path]:
    logger.newline()
    logger.info(f"Searching for {checkpoint_suffix} in {all_model_checkpoints_dir} ...")
    logger.newline()
    found_files = list(all_model_checkpoints_dir.rglob(f'*{checkpoint_suffix}'))
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

def aggregate_and_save_ensemble_inference_results(all_inference_results: list, ensemble_inference_target_labels: torch.Tensor, ensemble_member_checkpoint_paths: list, ensemble_inference_dir: Path):
    logger.newline()
    logger.info("--- All processing complete ---")
        
    if not all_inference_results:
        logger.info("No inference results were generated from any model.")
    elif len(all_inference_results) == 1:
        logger.info("No ensemble aggregation was performed for a single model inference.")
    else:
        ensemble_inference_results = torch.cat(all_inference_results, dim=0)
        
        logger.newline()
        logger.info(f"Final ensemble inference results shape: {ensemble_inference_results.shape}")
        torch.set_printoptions(precision=2, linewidth=80, threshold=1000, sci_mode=False)
        logger.info(f"Ensemble inference results sample (from tensor):\n{ensemble_inference_results}")
        logger.info(f"Ensemble inference target labels (from tensor):\n{ensemble_inference_target_labels}")
        
        try:
            export_path_ensemble_inference_results = ensemble_inference_dir / f'ensemble_inference_results.pt'
            export_path_ensemble_inference_target_labels = ensemble_inference_dir / f'ensemble_inference_target_labels.pt'
            export_path_ensemble_member_checkpoint_paths = ensemble_inference_dir / f'ensemble_member_checkpoint_paths.csv'
            
            torch.save(ensemble_inference_results, export_path_ensemble_inference_results)
            torch.save(ensemble_inference_target_labels, export_path_ensemble_inference_target_labels)
            pd.DataFrame(ensemble_member_checkpoint_paths, columns=['Path']).to_csv(export_path_ensemble_member_checkpoint_paths, index_label='Index')
            
            logger.info(f"Aggregated ensemble inference results saved to: {export_path_ensemble_inference_results}")
            logger.info(f"Ensemble inference target labels saved to: {export_path_ensemble_inference_target_labels}")
            logger.info(f"Ensemble member checkpoint path list saved to: {export_path_ensemble_member_checkpoint_paths}")
        except Exception as e:
            logger.critical(f"Error saving aggregated ensemble tensors: {e}")

def main():
    start_time = time.strftime('%Y%m%d-%H%M%S')

    run_config = parse_cli_arguments()

    try:
        stem = run_config['all_model_checkpoints_dir'].stem

        # TODO: adjust export dir and everything to match the new dir scheme
        base_export_dir = run_config['base_export_dir']
        slurm_job_id = os.getenv('SLURM_JOB_ID')
        
        if slurm_job_id is not None:
            run_dir_name = f'inference_{slurm_job_id}_{start_time}'
        else:
            run_dir_name = f'inference_{start_time}'
            
        ensemble_inference_dir = base_export_dir / run_dir_name
        ensemble_inference_dir.mkdir(parents=True, exist_ok=True)
        run_config['ensemble_inference_dir'] = ensemble_inference_dir
    except Exception as e:
        logger.critical(f"Error creating ensemble inference directory: {e}")
        sys.exit(1)

    log_system_info()

    ensemble_inference_dir = run_config['ensemble_inference_dir']

    all_inference_results = []
    all_inference_model_checkpoint_paths = []
    reference_inference_target_labels = torch.tensor([])

    model_checkpoint_paths = find_model_checkpoint_paths(run_config['all_model_checkpoints_dir'], MODEL_CHECKPOINT_SUFFIX)

    datamodule = setup_datamodule(run_config)
    trainer = setup_trainer(ensemble_inference_dir)

    run_debug_metadata_save(datamodule, ensemble_inference_dir)

    for model_checkpoint_path in model_checkpoint_paths:
        model_name = get_model_name(model_checkpoint_path)
        logger.info(f"--- Processing model: {model_name}, checkpoint: {model_checkpoint_path.name} ---")
        try:
            model_inference_results, current_inference_target_labels = run_model_inference(run_config, model_checkpoint_path, model_name, datamodule=datamodule, trainer=trainer)
            
            if model_inference_results.numel() > 0:
                all_inference_results.append(model_inference_results)
                all_inference_model_checkpoint_paths.append(str(model_checkpoint_path))
                if reference_inference_target_labels.numel() == 0:
                    reference_inference_target_labels = current_inference_target_labels
                else:
                    assert torch.equal(reference_inference_target_labels, current_inference_target_labels), "Targets from previous model do not match targets from this model."
            else:
                logger.warning(f"Warning: run_model_inference() returned an empty tensor for {model_checkpoint_path}")
        except AssertionError as ae:
            logger.critical(f"Target mismatch between models: {ae}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"ERROR processing {model_checkpoint_path}: {e}", exc_info=__debug__)
            sys.exit(1)

    aggregate_and_save_ensemble_inference_results(all_inference_results, reference_inference_target_labels, all_inference_model_checkpoint_paths, run_config['ensemble_inference_dir'])

if __name__ == '__main__':
    main()
