import sys
import os
import time
import platform
import torch
import pandas as pd
import lightning.pytorch as pl
import base64
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from custom_logging import create_logger

class RunTracker:
    def __init__(self, seed: int, slurm_job_id: Any):
        self.logger = create_logger("RunTracker")
        self.start_time_monotonic = time.time()
        self.start_timestamp = time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())
        self.slurm_job_id = slurm_job_id if slurm_job_id else "Local"
        self.seed = seed
        
        self.data: Dict[str, Dict[str, Any]] = {
            'General': {},
            'System': {},
            'Environment': {},
            'Paths': {},
            'Dataset': {},
            'Model Config': {},
            'Training Config': {},
            'Optimization': {},
            'Augmentation': {},
            'Results (Train)': {},
            'Results (Validation)': {},
            'Results (Test)': {},
        }
        
        self._log_basic_info()

    def _log_basic_info(self):
        self.data['General']['Date'] = self.start_timestamp
        self.data['General']['Seed'] = self.seed
        self.data['General']['Job ID'] = self.slurm_job_id
        self.data['General']['Command'] = " ".join(sys.argv)
        
        self.data['System']['OS'] = platform.platform()
        self.data['System']['Node Name'] = platform.node()
        self.data['System']['Python Version'] = sys.version.split()[0]
        self.data['System']['PyTorch Version'] = torch.__version__
        self.data['System']['Lightning Version'] = pl.__version__
        self.data['System']['Numpy Version'] = np.__version__
        self.data['System']['CPU Count (Total)'] = os.cpu_count()

    def log(self, category: str, key: str, value: Any):
        if value is None:
            value = "N/A"
        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value

    def populate_from_tensorboard(self, export_dir: Path):
        """
        Parses the TensorBoard tfevents file to extract Best Val and Final Test metrics.
        """
        self.logger.info("Parsing TensorBoard logs for results...")
        
        # locate the events file (usually in **/lightning_logs/version_0/)
        log_dir = export_dir / "lightning_logs"
        if not log_dir.exists():
            self.logger.warning(f"Log dir {log_dir} not found. Skipping TB parsing.")
            return

        # find latest version directory
        versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("version_")], 
                          key=lambda x: int(x.name.split('_')[1]), reverse=True)
        
        if not versions:
            event_files = list(log_dir.glob("events.out.tfevents*"))
        else:
            event_files = list(versions[0].glob("events.out.tfevents*"))
            
        if not event_files:
            self.logger.warning("No events.out.tfevents file found.")
            return

        # use the largest event file
        event_path = str(max(event_files, key=os.path.getsize))

        # load Events
        try:
            ea = EventAccumulator(event_path, size_guidance={'scalars': 0, 'images': 0}) # size_guidance = 0 loads *all* the data
            ea.Reload()
            tags = ea.Tags()['scalars']
            image_tags = ea.Tags()['images']
            self.logger.info(f"Found TensorBoard tags: {tags}")
            self.logger.info(f"Found TensorBoard image tags: {image_tags}")
        except Exception as e:
            self.logger.error(f"Error loading TensorBoard file: {e}")
            return

        # --- Helper Functions ---
        
        def _get_scalar_at_step(tag_name: str, step: Optional[int] = None):
            """Returns scalar value at a specific step, or None."""
            if tag_name not in tags:
                return None

            scalars = ea.Scalars(tag_name)

            if step is not None and step != -1:
                # try to find scalar at exact step, fallback to last scalar
                return next((x.value for x in scalars if x.step == step), scalars[-1].value)
            else:
                # default to last scalar (for Train/Test)
                return scalars[-1].value

        def _get_image_html_at_step(tag_name: str, step: Optional[int] = None):
            """
            Finds an image by tag. If step is provided, tries to find that specific step.
            Otherwise (or if step not found), returns the last logged image.
            Returns formatted HTML string or None.
            """
            if tag_name not in image_tags:
                return None
            
            images = ea.Images(tag_name)
            
            if step is not None and step != -1:
                # try to find image at exact step, fallback to last image
                target_img = next((x for x in images if x.step == step), images[-1])
            else:
                # default to last image (for Train/Test)
                target_img = images[-1]

            # convert to Base64
            b64_str = base64.b64encode(target_img.encoded_image_string).decode('utf-8')
            return f'<img src="data:image/png;base64,{b64_str}" style="max-width: 300px; height: auto;" />'

        # --- Extract Final Training Metrics ---
        if 'train_loss' in tags: self.log('Results (Train)', 'Final Train Loss', _get_scalar_at_step('train_loss', step=-1))
        if 'train_acc' in tags: self.log('Results (Train)', 'Final Train Acc', _get_scalar_at_step('train_acc', step=-1))
        if 'train_avg_class_acc' in tags: self.log('Results (Train)', 'Final Train Avg Class Acc', _get_scalar_at_step('train_avg_class_acc', step=-1))
        if 'train_per_class_acc' in tags: self.log('Results (Train)', 'Per Class Acc', _get_image_html_at_step('train_per_class_acc', step=-1))

        # --- Extract Best Epoch Validation Metrics ---
        if 'val_loss' in tags:
            # find event with min value
            best_event = min(ea.Scalars('val_loss'), key=lambda x: x.value)
            self.log('Results (Validation)', 'Best Val Loss', _get_scalar_at_step('val_loss', step=best_event.step))
            
            if 'val_acc' in tags: self.log('Results (Validation)', 'Best Val Acc', _get_scalar_at_step('val_acc', best_event.step))
            if 'val_avg_class_acc' in tags: self.log('Results (Validation)', 'Best Val Avg Class Acc', _get_scalar_at_step('val_avg_class_acc', best_event.step))
            if 'val_per_class_acc' in tags: self.log('Results (Validation)', 'Per Class Acc', _get_image_html_at_step('val_per_class_acc', step=best_event.step))
        
        # --- Extract Test Metrics ---
        if 'test_loss' in tags: self.log('Results (Test)', 'Test Loss', _get_scalar_at_step('test_loss', step=-1))
        if 'test_acc' in tags: self.log('Results (Test)', 'Test Acc', _get_scalar_at_step('test_acc', step=-1))
        if 'test_avg_class_acc' in tags: self.log('Results (Test)', 'Test Avg Class Acc', _get_scalar_at_step('test_avg_class_acc', step=-1))
        if 'test_per_class_acc' in tags: self.log('Results (Test)', 'Per Class Acc', _get_image_html_at_step('test_per_class_acc', step=-1))

    def format_augmentation(self, transforms):
        """
        Formats the augmentation pipeline string to be readable with HTML line breaks.
        Replaces standard newlines with <br> and spaces with &nbsp;.
        """
        raw_str = str(transforms)
        
        # replace newlines with HTML breaks
        formatted = raw_str.replace('\n', '<br><br>')
        
        # replace indentation spaces (usually 2 or 4) with non-breaking spaces
        formatted = formatted.replace('  ', '&nbsp;&nbsp;')
        
        return formatted

    def save(self, export_dir: Path):
        duration_seconds = time.time() - self.start_time_monotonic
        m, s = divmod(duration_seconds, 60)
        h, m = divmod(m, 60)
        self.log('General', 'Duration', f"{int(h)}h {int(m)}m {int(s)}s")
        self.log('General', 'Duration (Seconds)', f"{duration_seconds:.2f}")

        self.logger.info("\nGenerating Run Info Report...")
        
        flat_data = {}
        for category, params in self.data.items():
            for param_name, param_value in params.items():
                flat_data[(category, param_name)] = param_value
        
        if not flat_data:
            self.logger.warning("RunTracker is empty. Nothing to save.")
            return

        df = pd.DataFrame([flat_data])
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        df_T = df.T
        
        html_path = export_dir / "run_info.html"
        pkl_path = export_dir / "run_info.pkl"
        csv_path = export_dir / "run_info.csv"
        
        # HTML with some basic CSS for readability of the long augmentation string
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
                h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; box-shadow: 0 2px 15px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
                th {{ background-color: #007bff; color: white; font-weight: 600; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .long-text {{ white-space: pre-wrap; font-family: monospace; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h2>Run Information: Job {self.slurm_job_id}</h2>
            <p><strong>Date:</strong> {self.start_timestamp} | <strong>Seed:</strong> {self.seed}</p>
            {df_T.to_html(classes='table', escape=False)} 
        </body>
        </html>
        """
        # note: escape=False is needed to render the <br> tags in augmentation string AND the <img> tags for heatmaps
        
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        df.to_pickle(pkl_path)
        df.to_csv(csv_path, index=False)

        self.logger.info(f'Run Info saved to: {html_path}')
