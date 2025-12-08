import sys
import os
import time
import platform
import torch
import pandas as pd
import lightning.pytorch as pl
from pathlib import Path
from typing import Any, Dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class RunTracker:
    def __init__(self, seed: int, slurm_job_id: Any):
        self.start_time_monotonic = time.time()
        self.start_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
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
        self.data['System']['CPU Count (Total)'] = os.cpu_count()

    def log(self, category: str, key: str, value: Any):
        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value

    def populate_from_tensorboard(self, export_dir: Path):
        """
        Parses the TensorBoard tfevents file to extract Best Val and Final Test metrics.
        """
        print("Parsing TensorBoard logs for results...")
        
        # 1. Locate the events file (usually in lightning_logs/version_0)
        log_dir = export_dir / "lightning_logs"
        if not log_dir.exists():
            print(f"Warning: Log dir {log_dir} not found. Skipping TB parsing.")
            return

        # Find latest version directory
        versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("version_")], 
                          key=lambda x: int(x.name.split('_')[1]), reverse=True)
        
        if not versions:
            event_files = list(log_dir.glob("events.out.tfevents*"))
        else:
            event_files = list(versions[0].glob("events.out.tfevents*"))
            
        if not event_files:
            print("Warning: No events.out.tfevents file found.")
            return

        # Use the largest event file
        event_path = str(max(event_files, key=os.path.getsize))

        # 2. Load Events
        try:
            ea = EventAccumulator(event_path, size_guidance={ 'scalars': 0 })
            ea.Reload()
            tags = ea.Tags()['scalars']
            print(f"Found TensorBoard tags: {tags}") # Debug print
        except Exception as e:
            print(f"Error loading TensorBoard file: {e}")
            return

        # 3. Extract Validation Metrics (Find Best Epoch)
        if 'val_loss' in tags:
            val_loss_events = ea.Scalars('val_loss')
            # Find event with min value
            best_event = min(val_loss_events, key=lambda x: x.value)
            
            self.log('Results (Validation)', 'Best Epoch', best_event.step)
            self.log('Results (Validation)', 'Best Val Loss', best_event.value)
            
            # Helper to find corresponding values at the best step
            def log_at_step(tag_name, display_name):
                if tag_name in tags:
                    events = ea.Scalars(tag_name)
                    # Find value closest to the best step (Lightning usually logs same step)
                    val = next((x.value for x in events if x.step == best_event.step), "N/A")
                    self.log('Results (Validation)', display_name, val)

            log_at_step('val_acc', 'Best Val Acc')
            log_at_step('val_avg_class_acc', 'Best Val Avg Class Acc')
        
        # 4. Extract Test Metrics
        # Test metrics are usually logged at the very end. 
        # We search for explicit test tags.
        test_tags = [t for t in tags if 'test_' in t]
        if test_tags:
            for tag in test_tags:
                final_val = ea.Scalars(tag)[-1].value
                clean_name = tag.replace('test_', '').replace('_', ' ').title() # e.g. "Acc" or "Avg Class Acc"
                self.log('Results (Test)', clean_name, final_val)
        else:
            # Fallback: Check if we have them but named differently or just missing
            print("Warning: No tags with 'test_' prefix found in TensorBoard.")

        # 5. Extract Final Training Metrics
        for tag, name in [('train_loss', 'Final Train Loss'), 
                          ('train_acc', 'Final Train Acc'), 
                          ('train_avg_class_acc', 'Final Train Avg Class Acc')]:
            if tag in tags:
                final_val = ea.Scalars(tag)[-1].value
                self.log('Results (Train)', name, final_val)

    def format_augmentation(self, transforms):
        """
        Formats the augmentation pipeline string to be readable with HTML line breaks.
        Replaces standard newlines with <br> and spaces with &nbsp;.
        """
        raw_str = str(transforms)
        
        # 1. Replace newlines with HTML breaks
        formatted = raw_str.replace('\n', '<br><br>')
        
        # 2. Replace indentation spaces (usually 2 or 4) with non-breaking spaces
        # We replace pairs of spaces to maintain approximate visual structure
        formatted = formatted.replace('  ', '&nbsp;&nbsp;')
        
        return formatted

    def save(self, export_dir: Path):
        duration_seconds = time.time() - self.start_time_monotonic
        m, s = divmod(duration_seconds, 60)
        h, m = divmod(m, 60)
        self.log('General', 'Duration', f"{int(h)}h {int(m)}m {int(s)}s")
        self.log('General', 'Duration (Seconds)', f"{duration_seconds:.2f}")

        print("\nGenerating Run Info Report...")
        
        flat_data = {}
        for category, params in self.data.items():
            for param_name, param_value in params.items():
                flat_data[(category, param_name)] = param_value
        
        if not flat_data:
            print("Warning: RunTracker is empty. Nothing to save.")
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
        # Note: escape=False is needed to render the <br> tags in augmentation string
        
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        df.to_pickle(pkl_path)
        df.to_csv(csv_path, index=False)

        print(f'Run Info saved to: {html_path}')
