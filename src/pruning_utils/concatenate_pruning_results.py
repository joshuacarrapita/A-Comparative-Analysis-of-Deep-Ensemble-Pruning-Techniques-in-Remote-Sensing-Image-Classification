# std
import sys
import os
import time

# third party
import pandas as pd
import argparse
from pathlib import Path

# local libs
from shared_utils.custom_logging import create_logger

def parse_cli_args():
    
    # define parser
    parser = argparse.ArgumentParser(
            prog='concatenate_pruning_results.py',
            description='''Concatenates the results produced by running multiple
                ensemble pruning algos separately on ensemble inference output''',
            )

    # define available parser arguments
    parser.add_argument('--pruning_root', type=Path, help='''The path to the
        root directory of the run where the separate pruning outputs are 
        saved. The pruning root needs to be the SLURM_ARRAY_JOB_ID or the 
        timestamp of relevant pruning run''')
    parser.add_argument('--identifier', type=str, help='''The SLURM_
        ARRAY_JOB_ID or the timestamp of the job that ran the pruning task''')
    parser.add_argument('--target_size', type=int, help='''The target ensemble
        size to which the ensemble was pruned''')
    parser.add_argument('--outdir', type=Path, help='''The path to the output
        directory where the combined csv should be saved.''')

    # check arguments format
    args = parser.parse_args()
    if not args.pruning_root or not args.pruning_root.is_dir():
        parser.error('--pruning_root is not a directory or doesn`t exist.')

    return args

def main():

    logger = create_logger("Concatenate Pruning Results Logger")

    try:
        args = parse_cli_args()

        # find all results.csv for a given ensemble size of this pruning run 
        # target_size and identifier 
        pruning_root = args.pruning_root.resolve()

        pruning_results_csv = pruning_root.glob(f'**/{args.identifier}/**/{args.target_size}/**/results.csv')

        # concatenate these results
        concatenated_pruning_results = pd.concat([pd.read_csv(r) for r in pruning_results_csv])

        # create unique outdir
        slurm_job_id = os.getenv('SLURM_JOB_ID', default='')
        export_time = ('' if slurm_job_id != '' else time.strftime('%d-%m-%Y_%H-%M-%S'))
        outdir = args.outdir / f'{args.identifier}' / f'{args.target_size}' / f'{slurm_job_id}' / f'{export_time}'
        outdir.mkdir(parents=True, exist_ok=True)

        # export results to csv
        concatenated_pruning_results.to_csv(outdir / f'results.csv', index=False)
    
        # export results to html
        html_string = concatenated_pruning_results.to_html(classes='table table-striped', index=False)
        final_html = f"""
        <html>
        <head>
            <title>Ensemble Pruning Benchmark Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>Ensemble Pruning Results</h2>
            <p><strong>Input Pruning Run (Identifier):</strong> {args.identifier}</p>
            <p><strong>Target Ensemble Size:</strong> {args.target_size}</p>
            <br>
            {html_string}
        </body>
        </html>
        """
        
        with open(outdir / f'results.html', 'w') as f:
            f.write(final_html)
            
        print(f"\nResults successfully saved to {args.outdir}")

    except Exception as e:
        logger.critical(f'Error during processing: {e}')
        sys.exit(1)
    
if __name__ == '__main__':
    main()
