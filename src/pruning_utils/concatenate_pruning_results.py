# std
import sys
import os
import time
import csv
import traceback

# third party
import pandas as pd
import argparse
from pathlib import Path
from alive_progress import alive_it, alive_bar

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
    parser.add_argument('--pruning-root', type=Path, required=True, help='''The path to the
        root directory of the run where the separate pruning outputs are 
        saved. The pruning root needs to be the SLURM_ARRAY_JOB_ID or the 
        timestamp of relevant pruning run''')
    parser.add_argument('--identifier', type=str, required=True, help='''The SLURM_
        ARRAY_JOB_ID or the timestamp of the job that ran the pruning task''')
    parser.add_argument('--target-size', type=int, nargs='+', required=True, help='''The target ensemble
        size to which the ensemble was pruned''')
    parser.add_argument('--outdir', type=Path, nargs='?', default=Path('.').resolve(), help='''The path to the output
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

        # create unique outdir
        slurm_job_id = os.getenv('SLURM_JOB_ID', default='')
        export_time = ('' if slurm_job_id != '' else time.strftime('%d-%m-%Y_%H-%M-%S'))
        target_size = '-'.join(str(n) for n in args.target_size)
        outdir = args.outdir.resolve() / f'{args.identifier}' / f'concatenated' / f'{target_size}' / f'{slurm_job_id}' / f'{export_time}'
        outdir.mkdir(parents=True, exist_ok=True)

        # find all results.csv for a given ensemble size of this pruning run 
        # target_size and identifier 
        pruning_root = args.pruning_root.resolve()

        # open a new file where all csv are concatenated
        with open(outdir / f'concatenated_results.csv', 'a') as outfile:

            # set first target_size
            target_size = args.target_size[0] 

            logger.info(f"Processing {target_size}...")

            # collect all csv's for first target_size
            pruning_results_csv = list(pruning_root.glob(f'{args.identifier}/separate/{target_size}/**/results.csv'))

            # go over all the csv's for first target_size (with manual progress bar)
            with alive_bar(len(pruning_results_csv)) as bar:

                # save information of first csv and prepend Target Size
                with open(pruning_results_csv[0], "r", encoding="utf-8") as first_csv:

                    # save header and prepend "Target Size"
                    outfile.write("Target Size," + first_csv.readline())

                    # save data and prepend target_size
                    for line in first_csv:
                        outfile.write(f"{target_size}," + line)

                    # forward progress bar
                    bar()

                # go over all remaining csv's
                for path in pruning_results_csv[1:]:

                    # save data of csv and prepend target_size (skipping header)
                    with open(path, "r", encoding="utf-8") as csv:

                        # skip header
                        csv.readline()
                        
                        # save data and prepend target_size
                        for line in csv:
                            outfile.write(f"{target_size}," + line)

                        # forward progress bar
                        bar()

            logger.info(f"Finished processing {target_size}")

            # for all remaining target sizes
            for target_size in args.target_size[1:]:

                logger.info(f"Processing {target_size}...")

                # collect all csv's for target_size
                pruning_results_csv = list(pruning_root.glob(f'{args.identifier}/separate/{target_size}/**/results.csv'))

                # go over all collected csv's (with automatic progress bar)
                for path in alive_it(pruning_results_csv):

                    # save data of csv and prepend target_size (skipping header)
                    with open(path, "r", encoding="utf-8") as csv:

                        # skip header
                        csv.readline()
                        
                        # save data and prepend target_size
                        for line in csv:
                            outfile.write(f"{target_size}," + line)

                logger.info(f"Finished processing {target_size}")

        # TODO: Convert csv to html while grouping the 'Target Size' column
        concatenated_pruning_results = pd.read_csv(outdir / f'concatenated_results.csv')

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
            <p><strong>Target Ensemble Size(s):</strong> {args.target_size}</p>
            <br>
            {html_string}
        </body>
        </html>
        """
        
        with open(outdir / f'concatenated_results.html', 'w') as outfile:
            outfile.write(final_html)
            
        print(f"\nResults successfully saved to {outdir}")

    except Exception as e:
        logger.critical(f'Error during processing: {e}')
        logger.critical(traceback.format_exc())
        sys.exit(1)
    
if __name__ == '__main__':
    main()
