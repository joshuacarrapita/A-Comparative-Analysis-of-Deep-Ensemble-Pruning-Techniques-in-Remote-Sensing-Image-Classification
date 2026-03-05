# std
import sys

# third party
import pandas as pd
import argparse
from pathlib import Path

# local libs
from shared_utils.custom_logging import create_logger

def parse_cli_args():
    
    # define parser
    parser = argparse.ArgumentParser(
            prog='Concatenate Pruning Results',
            description='''Concatenates the results produced by running multiple
                ensemble pruning algos separately on ensemble inference output''',
            )

    # define available parser arguments
    parser.add_argument('--pruning-root', type=Path, help='''The path to the
        root directory of the run where the separate pruning outputs are 
        saved. The pruning root needs to be the SLURM_ARRAY_JOB_ID or the 
        timestamp of relevant pruning run''')
    parser.add_argument('--slurm_array_job_id', type=int, help='''The SLURM_
        ARRAY_JOB_ID of the job that ran the pruning task''')
    parser.add_argument('--target-size', type=int, help='''The target ensemble
        size to which the ensemble was pruned''')
    parser.add_argument('--outdir', type=Path, help='''The path to the output
        directory where the combined csv should be saved.''')

    # check arguments format
    args = parser.parse_args()
    if not args['pruning-root'].is_dir():
        parser.error('--pruning-root is not a directory or doesn`t exist.')

    return args

def main():

    logger = create_logger("Concatenate Pruning Results Logger")

    try:
        args = parse_cli_args()

        pruning_root = args['pruning-root'].resolve()

        pruning_results_csv = pruning_root.glob(f'**/{args["slurm_array_job_id"]}/**/{args["target-size"]}/**/results.csv')

        concatenated_pruning_results = pd.concat([pd.read_csv(r) for r in pruning_results_csv])

        concatenated_pruning_results.to_csv(args['outdir'], index=False)

    except Exception as e:
        logger.critical(f'Error during processing: {e}')
        sys.exit(1)
    
if __name__ == '__main__':
    main()
