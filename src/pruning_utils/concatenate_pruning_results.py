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
from glob import glob as lnsglob
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
    parser.add_argument('--identifier', type=str, default='', help='''The SLURM_
        ARRAY_JOB_ID or the timestamp of the job that ran the pruning task; if not provided
        --pruning-root needs to contain all and only jobs relevant for concatenation''')
    parser.add_argument('--target-size', type=int, nargs='*', default=[], help='''The target ensemble
        size(s) for which you want to concatenate results; concatenates all if empty''')
    parser.add_argument('--outdir', type=Path, nargs='?', default=Path('.').resolve(), help='''The path to the output
        directory where the combined csv should be saved''')
    parser.add_argument('--list-missing', action='store_true', help='''Print missing algorithms for each target size and append them to the concatenated results csv at the respective target size''')

    # check arguments format
    args = parser.parse_args()
    if not args.pruning_root or not args.pruning_root.is_dir():
        parser.error('--pruning_root is not a directory or doesn`t exist.')

    return args

def main():

    logger = create_logger("Concatenate Pruning Results Logger")

    # define the expected structure for the missing check
    EXPECTED_ALGOS = {
        "EPFD": ["ES", "KP", "KL", "RE", "OO", "DREP", "SEP", "OEP", "PEP", "GMA", "LCS", "COMEP", "DOMEP"],
        "GASEN": ["GASEN"],
        "PyPruning": [
            "individual_margin_diversity", "individual_contribution", "individual_error",
            "individual_kappa_statistic", "reduced_error", "complementariness", "drep",
            "margin_distance", "combined", "reference_vector", "combined_error", "error_ambiguity",
            "largest_mean_distance", "cluster_accuracy", "cluster_centroids", "random"
        ],
        "Baseline": ["bsm", "wsm", "random_no_replacement"]
    }

    try:
        logger.info(f"Parsing input args...")
        args = parse_cli_args()

        # parse paths
        pruning_root = args.pruning_root.resolve()
        outdir = args.outdir.resolve()

        # create unique outdir
        logger.info(f"Creating outdir...")
        slurm_job_id = os.getenv('SLURM_JOB_ID', default='')
        export_time = ('' if slurm_job_id != '' else time.strftime('%d-%m-%Y_%H-%M-%S'))
        target_size = '-'.join(str(n) for n in args.target_size)
        outdir = outdir / f'{args.identifier}' / f'concatenated' / f'{target_size}' / f'{slurm_job_id}' / f'{export_time}'
        outdir.mkdir(parents=True, exist_ok=True)

        # open a new file where all csv are concatenated
        logger.info(f"Creating concatenated_results.csv...")
        with open(outdir / f'concatenated_results.csv', 'a') as outfile:

            # set first target_size
            if not args.target_size:
                logger.info(f"Collecting target sizes...")
                with alive_bar() as bar:
                    args.target_size = sorted(list({int(Path(x).stem) for x in lnsglob(f"{pruning_root}/**/separate/*", recursive=True)}))
                    bar()
            target_size = args.target_size[0] 
            logger.info(f"Target sizes collected: {args.target_size}")

            logger.info(f"Processing target size: {target_size}...")

            # collect all csv's for first target_size
            logger.info(f"Collecting results.csv...")
            with alive_bar() as bar:
                pruning_results_csv = list(lnsglob(f'{pruning_root}/**/{args.identifier}/separate/{target_size}/**/results.csv', recursive=True))
                logger.info(f"Collecting DONE")
                bar()

            # go over all the csv's for first target_size (with manual progress bar)
            logger.info(f"Concatenating results.csv...")
            found_combos = set()
            with alive_bar(len(pruning_results_csv)) as bar:

                # save information of first csv and prepend Target Size
                if pruning_results_csv:
                    p0_parts = Path(pruning_results_csv[0]).parts
                    sep_idx = p0_parts.index('separate')
                    found_combos.add((p0_parts[sep_idx+2], p0_parts[sep_idx+3]))
                    
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
                    p_parts = Path(path).parts
                    sep_idx = p_parts.index('separate')
                    found_combos.add((p_parts[sep_idx+2], p_parts[sep_idx+3]))

                    # save data of csv and prepend target_size (skipping header)
                    with open(path, "r", encoding="utf-8") as csv:
                        # skip header
                        csv.readline()
                        
                        # save data and prepend target_size
                        for line in csv:
                            outfile.write(f"{target_size}," + line)

                        # forward progress bar
                        bar()

            # check for missing algorithms
            if args.list_missing:
                for lib, algs in EXPECTED_ALGOS.items():
                    for alg in algs:
                        if (lib, alg) not in found_combos:
                            logger.info(f"Missing (target size {target_size}): {lib} - {alg}")
                            outfile.write(f'{target_size},{lib},{alg},-1,"missing"\n')

            logger.info(f"Concatenating DONE")
            logger.info(f"Finished processing target size: {target_size}")

            # for all remaining target sizes
            for target_size in args.target_size[1:]:

                logger.info(f"Processing target size: {target_size}...")

                # collect all csv's for target_size
                logger.info(f"Collecting results.csv...")
                with alive_bar() as bar:
                    pruning_results_csv = list(lnsglob(f'{pruning_root}/**/{args.identifier}/separate/{target_size}/**/results.csv', recursive=True))
                    logger.info(f"Collecting DONE")
                    bar()

                # go over all collected csv's (with automatic progress bar)
                logger.info(f"Concatenating results.csv...")
                found_combos = set()
                for path in alive_it(pruning_results_csv):
                    p_parts = Path(path).parts
                    sep_idx = p_parts.index('separate')
                    found_combos.add((p_parts[sep_idx+2], p_parts[sep_idx+3]))

                    # save data of csv and prepend target_size (skipping header)
                    with open(path, "r", encoding="utf-8") as csv:

                        # skip header
                        csv.readline()
                        
                        # save data and prepend target_size
                        for line in csv:
                            outfile.write(f"{target_size}," + line)

                # check for missing algorithms
                if args.list_missing:
                    for lib, algs in EXPECTED_ALGOS.items():
                        for alg in algs:
                            if (lib, alg) not in found_combos:
                                logger.info(f"Missing (target size {target_size}): {lib} - {alg}")
                                outfile.write(f'{target_size},{lib},{alg},-1,"missing"\n')

                logger.info(f"Concatenating DONE")
                logger.info(f"Finished processing target size: {target_size}")

        # convert csv to html while grouping the 'Target Size' column
        logger.info(f"Exporting to html...")
        with alive_bar() as bar:
            logger.info(f"Reading csv...")
            concatenated_pruning_results = pd.read_csv(outdir / f'concatenated_results.csv').set_index(['Target Size', 'Library', 'Algorithm'])

            logger.info(f"Sorting csv by 'Runtime' ascending...")
            concatenated_pruning_results = concatenated_pruning_results.groupby(level=["Target Size", "Library"], group_keys=False).apply(lambda g: g.sort_values("Runtime"))

            # export results to html
            logger.info(f"Converting to html...")
            html_string = concatenated_pruning_results.to_html(classes='table table-striped')
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
            
            logger.info(f"Saving...")
            with open(outdir / f'concatenated_results.html', 'w') as outfile:
                outfile.write(final_html)

            bar()
            
        logger.info(f"\nResults successfully saved to {outdir}")

    except Exception as e:
        logger.critical(f'Error during processing: {e}')
        logger.critical(traceback.format_exc())
        sys.exit(1)
    
if __name__ == '__main__':
    main()
