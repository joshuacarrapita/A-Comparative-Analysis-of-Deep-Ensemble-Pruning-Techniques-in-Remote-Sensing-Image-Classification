# === IMPORTS ===
# stdlib imports
import argparse
import sys

# 3-rd party imports
from pathlib import Path
from alive_progress import alive_bar

# local imports
from shared_utils.custom_logging import create_logger


# === GLOBAL VARIABLES ===
logger = create_logger("Analyse Evaluation Results Logger")


# === HELPER FUNCTIONS ===
def parse_args():

    # init parser
    parser = argparse.ArgumentParser()

    # add args
    parser.add_argument("evaluation_dir", help="Path to directory with evaluation results")

    # parse args
    args = parser.parse_args()

    # print info about parsed args
    for key, val in vars(args).items():
        logger.info(f"Parsed the following arguments:")
        logger.info(f"{key}: {val}")

    # return namespace object
    return args


# === MAIN EXECUTION LOOP ===
def main():

    # parse args
    args = parse_args()
    
    # TODO: create plots for each pruning algorithm and each global value
    #       over all ensemble sizes

    # resolve evaluation_dir to path
    try:
        evaluation_path = Path(args.evaluation_dir).resolve()
        if not evaluation_path.is_dir():
            if not evaluation_path.exists():
                raise FileNotFoundError(f"Evaluation directory not found")
            else:
                raise TypeError(f"Evaluation directory is not a directory")
    except FileNotFoundError:
        logger.critical(f"Evaluation directory not found")
        raise
        sys.exit(1)
    except TypeError:
        logger.critical(f"Evaluation directory is not a directory")
        raise
        sys.exit(1)

    # retrieve evaluation results
    with alive_bar(1) as bar:
        logger.info(f"Retrieving evaluation results...")
        evaluation_results = evaluation_path.rglob(f'metrics_results.json')
        bar()
        print([x for x in evaluation_results])


if __name__ == '__main__':
    main()
