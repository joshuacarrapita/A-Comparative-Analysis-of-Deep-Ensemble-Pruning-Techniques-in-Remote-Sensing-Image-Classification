from pathlib import Path
import pandas as pd
import sys

#-----------------------------------------------------------------------------
# INIT 
try:
    datapath = {
        'multi_label_parquet': Path(sys.argv[1]).resolve(),
        'single_label_parquet': Path(sys.argv[2]).resolve(),
        'outfile': Path(sys.argv[3]).resolve(),
    }
    for key, path in datapath.items():
        if key != 'outfile':
            assert path.is_file(), f"Input file not found or not a file: {path}"
        else: 
            if path.is_dir():
                print(f'\nNo output filename provided. Saving to {path}/out.parquet')
                datapath['outfile'] = path / 'out.parquet'
            else:
                assert path.parent.is_dir(), f"Output directory does not exist or is not a directory: {path.parent}"
                assert path.name, f"No output filename provided: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python rewrite_metadata_parqet.py /path/to/multi_label.parquet /path/to/single_label.parquet /path/to/outfile.parquet\n')

#-----------------------------------------------------------------------------
# EXEC
# read multi labels
multi_labels = pd.read_parquet(datapath["multi_label_parquet"])

# read single labels
single_labels = pd.read_parquet(datapath["single_label_parquet"])

# merge metadata and single labels
metadata = pd.merge(multi_labels, single_labels, on="patch_id", how="inner", suffixes=("","_single"))

# replace multi with single labels
metadata["labels"] = metadata["labels_single"]

# delete superfluous single label column
metadata.drop(columns="labels_single", inplace=True)

metadata.to_parquet(datapath['outfile'], compression='gzip')
