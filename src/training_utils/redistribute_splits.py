from pathlib import Path
import numpy as np
import pandas as pd
import sys

#-----------------------------------------------------------------------------
# INIT 
try:
    datapath = {
        'metadata_single_label_parquet': Path(sys.argv[1]).resolve(),
        'metadata_single_label_snow_cloud_parquet': Path(sys.argv[2]).resolve(),
        'outfile': Path(sys.argv[3]).resolve(),
    }
    for key, path in datapath.items():
        if key != 'outfile':
            assert path.is_file(), f"Input file not found or not a file: {path}"
        else: 
            assert path.is_dir(), f"Output directory does not exist or is not a directory: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python redistribute_splits.py /path/to/metadata_single_label.parquet /path/to/metadata_single_label_snow_cloud.parquet /path/to/outdir/\n')

NEW_LABELS_ORIGINAL_ORDER = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

#-----------------------------------------------------------------------------
# EXEC
# read metadata
print(f'\nLoading metadata...')
print(f'\n--------------------------------------------------------------------------------')
metadata_single_label = pd.read_parquet(datapath["metadata_single_label_parquet"])
metadata_single_label_snow_cloud = pd.read_parquet(datapath["metadata_single_label_snow_cloud_parquet"])

# adjust split for each class
for idx, metadata in enumerate([metadata_single_label, metadata_single_label_snow_cloud]):

    print(f'\nRedistributing split for metadata{":" if idx == 0 else "_snow_cloud:"}')
    print(f'\n--------------------------------------------------------------------------------')

    for label in NEW_LABELS_ORIGINAL_ORDER:

        print(f'\nRedistributing split for label {label}:')

        metadata_specific_label = metadata[metadata['labels'] == label]
        metadata_specific_label_length = len(metadata_specific_label)
        print(f'\nItems for label {label}: {metadata_specific_label_length}')

        if metadata_specific_label_length >= 3:

            ten_percent = max(round(metadata_specific_label_length*0.1),1)
            twenty_percent = ten_percent*2
            print(f'Ten percent index: {ten_percent}')
            print(f'Twenty percent index: {twenty_percent}')

            print(f'\nMetadata for label {label}, before redistribution:\n{metadata_specific_label["split"].value_counts()}')

            writer = ['test'] * ten_percent + ['validation'] * ten_percent + ['train'] * (metadata_specific_label_length - twenty_percent)
            writer_length = len(writer)
            values, counts = np.unique(writer, return_counts=True) 
            writer_df = pd.DataFrame({"split": values, "": counts}).set_index("split")
            print(f'\nWriter for label {label}: {str(writer_df)}')

            # adjust all labels for metadata_specific_label simultaneously
            length_difference = writer_length - metadata_specific_label_length
            assert writer_length == metadata_specific_label_length, f"Writer and metadata length do not match. Length difference (writer:metadata): {length_difference}."
           
            metadata_specific_label.loc[:,'split'] = writer
            print(f'\nRedistributed metadata for label {label}, after adjusting all splits simultaneously:\n{metadata_specific_label["split"].value_counts()}')

        metadata.loc[metadata['labels'] == label,'split'] = metadata_specific_label['split'].sample(frac=1)
        print(f'\n--------------------------------------------------------------------------------')

    if idx == 0:
        metadata_single_label = metadata.copy(deep=True)
    else:
        metadata_single_label_snow_cloud = metadata.copy(deep=True)

print(f'\nSaving metadata...')
metadata_single_label.to_parquet(datapath['outfile'] / 'metadata.parquet', compression='gzip')
metadata_single_label_snow_cloud.to_parquet(datapath['outfile'] / 'metadata_for_snow_cloud_and_shadow.parquet', compression='gzip')

