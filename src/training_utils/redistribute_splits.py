from pathlib import Path
import numpy as np
import pandas as pd
import sys

#-----------------------------------------------------------------------------
# HELPER 
def HalfRoundUp(value):
    return int(value + 0.5)

#-----------------------------------------------------------------------------
# INIT 
try:
    input_args = {
        'metadata_single_label_parquet': Path(sys.argv[1]).resolve(),
        'metadata_single_label_snow_cloud_parquet': Path(sys.argv[2]).resolve(),
        'outfile': Path(sys.argv[3]).resolve(),
        'train_percent': int(sys.argv[4]),
        'validation_percent': int(sys.argv[5]),
        'test_percent': int(sys.argv[6]),
    }
    sum = 0
    for key, value in input_args.items():
        if key == 'metadata_single_label_parquet' or key == 'metadata_single_label_snow_cloud_parquet':
            assert value.is_file(), f"Input file not found or not a file: {value}"
        elif key == 'outfile': 
            assert value.is_dir(), f"Output directory does not exist or is not a directory: {value}"
        else:
            # convert to mathematic notation and make sure it adds up to 100
            if value > 1:
                input_args[key] /= 100
            sum += value
    assert sum == 1, f"Input split does not add up to 100% total: {sum}"

except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python redistribute_splits.py /path/to/metadata_single_label.parquet /path/to/metadata_single_label_snow_cloud.parquet /path/to/outdir/ [train percent] [validation percent] [test percent]\n')

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
metadata_single_label = pd.read_parquet(input_args["metadata_single_label_parquet"])
metadata_single_label_snow_cloud = pd.read_parquet(input_args["metadata_single_label_snow_cloud_parquet"])

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

            # split the data into sections but make sure we don't assign more than we can
            train_absolute = max(HalfRoundUp(metadata_specific_label_length*input_args["train_percent"]),1)
            remaining_length = metadata_specific_label_length - train_absolute
            validation_absolute = min(max(HalfRoundUp(metadata_specific_label_length*input_args["validation_percent"]),1), remaining_length)
            remaining_length = remaining_length - validation_absolute
            test_absolute = min(max(HalfRoundUp(metadata_specific_label_length*input_args["test_percent"]),1), remaining_length)

            print(f'Total data (100%): {metadata_specific_label_length}')
            print(f'Train data (~{input_args["train_percent"]*100}%): {train_absolute}')
            print(f'Validation data (~{input_args["validation_percent"]*100}%): {validation_absolute}')
            print(f'Test data (~{input_args["test_percent"]*100}%): {test_absolute}')

            print(f'\nMetadata for label {label}, before redistribution:\n{metadata_specific_label["split"].value_counts()}')

            writer = ['test'] * test_absolute + ['validation'] * validation_absolute + ['train'] * train_absolute
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
metadata_single_label.to_parquet(input_args["outfile"] / 'metadata_single_labels.parquet', compression='gzip')
metadata_single_label_snow_cloud.to_parquet(input_args["outfile"] / 'metadata_for_patches_with_snow_cloud_or_shadow_single_labels.parquet', compression='gzip')
