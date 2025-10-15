from pathlib import Path
import pandas as pd
import sys

#-----------------------------------------------------------------------------
# INIT 
try:
    datapath = {
        'metadata.parquet': Path(sys.argv[1]).resolve(),
        'metadata_for_patches_with_snow_cloud_or_shadow.parquet': Path(sys.argv[2]).resolve(),
        'outdir': Path(sys.argv[3]).resolve(),
    }
    for key, path in datapath.items():
        if key != 'outdir':
            assert path.is_file(), f"Input file not found or not a file: {path}"
        else: 
            assert path.is_dir(), f"Output directory does not exist or is not a directory: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python metadata_to_class_distribution.py /path/to/metadata.parquet /path/to/metadata_for_patches_with_snow_cloud_or_shadow.parquet /path/to/outdir/\n')

#-----------------------------------------------------------------------------
# EXEC
# read normal metadata
metadata = pd.read_parquet(datapath["metadata.parquet"])

# read metadata_for_patches_with_snow_cloud_or_shadow
metadata_snow_cloud = pd.read_parquet(datapath["metadata_for_patches_with_snow_cloud_or_shadow.parquet"])

# calculate occurances for all versions
label_counts_metadata = metadata['labels'].astype(str).value_counts().to_frame(name='counts').reset_index(names='labels')
label_freq_metadata = metadata['labels'].astype(str).value_counts(normalize=True).to_frame(name='frequencies').reset_index(names='labels')
label_counts_metadata_snow_cloud = metadata_snow_cloud['labels'].astype(str).value_counts().to_frame(name='counts').reset_index(names='labels')
label_freq_metadata_snow_cloud = metadata_snow_cloud['labels'].astype(str).value_counts(normalize=True).to_frame(name='frequencies').reset_index(names='labels')
label_counts_both = label_counts_metadata.add(label_counts_metadata_snow_cloud, fill_value=0)
label_freq_both = label_freq_metadata.merge(label_freq_metadata_snow_cloud, how='left', on='labels', suffixes=('_no_snow_cloud','_yes_snow_cloud'))


# output to file
label_counts = {
    'label_counts_metadata': label_counts_metadata,
    'label_freq_metadata': label_freq_metadata,
    'label_counts_metadata_snow_cloud': label_counts_metadata_snow_cloud,
    'label_freq_metadata_snow_cloud': label_freq_metadata_snow_cloud,
    'label_counts_both': label_counts_both,
    'label_freq_both': label_freq_both,
}

for name, variable in label_counts.items():
    outfile = datapath['outdir'] / f'{name}.parquet'
    variable.to_parquet(outfile, compression='gzip')
    print(f"\nName: {name}\nContent:\n{variable.to_string()}")
