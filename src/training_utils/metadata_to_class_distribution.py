from pathlib import Path
import pandas as pd
import sys

#-----------------------------------------------------------------------------
# INIT 
try:
    input_args = {
        'metadata.parquet': Path(sys.argv[1]).resolve(),
        'metadata_for_patches_with_snow_cloud_or_shadow.parquet': Path(sys.argv[2]).resolve(),
        'outdir': Path(sys.argv[3]).resolve(),
        'full': str(sys.argv[4]) if len(sys.argv) > 4 else ''
    }
    for key, path in input_args.items():
        if key != 'outdir':
            assert path.is_file(), f"Input file not found or not a file: {path}"
        else: 
            assert path.is_dir(), f"Output directory does not exist or is not a directory: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python metadata_to_class_distribution.py /path/to/metadata.parquet /path/to/metadata_for_patches_with_snow_cloud_or_shadow.parquet /path/to/outdir/\n')

#-----------------------------------------------------------------------------
# EXEC
# read metadata
metadata = pd.read_parquet(input_args["metadata.parquet"])
metadata_snow_cloud = pd.read_parquet(input_args["metadata_for_patches_with_snow_cloud_or_shadow.parquet"])

# calculate distributions per split ('train', 'validation', 'test') using crosstab and sort descending by total
metadata_class_distribution_absolute_all_splits = pd.crosstab(metadata['labels'].astype(str), metadata['split'], margins=True, margins_name='total').sort_values('total', ascending=False).reset_index()
metadata_class_distribution_relative_all_splits = pd.crosstab(metadata['labels'].astype(str), metadata['split'], margins=True, margins_name='total', normalize='columns').sort_values('total', ascending=False).reset_index()

metadata_snow_cloud_class_distribution_absolute_all_splits = pd.crosstab(metadata_snow_cloud['labels'].astype(str), metadata_snow_cloud['split'], margins=True, margins_name='total').sort_values('total', ascending=False).reset_index()
metadata_snow_cloud_class_distribution_relative_all_splits = pd.crosstab(metadata_snow_cloud['labels'].astype(str), metadata_snow_cloud['split'], margins=True, margins_name='total', normalize='columns').sort_values('total', ascending=False).reset_index()

metadata_both = pd.concat([metadata, metadata_snow_cloud])
both_class_distribution_absolute_all_splits = pd.crosstab(metadata_both['labels'].astype(str), metadata_both['split'], margins=True, margins_name='total').sort_values('total', ascending=False).reset_index()
both_class_distribution_relative_all_splits = pd.crosstab(metadata_both['labels'].astype(str), metadata_both['split'], margins=True, margins_name='total', normalize='columns').sort_values('total', ascending=False).reset_index()

both_class_distribution_relative_old_all_splits = metadata_class_distribution_relative_all_splits.merge(metadata_snow_cloud_class_distribution_relative_all_splits, how='left', on='labels', suffixes=('_no_snow_cloud','_yes_snow_cloud')).sort_values('total_no_snow_cloud', ascending=False)

# calculate counts using series (index = label)
s_counts_meta = metadata['labels'].astype(str).value_counts()
s_counts_snow = metadata_snow_cloud['labels'].astype(str).value_counts()

# add counts (matching their indices which are labels in this case)
s_counts_both = s_counts_meta.add(s_counts_snow, fill_value=0).astype(int).sort_values(ascending=False)

# calculate frequencies
s_freq_meta = metadata['labels'].astype(str).value_counts(normalize=True)
s_freq_snow = metadata_snow_cloud['labels'].astype(str).value_counts(normalize=True)

# calculate combined frequency based on the total combined counts
s_freq_both = s_counts_both / s_counts_both.sum()

# format outputs (reset index)
metadata_class_distribution_absolute = s_counts_meta.to_frame(name='counts').reset_index(names='labels')
metadata_class_distribution_relative = s_freq_meta.to_frame(name='frequencies').reset_index(names='labels')

metadata_snow_cloud_class_distribution_absolute = s_counts_snow.to_frame(name='counts').reset_index(names='labels')
metadata_snow_cloud_class_distribution_relative = s_freq_snow.to_frame(name='frequencies').reset_index(names='labels')

both_class_distribution_absolute = s_counts_both.to_frame(name='counts').reset_index(names='labels')
both_class_distribution_relative = s_freq_both.to_frame(name='frequencies').reset_index(names='labels')

both_class_distribution_relative_old = metadata_class_distribution_relative.merge(metadata_snow_cloud_class_distribution_relative, how='left', on='labels', suffixes=('_no_snow_cloud','_yes_snow_cloud'))


# output to file
all_class_distributions = {
    'metadata_class_distribution_absolute': metadata_class_distribution_absolute,
    'metadata_class_distribution_absolute_all_splits': metadata_class_distribution_absolute_all_splits,
    'metadata_class_distribution_relative': metadata_class_distribution_relative,
    'metadata_class_distribution_relative_all_splits': metadata_class_distribution_relative_all_splits,
    'metadata_snow_cloud_class_distribution_absolute': metadata_snow_cloud_class_distribution_absolute,
    'metadata_snow_cloud_class_distribution_absolute_all_splits': metadata_snow_cloud_class_distribution_absolute_all_splits,
    'metadata_snow_cloud_class_distribution_relative': metadata_snow_cloud_class_distribution_relative,
    'metadata_snow_cloud_class_distribution_relative_all_splits': metadata_snow_cloud_class_distribution_relative_all_splits,
    'both_class_distribution_absolute': both_class_distribution_absolute,
    'both_class_distribution_absolute_all_splits': both_class_distribution_absolute_all_splits,
    'both_class_distribution_relative': both_class_distribution_relative,
    'both_class_distribution_relative_all_splits': both_class_distribution_relative_all_splits,
    'both_class_distribution_relative_old': both_class_distribution_relative_old,
    'both_class_distribution_relative_old_all_splits': both_class_distribution_relative_old_all_splits,
}

for name, variable in all_class_distributions.items():
    if input_args['full'] == "--full" or {name} == "metadata_class_distribution_absolute":
        outfile = input_args['outdir'] / f'{name}.parquet'
        variable.to_parquet(outfile, compression='gzip')
    print(f"\nName: {name}\nContent:\n{variable.to_string()}")
