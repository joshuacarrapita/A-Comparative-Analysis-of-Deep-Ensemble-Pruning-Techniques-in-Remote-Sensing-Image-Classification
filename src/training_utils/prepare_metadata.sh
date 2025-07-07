#! /bin/bash
# print parquet info
echo -e "\n------------------------------------------------------------------------------"
echo -e "\nPrinting parquet info:"
echo -e "\nMetadata parquet structure:"
(set -x; python -O /app/src/parquettest.py /app/data/metadata.parquet)
echo -e "\nMetadata snow parquet structure:"
(set -x; python -O /app/src/parquettest.py /app/data/metadata_for_patches_with_snow_cloud_or_shadow.parquet)

# extract single labels from reference maps
echo -e "\n------------------------------------------------------------------------------"
echo -e "\nExtracting single labels:"
(set -x; python -O /app/src/multi_to_single_label.py /app/data/Reference_Maps/ /app/data/single_labels.parquet)

# overwrite parquet files
echo -e "\n------------------------------------------------------------------------------"
echo -e "\nOverwriting parquet files:"
(set -x; python -O /app/src/rewrite_metadata_parquet.py /app/data/metadata.parquet /app/data/single_labels.parquet /app/data/metadata_single_labels.parquet)
(set -x; python -O /app/src/rewrite_metadata_parquet.py /app/data/metadata_for_patches_with_snow_cloud_or_shadow.parquet /app/data/single_labels.parquet /app/data/metadata_for_patches_with_snow_cloud_or_shadow_single_labels.parquet)

# print parquet info
echo -e "\n------------------------------------------------------------------------------"
echo -e "\nPrinting single label parquet info:"
echo -e "\Single labels parquet structure:"
(set -x; python -O /app/src/parquettest.py /app/data/single_labels.parquet)
echo -e "\nMetadata single labels parquet structure:"
(set -x; python -O /app/src/parquettest.py /app/data/metadata_single_labels.parquet)
echo -e "\nMetadata snow single labels parquet structure:"
(set -x; python -O /app/src/parquettest.py /app/data/metadata_for_patches_with_snow_cloud_or_shadow_single_labels.parquet)

# cleanup
echo -e "\nThe single label parquet files have been created and can now be found in your mounted volume directory as 'single_labels.parquet', 'metadata_single_labels.parquet' and 'metadata_for_patches_with_snow_cloud_or_shadow_single_labels.parquet'!"
echo -e "\nUse them just like 'metadata.parquet' and 'metadata_for_patches_with_snow_cloud_or_shadow.parquet'!"
