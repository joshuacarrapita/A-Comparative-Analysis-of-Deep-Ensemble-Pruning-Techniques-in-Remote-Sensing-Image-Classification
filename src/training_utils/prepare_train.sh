#!/bin/bash

# print parquet info
echo -e "\nPrinting parquet info:"
echo -e "\nMetadata parquet structure:"
python parquettest.py /app/data/metadata.parquet
echo -e "\nMetadata snow parquet structure:"
python parquettest.py /app/data/metadata_snow.parquet

# extract single labels from reference maps
echo -e "\nExtracting single labels:"
python multi_to_single_label.py /app/data/Reference_Maps/ /app/data/single_labels.parquet

# overwrite parquet files
echo -e "\nOverwriting parquet files:"
python rewrite_metadata_parqet.py /app/data/metadata.parquet app/data/single_labels.parquet /app/data/metadata_single_labels.parquet
python rewrite_metadata_parqet.py /app/data/metadata_snow.parquet app/data/single_labels.parquet /app/data/metadata_snow_single_labels.parquet

# print parquet info
echo -e "\nPrinting single label parquet info:"
echo -e "\Single labels parquet structure:"
python parquettest.py /app/data/single_labels.parquet
echo -e "\nMetadata single labels parquet structure:"
python parquettest.py /app/data/metadata_single_labels.parquet
echo -e "\nMetadata snow single labels parquet structure:"
python parquettest.py /app/data/metadata_snow_single_labels.parquet

# cleanup
echo -e "\nThe single label parquet files have been created and can now be found in your mounted volume directory as 'single_labels.parquet', 'metadata_single_labels.parquet' and 'metadata_snow_single_labels.parquet'!"
echo -e "\nUse them just like 'metadata.parquet' and 'metadata_snow.parquet'!"
