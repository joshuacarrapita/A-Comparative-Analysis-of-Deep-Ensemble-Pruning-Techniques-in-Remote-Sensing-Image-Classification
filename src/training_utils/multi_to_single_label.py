from pathlib import Path
from alive_progress import alive_bar
import rasterio
import numpy as np
import pandas as pd
import sys

#-----------------------------------------------------------------------------

# INIT
if __debug__:
    print ("\n--DEBUG ON--")
else:
    print ("\n--DEBUG OFF--")

# Utility so dataframe print shows the whole patch_id 
pd.set_option('display.max_colwidth', None)

try:
    datapath = {
        'reference_maps_root_dir_path': Path(sys.argv[1]).resolve(),
        'outfile': Path(sys.argv[2]).resolve(),
    }
    for key, path in datapath.items():
        if key != 'outfile':
            assert path.is_dir(), f"Input directory does not exist or is not a directory: {path}"
        else: 
            if path.is_dir():
                sys.exit(f'\nNo output filename provided. Exiting.')
            else:
                assert path.parent.is_dir(), f"Output directory does not exist or is not a directory: {path.parent}"
                assert path.name, f"No output filename provided: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python multi_to_single_label.py /path/to/reference_maps_dir/ /path/to/outfile.parquet\n')

print('\nIndexing reference maps, please be patient...')

# Obtain the paths to all reference maps
with alive_bar() as bar:
    image_patch_file_paths = list(datapath['reference_maps_root_dir_path'].rglob('*.tif'))

print('\nFinished indexing reference maps.')

patch_names = []
patch_labels = []

clc_lvl_3_label_dictionary = {
 	111: 'Continuous urban fabric',
    112: 'Discontinuous urban fabric',
    121: 'Industrial or commercial units',
    122: 'Road and rail networks and associated land',
    123: 'Port areas',
    124: 'Airports',
    131: 'Mineral extraction sites',
    132: 'Dump sites',
    133: 'Construction sites',
    141: 'Green urban areas',
    142: 'Sport and leisure facilities',
    211: 'Non-irrigated arable land',
    212: 'Permanently irrigated land',
    213: 'Rice fields',
    221: 'Vineyards',
    222: 'Fruit trees and berry plantations',
    223: 'Olive groves',
    231: 'Pastures',
    241: 'Annual crops associated with permanent crops',
    242: 'Complex cultivation patterns',
    243: 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    244: 'Agro-forestry areas',
    311: 'Broad-leaved forest',
    312: 'Coniferous forest',
    313: 'Mixed forest',
    321: 'Natural grasslands',
    322: 'Moors and heathland',
    323: 'Sclerophyllous vegetation',
    324: 'Transitional woodland, shrub',
    331: 'Beaches, dunes, sands',
    332: 'Bare rocks',
    333: 'Sparsely vegetated areas',
    334: 'Burnt areas',
    335: 'Glaciers and perpetual snow',
    411: 'Inland marshes',
    412: 'Peat bogs',
    421: 'Salt marshes',
    422: 'Salines',
    423: 'Intertidal flats',
    511: 'Water courses',
    512: 'Water bodies',
    521: 'Coastal lagoons',
    522: 'Estuaries',
    523: 'Sea and ocean',
    999: 'Unlabeled',
}

nineteen_class_label_dictionary = {
    0: 'Urban fabric',
    1: 'Industrial or commercial units',
    2: 'Unlabeled',
    3: 'Arable land',
    4: 'Permanent crops',
    5: 'Pastures',
    6: 'Complex cultivation patterns',
    7: 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    8: 'Agro-forestry areas',
    9: 'Broad-leaved forest',
    10: 'Coniferous forest',
    11: 'Mixed forest', 
    12: 'Natural grassland and sparsely vegetated areas',
    13: 'Moors, heathland and sclerophyllous vegetation',
    14: 'Transitional woodland, shrub',
    15: 'Beaches, dunes, sands',
    16: 'Inland wetlands',
    17: 'Coastal wetlands',
    18: 'Inland waters',
    19: 'Marine waters',
}

clc_lvl_3_to_19_class_label_dictionary = {
    111: 0,
    112: 0,
    121: 1,
    122: 2,
    123: 2,
    124: 2,
    131: 2,
    132: 2,
    133: 2,
    141: 2,
    142: 2,
    211: 3,
    212: 3,
    213: 3,
    221: 4,
    222: 4,
    223: 4,
    231: 5,
    241: 4,
    242: 6,
    243: 7,
    244: 8,
    311: 9,
    312: 10,
    313: 11,
    321: 12,
    322: 13,
    323: 13,
    324: 14,
    331: 15,
    332: 2,
    333: 12,
    334: 2,
    335: 2,
    411: 16,
    412: 16,
    421: 17,
    422: 17,
    423: 2,
    511: 18,
    512: 18,
    521: 19, 
    522: 19, 
    523: 19, 
    999: 2,
}

#-----------------------------------------------------------------------------

# EXEC

# vectorize the dictionary mapping function
vectorized_dict_get = np.vectorize(clc_lvl_3_to_19_class_label_dictionary.get)

print('\nExtracting single labels, please be patient...')

if __debug__:
    rounds = 10
else:
    rounds = len(image_patch_file_paths)

with alive_bar(rounds) as bar:
    for file_path in image_patch_file_paths:

        with rasterio.open(file_path) as src:

            # update progress bar
            bar()

            # read tif image
            data = src.read(1)

            # convert the data according to mapping
            mapped_data = vectorized_dict_get(data)

            # get unique labels with counts
            unique_labels, label_counts = np.unique(mapped_data, return_counts=True)
            # TODO: check if total_labels is always the same as the pixels are always the same

            # Make sure the data mapping works
            assert None not in unique_labels, "Dict data mapping returned None."
            # Find out what percentage of the image each label covers
            label_percentages = label_counts * 100 / np.sum(label_counts)

            if __debug__:
                print(unique_labels)
                print(label_counts)
                print(np.sum(label_counts))
                print(label_percentages)

            # Get the index of the count for 'Unlabeled' if it exists
            unlabeled_index, = np.where(unique_labels == 2)

            # Make sure that no patches with more than 25% 'Unlabeled' pixels are included
            if unlabeled_index.size > 0 and label_percentages[unlabeled_index] >= 25:
                continue

            # Get index, then number and then text of most common label
            max_count_index = np.argmax(label_counts)

            # Make sure that found majority label has more than 50% share or so of image pixels cause otherwise it might impact learning performance
            if label_percentages[max_count_index] <= 50:
                continue

            # Check that we skip images where the majority class is unlabeled (afaik after the first two checks, this should never occur and this part of the code is unecessary
            max_count_label_no = unique_labels[max_count_index]
            max_count_label = nineteen_class_label_dictionary[max_count_label_no]
            if max_count_label == 'Unlabeled':
                continue

            # add patch_name to list
            patch_names.append(file_path.stem.removesuffix('_reference_map'))

            # add this label to list
            patch_labels.append([max_count_label])
        
        if __debug__:
            if rounds > 0:
                rounds -= 1
            else:
                break

    # create a pandas dataframe based on the patch_id and labels list
    df = pd.DataFrame(list(zip(patch_names,patch_labels)), columns=['patch_id', 'labels'])

    # create and save a parquet file out of the dataframe
    df.to_parquet(datapath['outfile'], compression='gzip')

    if __debug__:
        print(df)
        print(pd.read_parquet(datapath['outfile']))
