from pathlib import Path
import rasterio
import numpy as np
import pandas as pd

# INIT
# TODO: Adjust path for docker container
# Utility so dataframe print shows the whole patch_id 
pd.set_option('display.max_colwidth', None)

# Set entrypoint for recursive path search
entrypoint = Path('/home') / 'user' / 'Documents' / 'backups' / 'datasets' / 'Reference_Maps' 

# Obtain the paths to all reference maps
image_patch_file_paths = list(entrypoint.rglob('*.tif'))

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
    324: 'Transitional woodland-shrub',
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
    14: 'Transitional woodland-shrub',
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

# EXEC
for file_path in image_patch_file_paths:
   
    # add patch_name to list
    patch_names.append(file_path.stem.removesuffix('_reference_map'))

    with rasterio.open(file_path) as src:

        # read tif image
        data = src.read(1)

        # vectorize the dictionary mapping function
        vectorized_dict = np.vectorize(clc_lvl_3_to_19_class_label_dictionary.get)

        # convert the data according to mapping
        mapped_data = vectorized_dict(data)

        # get unique labels with counts
        unique_labels, label_counts = np.unique(mapped_data, return_counts=True)

        # get index, then number and then text of most common label
        max_count_index = np.argmax(label_counts)
        max_count_label_no = unique_labels[max_count_index]
        max_count_label = nineteen_class_label_dictionary[max_count_label_no]

        # add this label to list
        patch_labels.append([max_count_label])

# create a pandas dataframe based on the patch_id and labels list
df = pd.DataFrame(list(zip(patch_names,patch_labels)), columns=['patch_id', 'labels'])

# TODO: Replace the patch_id and label columns of the metadata_parquet / dataframe with updated single labels
