from pathlib import Path
from alive_progress import alive_bar
import rasterio
import numpy as np
import pandas as pd
import sys

#-----------------------------------------------------------------------------
# INIT
#-----------------------------------------------------------------------------

if __debug__:
    print ("\n--DEBUG ON--")
else:
    print ("\n--DEBUG OFF--")

# utility so dataframe print shows the whole patch_id 
pd.set_option('display.max_colwidth', None)

try:
    assert len(sys.argv) == 5, 'Must provide arguments according to usage'
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
    method = str(sys.argv[3])
    METHODS = ['main-label-threshold', 'top-vs-sec']
    assert method in METHODS, f'Method must be one of {str(METHODS)}'
    threshold = float(sys.argv[4])
    assert threshold >= 0 and threshold <=1, f'Threshold must be between 0 and 1'
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python multi_to_single_label.py /path/to/reference_maps_dir/ /path/to/outfile.parquet [main-label-threshold, top-vs-sec] threshold\n')

print('\nIndexing reference maps, please be patient...')

# obtain the paths to all reference maps
with alive_bar() as bar:
    image_patch_file_paths = list(datapath['reference_maps_root_dir_path'].rglob('*.tif'))

print(f'\nFinished indexing reference maps. {len(image_patch_file_paths)} reference maps found.')

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
#-----------------------------------------------------------------------------

# vectorize the dictionary mapping function
clc_lvl_3_to_19_class_label_dictionary_vectorized_get = np.vectorize(clc_lvl_3_to_19_class_label_dictionary.get)
nineteen_class_label_dictionary_vectorized_get = np.vectorize(nineteen_class_label_dictionary.get)

print('\nExtracting single labels, please be patient...')

# initizalize the counting variables
if __debug__:
    max_rounds = 10
else:
    max_rounds = len(image_patch_file_paths)
round = 1
skipped_unlabeled = 0
skipped_share = 0
one_label = 0

np.random.seed(0) # make run predictable in case e.g. argmax needs to randomly choose between same-rank candidates

# main loop
with alive_bar(max_rounds) as bar:

    for file_path in image_patch_file_paths:

        with rasterio.open(file_path) as src:

            # --- prepare data ---
            # update progress bar
            bar()

            # read tif image
            data = src.read(1)

            # convert the data from clc_lvl_3 to 19_class_labels according to mapping
            mapped_data = clc_lvl_3_to_19_class_label_dictionary_vectorized_get(data)

            # get unique labels with counts
            unique_labels, label_counts = np.unique(mapped_data, return_counts=True)

            # make sure the data mapping works
            assert None not in unique_labels, "Dict data mapping returned None."

            # convert the unique labels from 19_class_label numbers to 19_class_label strings according to mapping
            mapped_unique_labels = nineteen_class_label_dictionary_vectorized_get(unique_labels)

            # get total count of labelled pixels
            total_label_count = np.sum(label_counts)

            # find out what percentage of the image each label covers
            label_percentages = label_counts / total_label_count

            def round_debug():
                if __debug__:
                    print(f'\n\nProcessing file: {file_path}')
                    print(f'Unique labels: {mapped_unique_labels}')
                    print(f'Label counts: {label_counts}')
                    print(f'Label percentages: {label_percentages}')
                    print(f'Total label count: {total_label_count}')

            # --- filter unfitting patches ---
            # get the index of the count for 'Unlabeled' if it exists
            unlabeled_index, = np.where(mapped_unique_labels == 'Unlabeled')

            # make sure that no patches with more than 25% 'Unlabeled' pixels are included
            if unlabeled_index.size > 0:
                if unlabeled_index.size == 1: 
                    unlabeled_percent = label_percentages[unlabeled_index[0]]
                    if unlabeled_percent >= 0.25:
                        round_debug()
                        print(f'Skipping round {round}: {unlabeled_percent*100:.2f}% of patch pixels are unlabeled.')
                        round += 1
                        skipped_unlabeled += 1
                        if round > max_rounds:
                            break
                        else:
                            continue
                else:
                    print(f'\nFound multiple indices for `Unlabeled` in unique label list for patch {file_path} in round {round}. Aborting process.')
                    print(f'\nUnique labels: {unique_labels}')
                    print(f'\nMapped unique labels: {mapped_unique_labels}')
                    print(f'\nUnlabeled indices: {unlabeled_index}')
                    break

            if method == "main-label-threshold":
                # get index, text and pixel-share of most common label
                max_count_index = np.argmax(label_counts)
                max_count_label = mapped_unique_labels[max_count_index]
                max_count_share = label_percentages[max_count_index]

                # make sure that found majority label has more than 50% share or so of image pixels cause otherwise it might impact learning performance
                if max_count_share <= threshold:
                    round_debug()
                    print(f'Skipping round {round}: Majority label share is only {max_count_share*100:.2f}% of patch pixels.')
                    round += 1
                    skipped_share += 1
                    if round > max_rounds:
                        break
                    else:
                        continue
            else:
                # get index, text and pixel-share of most common label
                top_two_indices = (-label_counts).argsort()[:2] # -array for descending order, argsort returns indices that would sort the array in ascending order
                top_two_labels = mapped_unique_labels[top_two_indices]
                max_count_label = top_two_labels[0]
                top_two_shares = label_percentages[top_two_indices]
                if top_two_shares.size > 1:
                    top_two_share_difference = top_two_shares[0] - top_two_shares[1]
                else:
                    top_two_share_difference = 1
                    one_label += 1

                # make sure that found majority label has more than 15% more share of image pixels than second majority label cause otherwise it might impact learning performance
                if  top_two_share_difference < threshold:
                    round_debug()
                    print(f'Skipping round {round}: Majority label share of patch pixels is only {top_two_share_difference*100:.2f}% larger than second majority label share of patch pixels.')
                    round += 1
                    skipped_share += 1
                    if round > max_rounds:
                        break
                    else:
                        continue


            # --- save fitting patches ---
            # add patch_name to list
            patch_names.append(file_path.stem.removesuffix('_reference_map'))

            # add this label to list
            patch_labels.append([max_count_label])

            # check if we're done
            round_debug()
            round += 1
            if round > max_rounds:
                break

# print run info
round -= 1
skipped = skipped_unlabeled + skipped_share
converted = round-skipped
print(f'\nFinished converting multi- to single labels.')
if method == 'main-label-threshold':
    print(f'\n{skipped} patches skipped in total. Skipped {skipped_unlabeled} patches with >25% "Unlabeled" pixels and {skipped_share} patches with <{threshold * 100}% majority label share of pixels. {one_label} patches only had one label.')
else:
    print(f'\n{skipped} patches skipped in total. Skipped {skipped_unlabeled} patches with >25% "Unlabeled" pixels and {skipped_share} patches with majority label share of pixels <{threshold * 100}% apart from second most occuring label. {one_label} patches only had one label.')
print(f'\nConverted labels for {converted}/{round} ({converted*100/round:.2f}% of) patches.')

# create a pandas dataframe based on the patch_id and labels list
df = pd.DataFrame(list(zip(patch_names,patch_labels)), columns=['patch_id', 'labels'])

# create and save a parquet file out of the dataframe
df.to_parquet(datapath['outfile'], compression='gzip')

# print output info
print(f'\n\nFinal Single-Labels Dataframe: \n{df}')
if __debug__:
    print(f'\n\nFinal Single-Labels Dataframe (Correctness-Check): \n{pd.read_parquet(datapath["outfile"])}')
