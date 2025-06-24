from pathlib import Path


entrypoint = Path('home') / 'user' / 'Documents' / 'backups' / 'datasets' / 'Reference_Maps' 

image_patch_file_paths = entrypoint.rglob('*.tif') 

patch_names = []
patch_label = []

for file in image_patch_file_paths:
    # TODO: For each file append the name to patch_names and the generate the
    # label and append to patch_label
