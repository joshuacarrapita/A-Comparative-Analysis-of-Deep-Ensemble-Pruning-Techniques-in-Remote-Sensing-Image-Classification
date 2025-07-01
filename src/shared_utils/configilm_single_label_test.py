from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule

try:
    datapath = {
        "images_lmdb": Path(sys.argv[1]).resolve(),
        "metadata_parquet": Path(sys.argv[2]).resolve(),
        "metadata_snow_cloud_parquet": Path(sys.argv[3]).resolve(),
    }
    for key, path in datapath.items():
        assert path.is_file(), f"Input file not found or not a file: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python rewrite_metadata_parqet.py /path/to/multi_label.parquet /path/to/single_label.parquet /path/to/outfile.parquet\n')

ds = BENv2_DataSet.BENv2DataSet(
    data_dirs=my_data_path, # path set by to dataset
    max_len=1000,
)

img, lbl = ds[0]
img = img # only choose RGB channels
