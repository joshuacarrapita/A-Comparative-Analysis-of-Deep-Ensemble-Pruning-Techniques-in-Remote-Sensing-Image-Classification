import pandas as pd
import sys
from pathlib import Path

pd.set_option('display.max_colwidth', None)

try:
    optarg = sys.argv[1]
    if optarg == '--full':
        parquet_path = Path(sys.argv[2]).resolve()
        optarg = True
    else:
        parquet_path = Path(optarg).resolve()
        optarg = False
    assert parquet_path.is_file(), f"Input file not found or not a file: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python parquettest.py (--full) /path/to/parquetfile.parquet\n')

if optarg:
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.expand_frame_repr', False)  
    df = pd.read_parquet(parquet_path)
else:
    df = pd.read_parquet(parquet_path,columns=['patch_id','labels'])

print(df)
