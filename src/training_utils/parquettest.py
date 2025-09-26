import pandas as pd
import sys
from pathlib import Path

try:
    optarg = sys.argv[1]
    if optarg == '--full':
        parquet_path = Path(sys.argv[2]).resolve()
    elif optarg == '--unique':
        parquet_path = Path(sys.argv[2]).resolve()
    else: 
        parquet_path = Path(sys.argv[1]).resolve()
    assert parquet_path.is_file(), f"Input file not found or not a file: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python parquettest.py (--full) /path/to/parquetfile.parquet\n')

if optarg == '--full':
    df = pd.read_parquet(parquet_path)
    print(df.to_string())
    exit()
elif optarg == '--unique':
    df = pd.read_parquet(parquet_path,columns=['patch_id','labels'])
    df = df.drop_duplicates(subset='patch_id', keep=False)
else:
    df = pd.read_parquet(parquet_path,columns=['patch_id','labels'])

print(df)
