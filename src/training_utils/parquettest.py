import pandas as pd
import sys
from pathlib import Path

pd.set_option('display.max_colwidth', None)

try:
    parquet_path = Path(sys.argv[1]).resolve()
    assert parquet_path.is_file(), f"Input file not found or not a file: {path}"
except Exception as e:
    sys.exit(f'\nAn error occured: {e}.\n\nUsage: python parquettest.py /path/to/parquetfile.parquet\n')

df = pd.read_parquet(parquet_path,columns=['patch_id','labels'])

print(df)
