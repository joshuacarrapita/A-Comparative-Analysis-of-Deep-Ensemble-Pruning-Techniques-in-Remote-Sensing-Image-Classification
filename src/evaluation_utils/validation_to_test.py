import argparse
import csv
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Translate validation results.pt model indices to test results.pt model indices based on file paths.")
    parser.add_argument("--val_csv", type=Path, required=True, help="Path to the validation CSV.")
    parser.add_argument("--test_csv", type=Path, required=True, help="Path to the test CSV.")
    parser.add_argument("--out_json", type=Path, default="index_mapping.json", help="Path for the output JSON.")
    args = parser.parse_args()

    # load path to index mapping as dicts
    with open(args.val_csv, 'r') as f:
        val_map = {row['Path']: int(row['Index']) for row in csv.DictReader(f)}
        
    with open(args.test_csv, 'r') as f:
        test_map = {row['Path']: int(row['Index']) for row in csv.DictReader(f)}

    # translate val index to test index based on path
    index_mapping = {}
    for path, val_idx in val_map.items():
        if path in test_map:
            index_mapping[val_idx] = test_map[path]
        else:
            print(f"[WARNING] Path in validation results missing in test results: {path}")

    # save translation
    with open(args.out_json, 'w') as f:
        json.dump(index_mapping, f, indent=4)

    print(f"[INFO] Successfully translated {len(index_mapping)} indices and saved the result to {args.out_json}")

if __name__ == "__main__":
    main()
