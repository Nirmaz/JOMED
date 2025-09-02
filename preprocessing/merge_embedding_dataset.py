#!/usr/bin/env python3
"""
Simple JSONL File Merger

Combines three JSONL files (MIMIC-CXR, ROCO, and PMC-OA) into a single output file.
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Merge three JSONL files into one.")

    # Required arguments
    parser.add_argument("--mimic_file", required=True, help="Path to MIMIC-CXR JSONL file")
    parser.add_argument("--roco_file", required=True, help="Path to ROCO JSONL file")
    parser.add_argument("--pmc_file", required=True, help="Path to PMC-OA JSONL file")

    # Optional arguments
    parser.add_argument("--output_file", default="merged_datasets.jsonl",
                        help="Output file path (default: merged_datasets.jsonl)")

    return parser.parse_args()


def merge_jsonl_files(input_files, output_file):
    """Merge multiple JSONL files into a single file, adding dataset source."""

    dataset_names = ["mimic", "roco", "pmc"]  # Names for each dataset

    with open(output_file, 'w') as outfile:
        for idx, input_file in enumerate(input_files):
            dataset_name = dataset_names[idx]
            print(f"Processing {dataset_name} dataset: {input_file}")

            with open(input_file, 'r') as infile:
                for line in infile:
                    if line.strip():  # Skip empty lines
                        try:
                            entry = json.loads(line.strip())
                            # Add dataset source information
                            entry['dataset_source'] = dataset_name
                            # Write to output file
                            outfile.write(json.dumps(entry) + '\n')
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse line in {input_file}")
                            continue

    print(f"Merged file created at: {output_file}")


def main():
    args = parse_args()

    # Check if input files exist
    for file_path in [args.mimic_file, args.roco_file, args.pmc_file]:
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Merge the files
    input_files = [args.mimic_file, args.roco_file, args.pmc_file]
    merge_jsonl_files(input_files, args.output_file)


if __name__ == "__main__":
    main()