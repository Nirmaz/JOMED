import json
import os
import argparse
from src.util import load_jsonl_file
def create_jsonl_file_from_dict(list_dict, path_json):

    with open(path_json, 'w') as f:
        for entry in list_dict:
            json.dump(entry, f)
            f.write('\n')


def phrase_roco_captions_text(path):
    # Read the content of the file
    with open(path, 'r') as file:
        lines = file.readlines()

    # Process each line to create a list of dictionaries
    captions_list = []
    for line in lines:
        parts = line.strip().split('\t')  # Split each line by tab
        if len(parts) == 2:
            img_id, description = parts
            captions_list.append({'img_path': img_id, 'text': description})

    return captions_list



def parse_args():
    """Parse command line arguments for MIMIC-CXR data processing script."""
    parser = argparse.ArgumentParser(
        description='Process PMC-OA dataset and create JSONL files for medical image data and reports.'
    )

    # Required arguments
    parser.add_argument(
        '--home_pmc_oa_project',
        type=str,
        required=True,
        help='Base path to the Project'
    )

    parser.add_argument(
        '--path_save',
        type=str,
        required=True,
        help='Output path to save generated JSONL files'
    )

    # Optional arguments with defaults
    parser.add_argument(
        '--home_pmc_oa_images',
        type=str,
        help='Path to the location of the images'
    )





    return parser.parse_args()


if __name__ == '__main__':
    organize_for_mebeddings = False
    args = parse_args()
    ### Training ################
    path_to_dict_pmc = os.path.join(args.home_pmc_oa_project, "pmc_oa.jsonl?download=true")
    image_dict = load_jsonl_file(path_to_dict_pmc)

    # Create an empty dictionary to store the updated image paths
    updated_image_dict = []
    counter = 0
    # Walk through the home folder and its subfolders
    for i in range(len(image_dict)):
        if counter % 10000 == 0 and i != 0:
            # break
            print("counter", counter)
        if os.path.exists(os.path.join(args.home_pmc_oa_images, image_dict[i]['image'])):
            counter = counter + 1
            case_dict = {'img_path': os.path.join(args.home_pmc_oa_images, image_dict[i]['image']), 'text': image_dict[i]['caption'], "type":image_dict[i]['alignment_type'], "score":image_dict[i]['alignment_score'],"id": i}
            updated_image_dict.append(case_dict)

    create_jsonl_file_from_dict(updated_image_dict, args.path_save)








