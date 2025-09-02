import json
import os
import time
import argparse
def create_jsonl_file_from_dict(list_dict, path_json):
    # path_json_save = os.path.join(path_json, 'mimic.jsonl.all')
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
        description='Process ROCO dataset and create JSONL files for medical image data and reports.'
    )

    # Required arguments
    parser.add_argument(
        '--path_roco_dataset',
        type=str,
        required=True,
        help='Base path of dataset'
    )

    parser.add_argument(
        '--path_save',
        type=str,
        required=True,
        help='Output path to save generated JSONL files'
    )






    return parser.parse_args()

if __name__ == '__main__':
    organize_for_mebeddings = False
    args = parse_args()
    ### Training ################
    path_to_roco_images = os.path.join(args.path_roco_dataset, "data/train/radiology/images")
    path_to_roco_captions = os.path.join(args.path_roco_dataset, "data/train/radiology/captions.txt")
    captions_list_train = phrase_roco_captions_text(path_to_roco_captions)
    start_time = time.time()
    print("start", start_time)
    cap_list_train_final = []
    for cap_l in captions_list_train:
        img_path = os.path.join(path_to_roco_images, cap_l['img_path']  + '.jpg')
        cap_l['img_path'] = img_path
        if os.path.exists(img_path):
            cap_list_train_final.append(cap_l)

    end_time = time.time()
    print(end_time - start_time, "run time load list")
    print(captions_list_train)

    ### val ################

    path_to_roco_images = os.path.join(args.path_roco_dataset, "data/validation/radiology/images")
    path_to_roco_captions = os.path.join(args.path_roco_dataset, "data/validation/radiology/captions.txt")
    captions_list_val = phrase_roco_captions_text(path_to_roco_captions)
    start_time = time.time()
    print("start", start_time)
    cap_list_val_final = []
    for cap_l in captions_list_val:
        img_path = os.path.join(path_to_roco_images, cap_l['img_path'] + '.jpg')
        cap_l['img_path'] = img_path
        if os.path.exists(img_path):
            cap_list_val_final.append(cap_l)

    end_time = time.time()
    print(end_time - start_time, "run time load list")
    print(captions_list_train)
    print(captions_list_val)


    ### test ################

    path_to_roco_images = os.path.join(args.path_roco_dataset, "data/test/radiology/images")
    path_to_roco_captions = os.path.join(args.path_roco_dataset, "data/test/radiology/captions.txt")
    captions_list_test = phrase_roco_captions_text(path_to_roco_captions)

    cap_list_test_final = []
    for cap_l in captions_list_test:
        cap_l['img_path'] = os.path.join(path_to_roco_images, cap_l['img_path'] + '.jpg')
        if os.path.exists(cap_l['img_path']):
            cap_list_test_final.append(cap_l)


    print(cap_list_test_final)


    final_caption_list = cap_list_train_final + cap_list_val_final + cap_list_test_final

    counter = 0
    for cap_l in final_caption_list:
        cap_l["id"] = counter
        counter = counter + 1



    # path_save = "/cs/labs/tomhope/nirm/atlas_data_dir/data/corpora/vqa_rad/roco_tr_val_test.jsonl"
    create_jsonl_file_from_dict(final_caption_list, args.path_save)






