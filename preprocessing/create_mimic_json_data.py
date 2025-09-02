import json
import pandas as pd
import os
import time
import random
import torch
import numpy as np
from joblib import Parallel, delayed
import functools
import argparse
LABELS = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Lesion',
        'Lung Opacity',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices']


def parse_args():
    """Parse command line arguments for MIMIC-CXR data processing script."""
    parser = argparse.ArgumentParser(
        description='Process MIMIC-CXR dataset and create JSONL files for medical image data and reports.'
    )

    # Required arguments
    parser.add_argument(
        '--path_folder_location',
        type=str,
        required=True,
        help='Base path to the MIMIC-CXR dataset folder'
    )

    parser.add_argument(
        '--path_save',
        type=str,
        required=True,
        help='Output path to save generated JSONL files'
    )

    # Optional arguments with defaults
    parser.add_argument(
        '--path_data_excel',
        type=str,
        default='./',
        help='Path to save intermediate Excel data (default: current directory)'
    )


    return parser.parse_args()


def create_jsonl_file_from_dict(list_dict, path_json):

    with open(path_json, 'w') as f:
        for entry in list_dict:
            json.dump(entry, f)
            f.write('\n')

def read_patient_report_line(path_to_report):
    with open(path_to_report, 'r') as file:
        report_string = ''.join(line.strip() for line in file.readlines())
    return report_string

def loop_mimic_cxr(i, index, row, list_dicts_pne, list_dicts_no_finding, list_dicts_other_diseases, start_time, df_meta_data, position_to_id, partial, disease_classes, only_pne, save_full_report, split):
    if i % 10000 == 0:
            end_time = time.process_time()
            running_time = end_time - start_time

            print(f"we are currently at index {i}, current time {running_time}")
    item_dict = {}
    row_dict = row.to_dict()
    if only_pne:
        if row_dict['No Finding'] != 1 and row_dict['Pneumonia'] != 1:
            return

    if partial and i > 1000:
        return

    dicom_id = os.path.basename(row_dict['path']).split('.dcm')[0]

    position = df_meta_data.loc[df_meta_data['dicom_id'] == dicom_id, 'ViewPosition'].values[0]
    if not position == 'PA' and not position == 'AP':
        return

    text = read_patient_report_line(row_dict['path_report'])

    index_s = text.find('FINDINGS:')
    index_e = text.find('IMPRESSION:')
    text = text[index_s + len('FINDINGS:'):index_e]

    study_id = row_dict['study_id']
    if position in position_to_id.keys():
        index_position = position_to_id[str(position)]
    else:
        index_position = str(random.randint(1, 100))
    id = int(str(index_position) + str(study_id) + str(random.randint(1, 100)))

    item_dict['id'] = id
    if save_full_report:
        item_dict['text'] = text
    else:
        item_dict['text'] = "text"

    item_dict['img_path'] = row_dict['path'].replace('.dcm', '.jpg')
    item_dict['title'] = ''

    item_dict['type_split'] = split.loc[split['dicom_id'] == dicom_id, 'split'].iloc[0]

    for ddd in disease_classes:
        item_dict[ddd] = row_dict[ddd]

    if row_dict['Pneumonia'] == 1:
        list_dicts_pne.append(item_dict)
    elif row_dict['No Finding'] == 1:
        list_dicts_no_finding.append(item_dict)
    else:
        list_dicts_other_diseases.append(item_dict)





def create_mimic_list_of_dicts_pne_no_finding_diseases(path_to_excel, position_to_id, metadata_csv, disease_classes, only_pne=False, partial = False, run_in_parallel = False, save_full_report = False, split = None):
    df = pd.read_csv(path_to_excel)
    df_meta_data = pd.read_csv(metadata_csv)
    list_dicts_pne = []
    list_dicts_no_finding = []
    list_dicts_other_diseases = []
    random.seed(0)
    print("start iterate mimic data")
    start_time = time.process_time()

    num_cores = 10

    if not run_in_parallel:
        for i, (index, row) in enumerate(df.iterrows()):
            start_time = time.process_time()
            loop_mimic_cxr(i, index, row, list_dicts_pne, list_dicts_no_finding, list_dicts_other_diseases, start_time, df_meta_data, position_to_id, partial, disease_classes, only_pne, save_full_report, split)
    else:
        partial_function = functools.partial(loop_mimic_cxr, list_dicts_pne = list_dicts_pne, list_dicts_no_finding = list_dicts_no_finding, list_dicts_other_diseases= list_dicts_other_diseases, start_time=start_time, df_meta_data = df_meta_data,position_to_id = position_to_id, partial = partial, disease_classes = disease_classes, only_pne = only_pne, save_full_report = save_full_report, split = split)
        Parallel(n_jobs=num_cores)(delayed(partial_function)(i = i, index = index, row = row) for i, (index, row) in enumerate(df.iterrows()))

    end_time = time.process_time()
    running_time = end_time - start_time
    print(f" end loading data for jsonl {running_time}")

    return list_dicts_pne, list_dicts_no_finding, list_dicts_other_diseases



def create_mimic_list_of_dicts(path_to_excel, position_to_id, metadata_csv, disease_classes, only_pne = False, partial = False):
    df = pd.read_csv(path_to_excel)
    df_meta_data = pd.read_csv(metadata_csv)
    list_dicts = []
    random.seed(0)
    print("start iterate mimic data")
    start_time = time.process_time()
    for i, (index, row) in enumerate(df.iterrows()):
        # start_time = time.process_time()
        if i % 10000 == 0:
            end_time = time.process_time()
            running_time = end_time - start_time
            
            print(f"we are currently at index {i}, current time {running_time}")
        item_dict = {}
        row_dict = row.to_dict()
        if  only_pne:
            if row_dict['No Finding'] != 1 and row_dict['Pneumonia'] != 1:
                continue
        text = read_patient_report_line(row_dict['path_report'])
        study_id = row_dict['study_id']
        dicom_id = os.path.basename(row_dict['path']).split('.dcm')[0]
        position = df_meta_data.loc[df_meta_data['dicom_id'] == dicom_id, 'ViewPosition'].values[0]
        if position in position_to_id.keys():
            index_position = position_to_id[str(position)]
        else:
            index_position = str(random.randint(1, 100))
        id = int(str(index_position) + str(study_id) + str(random.randint(1, 100)))

        item_dict['id'] = id
        item_dict['text'] = text
        item_dict['img_path'] = row_dict['path'].replace('.dcm', '.jpg')
        item_dict['title'] = ''

        for ddd in disease_classes:
            item_dict[ddd] = row_dict[ddd]

        list_dicts.append(item_dict)

        if partial and i > 1000:
            break

    end_time = time.process_time()
    running_time = end_time - start_time
    print(f" end loading data for jsonl {running_time}")
    return list_dicts

def tensor_size_in_bytes(tensor):
    element_size = tensor.element_size()  # size in bytes for one element
    num_elements = tensor.numel()  # number of elements in the tensor
    return element_size * num_elements


def main(path_json, path_to_excel, path_to_split, metadata_csv):


    df_split = pd.read_csv(path_to_split)
    position_to_id = {'PA': 1, 'AP': 2, 'LATERAL': 3, 'LL': 4, 'nan': 5}


    save_full_report = True
    partial = False
    run_in_parallel = False
    take_partial_lists = False

    list_dicts_pne, list_dicts_no_finding, list_dicts_other_diseases = create_mimic_list_of_dicts_pne_no_finding_diseases(path_to_excel, position_to_id, metadata_csv, LABELS,
                                                       only_pne=False, partial = partial, run_in_parallel= run_in_parallel, save_full_report = save_full_report, split=df_split)


    if take_partial_lists:
        last_ex_no_finding = np.minimum([len(list_dicts_no_finding)], [3000])[0]
        last_ex_other_disease = np.minimum([len(list_dicts_other_diseases)], [3000])[0]
    else:
        last_ex_no_finding = len(list_dicts_no_finding)
        last_ex_other_disease = len(list_dicts_other_diseases)

    list_dicts = list_dicts_pne + list_dicts_no_finding[:last_ex_no_finding] + list_dicts_other_diseases[:last_ex_other_disease]
    # path_json = '/cs/labs/tomhope/nirm/atlas_data_dir/data/corpora/mimic_cxr'
    os.makedirs(path_json, exist_ok=True)

    if not save_full_report:
        path_save_json = os.path.join(path_json, f'mimic_cxr_pne_nofind_other_{len(list_dicts)}_pne_{len(list_dicts_pne)}.jsonl')
    else:
        path_save_json = os.path.join(path_json,
                                      f'mimic_cxr_pne_nofind_other_{len(list_dicts)}_pne_{len(list_dicts_pne)}_full_report.jsonl')

    if partial:
        path_save_json = path_save_json.split('.json')[0] + '_part_' + ".json"

    # if run_in_parallel:
    #     path_save_json = path_save_json.split('.json')[0] + "parallel" + ".json"

    print(f"save path {path_save_json}")
    if partial:
        create_jsonl_file_from_dict(list_dicts, path_save_json)
    else:
        create_jsonl_file_from_dict(list_dicts, path_save_json)

    fname = path_save_json
    for line in open(fname):
        if line.strip() != "":
            item = json.loads(line)
            assert "id" in item
        else:
            print("empty line")

def create_run_dataframe( path_save, path_image, split_file, cxr_record_file, labels_file,split_type):

    data_frame = {'path': [], 'subject_id':[], 'study_id': []}

    for ill in LABELS:
        data_frame[ill] = []

    df_split = pd.read_csv(split_file)
    df_paths = pd.read_csv(cxr_record_file)
    df_label_file = pd.read_csv(labels_file)

    #df_split_set = df_split[df_split.iloc[:, 3] == split_type]
    dicom_id_list_in_set = df_split.iloc[:, 0].tolist()
    df_paths_in_set = df_paths[df_paths.iloc[:, 2].isin(dicom_id_list_in_set)]
    df_paths_in_set_copy = df_paths_in_set.copy()

    # edit path to jpg images
    mask = df_paths_in_set.iloc[:, 3].str.endswith('.dcm')
    selected_rows = df_paths_in_set[mask]
    selected_rows.iloc[:, 3] = selected_rows.iloc[:, 3].str.replace('.dcm', '.jpg')
    df_paths_in_set.update(selected_rows)
    df_paths_in_set.iloc[:, 3] = path_image +'/'+ df_paths_in_set.iloc[:, 3].astype(str)
    list_path_images = df_paths_in_set.iloc[:, 3].tolist()
    list_subject_id = df_paths_in_set.iloc[:, 0].tolist()
    list_study_id = df_paths_in_set.iloc[:, 1].tolist()
    count = 0
    for kk, val in enumerate(zip(list_path_images, list_subject_id, list_study_id)):




        image_path, subject_id, study_id = val
        # print('index: ', kk)
        if len(df_label_file[(df_label_file['subject_id'] == subject_id) & (df_label_file['study_id'] == study_id)].index) == 0:
            print('No match')
            count = count + 1
            continue

        data_frame['path'].append(image_path)
        data_frame['subject_id'].append(subject_id)
        data_frame['study_id'].append(study_id)


        index = \
        df_label_file[(df_label_file['subject_id'] == subject_id) & (df_label_file['study_id'] == study_id)].index[
            0]
        for i in range(0, len(LABELS)):

            if (df_label_file[LABELS[i].strip()].iloc[index].astype('float') > 0):
                data_frame[LABELS[i].strip()].append(1)
            else:
                data_frame[LABELS[i].strip()].append(0)

    df = pd.DataFrame(data_frame)
    path_save = os.path.join(path_save, "new_all.csv")
    df.to_csv(path_save)
    print(f'number of not matching {count}')
    return path_save

if __name__ == '__main__':
    args = parse_args()

    path_main_dir = os.path.join(args.path_folder_location,"mimic-cxr-jpg-2.0.0.physionet.org" )
    split_path = os.path.join(path_main_dir, "mimic-cxr-2.0.0-split.csv")
    record_path =  os.path.join(path_main_dir, "cxr-record-list.csv")
    chest_expert_path =  os.path.join(path_main_dir, "mimic-cxr-2.0.0-chexpert.csv")
    metadata_csv_path = os.path.join(path_main_dir, "mimic-cxr-2.0.0-metadata.csv")

    path_excel_data = create_run_dataframe(args.path_data_excel, path_main_dir, split_path, record_path, chest_expert_path, "all")
    main(args.path_save, path_excel_data, path_to_split=split_path, metadata_csv=metadata_csv_path)


