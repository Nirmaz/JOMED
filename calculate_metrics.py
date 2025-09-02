from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import argparse

def load_jsonl_file(file_path):
    counter = 0
    # for line in open(file_path):
    #     counter = counter + 1
    #     print(counter)

    files = []
    counter = 0
    for line in open(file_path):
        if line.strip() != "":
            item = json.loads(line)
            files.append(item)
        counter = counter + 1
        # if counter > 1000:
        #     break
    return files





import re
def extract_numbers(text):
    """
    Extract all numbers from a string and return them as a list of integers.

    Args:
        text (str): Input string containing letters and numbers

    Returns:
        list: List of integers found in the string
    """
    # Find all sequences of digits in the string
    numbers = re.findall(r'\d+', text)

    # Convert string numbers to integers
    return [int(num) for num in numbers]





def create_value_class_binary(output_str, classes):
    flag = True
    for ccc in classes:
        if ccc in output_str:
            flag = False
            if 'ye' in ccc:
                num = 1
            elif 'no' in ccc:
                num = 0
            else:
                num = int(ccc)
    if flag:
        num = 0

    return num


def process_results(pred_files, classes):
    create_value_class = create_value_class_binary
    gt = []
    pred = []

    for i, p_file in enumerate(pred_files):
        gt_exp = p_file['answers'].lower()
        gt.append(create_value_class(gt_exp, classes))
        pred_model = p_file['generation'].lower()
        pred.append(create_value_class(pred_model, classes))

    return gt, pred


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return accuracy, f1, precision, recall





def run_prediction(path_predict, classes):

    prediction_files = load_jsonl_file(path_predict)

    gt, pred = process_results(prediction_files, classes)

    accuracy, f1, precision, recall = calculate_metrics(gt, pred)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics from prediction files')
    parser.add_argument('--prediction_file', required=True, type=str, 
                       help='Path to JSONL file containing predictions')
    parser.add_argument('--classes', nargs='+', required=True,
                       help='List of class labels (e.g., "0" "1" "2" "3" "4")')
    
    args = parser.parse_args()
    
    path_prediction = args.prediction_file
    classes = args.classes





    run_prediction(path_prediction,classes)


    exit()


