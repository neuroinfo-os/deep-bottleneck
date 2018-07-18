import json
import re
import os

def main():
    current_cohort_number, basic_dictionary, activation_functions_to_evaluate, architectures_to_evaluate = initialize_basic_datastructures()
    folder_string = create_basic_folder(current_cohort_number)
    generate_cohort_of_json_files(current_cohort_number, activation_functions_to_evaluate, folder_string, architectures_to_evaluate, basic_dictionary)



def initialize_basic_datastructures():
    current_cohort_number = 7

    basic_dictionary = {
    "epochs": 8000,
    "batch_size" : 256,
    "architecture" : [10, 7, 5, 4, 3],
    "learning_rate" : 0.0004,
    "calculate_mi_for": "full_dataset",
    "activation_fn" :"relu",
    "model" : "models.feedforward",
    "dataset" : "datasets.harmonics",
    "estimator" : "mi_estimator.upper",
    "discretization_range" : 0.001,
    "callbacks" : [],
    "optimizer" : "adam",
    "n_runs" : 5,
    "regularization" : False
    }

    activation_functions_to_evaluate = ["relu", "tanh"]
    architectures_to_evaluate = [[10, 7, 5, 4, 3],
                                 [10, 9, 7, 5, 3],
                                 [10, 9, 7, 3, 3],
                                 [10, 7, 7, 4, 3],
                                 [10, 7, 5, 4],
                                 [10, 7, 5],
                                 [10],
                                 [1, 1, 1, 1],
                                 []
                                 ]

    return current_cohort_number, basic_dictionary, activation_functions_to_evaluate, architectures_to_evaluate


def create_basic_folder(current_cohort_number):
    folder_string = "cohort_"+str(current_cohort_number)
    os.makedirs(folder_string, exist_ok=True)
    return folder_string


def generate_cohort_of_json_files(current_cohort_number, activation_functions_to_evaluate, folder_string, architectures_to_evaluate, basic_dictionary):
    for activation_fn in activation_functions_to_evaluate:
        current_folder_string = generate_folder_for_activation_functions(activation_fn, folder_string)
        for architecture in architectures_to_evaluate:
            json_string = create_json_file(activation_fn, architecture, basic_dictionary)
            save_json(basic_dictionary, current_cohort_number, current_folder_string, json_string)


def save_json(basic_dictionary, current_cohort_number, current_folder_string, json_string):
    file_name = "cohort_" + str(current_cohort_number) + "_" + basic_dictionary[
        "activation_fn"] + "_architecture_" + "_".join(
        str(current_value) for current_value in basic_dictionary["architecture"]) + ".json"
    with open(os.path.join(current_folder_string, file_name), "w") as outfile:
        outfile.write(json_string)


def create_json_file(activation_fn, architecture, basic_dictionary):
    basic_dictionary["activation_fn"] = activation_fn
    basic_dictionary["architecture"] = architecture
    js_string = json.dumps(basic_dictionary)
    js_string = re.sub(r",(?![^\[]*\])", ",\n", js_string)
    return js_string


def generate_folder_for_activation_functions(activation_fn, folder_string):
    current_folder_string = folder_string + "/" + str(activation_fn)
    os.makedirs(current_folder_string, exist_ok=True)
    return current_folder_string


if __name__ == '__main__':
    main()