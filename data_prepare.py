import os
import shutil
import sys


def prepare_data():
    path = sys.argv[1]  # Where the dataset folders are located
    phishing_data_path = path + "\\" + sys.argv[2]  # Where the phishing data is located
    benign_data_path = path + "\\" + sys.argv[3]  # Where the benign data is located
    misleading_data_path = path + "\\" + sys.argv[4]  # Where the misleading data is located

    # Path to the folder where the benign and misleading html.txt files will be combined
    benign_misleading_path = "benign_mislead"

    # Path to the folder where the phishing html.txt files will be combined
    phishing_path = "phishing"

    # Create the benign_misleading folder if it does not exist
    if not os.path.exists(benign_misleading_path):
        os.makedirs(benign_misleading_path)

    # Create the phishing folder if it does not exist
    if not os.path.exists(phishing_path):
        os.makedirs(phishing_path)
    counter = 0
    total = 0
    phis = 0
    ben_misled = 0
    # Browse all folders in benign folder and copy the html.txt files to benign_misleading folder
    for root, dirs, files in os.walk(benign_data_path):
        for file in files:
            if file.endswith("html.txt"):
                counter += 1
                total += 1
                ben_misled += 1
                shutil.copy(os.path.join(root, file),
                            benign_misleading_path + "\\" + root[len(benign_data_path) + 1:].replace(".",
                                                                                                     "_") + "_" + file)
                # print(benign_misleading_path + "\\" + root[len(benign_data_path) + 1:].replace(".","_") + "_" + file)
                # print(os.path.join(root, file))

    for root, dirs, files in os.walk(misleading_data_path):
        for file in files:
            if file.endswith("html.txt"):
                counter += 1
                total += 1
                ben_misled += 1
                shutil.copy(os.path.join(root, file),
                            benign_misleading_path + "\\" + root[len(misleading_data_path) + 1:].replace(".",
                                                                                                         "_") + "_" + file)
                # print(benign_misleading_path + "\\" + root[len(misleading_data_path) + 1:].replace(".","_") + "_" + file)
                # print(os.path.join(root, file))

    for root, dirs, files in os.walk(phishing_data_path):
        for file in files:
            if file.endswith("html.txt"):
                counter += 1
                total += 1
                phis += 1
                shutil.copy(os.path.join(root, file),
                            phishing_path + "\\" + root[len(phishing_data_path) + 1:].replace(".", "_") + "_" + file)
                # print(phishing_path + "\\" + root[len(phishing_data_path) + 1:].replace(".","_") + "_" + file)
                # print(os.path.join(root, file))
    print(counter)
    print(total)
    print(phis)
    print(ben_misled)


if __name__ == '__main__':
    prepare_data()
