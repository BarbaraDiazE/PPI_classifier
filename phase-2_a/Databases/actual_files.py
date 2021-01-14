import os


def actual_files(my_dir):
    files = list()
    for file in os.listdir(my_dir):
        if file.endswith('.csv'):
            files.append(file)
    return files
