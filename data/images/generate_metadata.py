"""

"""

import os
from collections import OrderedDict

DATASET_PATH = os.path.join(os.path.abspath(os.path.curdir), "datasets_new")
CSV_SEP = ','  # Separator Character in the Output CSV file

# CONSTANTS LITERAL
NAN = "NA"
SO1 = "so1"
SOS = "sos"
SO3P = "so3_p"
SO3 = "so3"
SO2 = "so2"
SO4 = 'so4'
SOM = "som"

# Mapping of values extracted from file names
MATURATION_CLASSES = {'e': "E", 'g': "G", 'l': "L"}
CONFIGURATIONS = {"s": "S", "b": "B", "g": "G", "gb": "GB"}
VARIETIES = {'l': "Lagorai", 'v': "Vajolet", 't': "Tulameen"}

# Metadata File name for SOM images
SOM_METADATA_FILENAME = 'som_metadata.csv'

# HEADINGS
FILENAME = "FILENAME"
DATASET = "DATASET"
CLASS = "CLASS"
CAMERA = "CAMERA"
CONF = "CONF"
VARIETY = "VARIETY"
SOMQ = "SOMQ"
SOSQ = 'SOSQ'
CAT = "CAT"
FILEPATH = "FILEPATH"

HEADING_LINE = CSV_SEP.join([FILENAME, DATASET, CLASS, CAMERA, CONF, VARIETY, SOSQ, SOMQ, CAT, FILEPATH])

METADATA_LINE = OrderedDict()
METADATA_LINE[FILENAME] = ''
METADATA_LINE[DATASET] = ''
METADATA_LINE[CLASS] = ''
METADATA_LINE[CAMERA] = ''
METADATA_LINE[CONF] = ''
METADATA_LINE[VARIETY] = ''
METADATA_LINE[SOMQ] = ''
METADATA_LINE[CAT] = ''

## DEFAULT MAPPINGS FOR SPECIFIC SOs
DATASET_METADATA = {
    SO1 : {CAMERA:NAN, DATASET:SO1, SOMQ:NAN, CAT:NAN, SOSQ:NAN},
    SO2 : {CAMERA:"Fr", DATASET:SO2, SOMQ:NAN, CAT:NAN, SOSQ:NAN},
    SO3 : {DATASET:SO3, SOMQ:NAN, CAT:NAN, VARIETY:NAN, SOSQ:NAN},
    SO3P: {DATASET:SO3, SOMQ:NAN, VARIETY:NAN, CAMERA: "Pr", CAT:NAN, SOSQ:NAN},
    SO4 : {DATASET:SO4, SOMQ:NAN, CAMERA:"Mi", CAT:NAN, SOSQ:NAN, },
    SOS : {DATASET:SOS, VARIETY:NAN, CAMERA:NAN, CONF:CONFIGURATIONS['gb'], CLASS:NAN, CAT:NAN, SOMQ:NAN},
    SOM : {DATASET:SOM, CAMERA:NAN, CONF:CONFIGURATIONS['b'], CLASS:NAN, SOSQ:NAN}
}


def process_SO_dataset(metadata_file, dataset_folder_name,
                       dataset_folder_path):
    """
    Function to gather information for all the SO datasets.
    These datasets have images collected in different folders, one per
    each class, and metadata infos are hard coded in image file names.

    The format of the image file is (for each maturation class):
        dsname_variety_maturation_configuration_number1[_number2].jpg
    """

    classes_folders = os.listdir(dataset_folder_path)
    classes_folders.sort()
    # print(dataset_folder_name, classes_folders)
    for class_name in classes_folders:
        image_files = os.listdir(os.path.join(dataset_folder_path, class_name))
        image_files.sort()
        for name in image_files:
            metadata_line = OrderedDict(METADATA_LINE)
            # Updates Metadata with Defaults
            metadata_line.update(DATASET_METADATA[dataset_folder_name])
            # Fill infos specific for file
            metadata_line[FILENAME] = name
            metadata_line[FILEPATH] = os.path.abspath(os.path.join(dataset_folder_path,
                                                                   class_name, name))

            if dataset_folder_name == SO1:
                _, var, matur, conf, _ = name.split("_")
            elif dataset_folder_name == SO2:
                _, var, _, matur, conf, _ = name.split('_')
            else: ## SO4
                _, var, matur, conf, *rest = name.split("_")

            metadata_line[VARIETY] = VARIETIES[var.lower()]
            metadata_line[CLASS] = MATURATION_CLASSES[matur.lower()]
            metadata_line[CONF] = CONFIGURATIONS[conf.lower()]

            line = CSV_SEP.join(metadata_line.values())
            metadata_file.write(line + "\n")
    print('{} DONE!'.format(dataset_folder_name))


def process_SO3_dataset(metadata_file, dataset_folder_name,
                        dataset_folder_path):
    """
    Function to gather information for the SO3 dataset ONLY.
    These datasets have images collected in different folders, one per
    each class, and metadata infos are hard coded in image file names.
    However, these photos do NOT have VARIETY information.

    The format of the image file is (for each maturation class):
        dsname_maturation_configuration_number1_number2.jpg
    """

    classes_folders = os.listdir(dataset_folder_path)
    classes_folders.sort()
    # print(dataset_folder_name, classes_folders)
    for class_name in classes_folders:
        image_files = os.listdir(os.path.join(dataset_folder_path, class_name))
        image_files.sort()
        for name in image_files:
            metadata_line = OrderedDict(METADATA_LINE)
            # Updates Metadata with Defaults
            metadata_line.update(DATASET_METADATA[dataset_folder_name])
            # Fill infos specific for file
            metadata_line[FILENAME] = name
            metadata_line[FILEPATH] = os.path.abspath(os.path.join(dataset_folder_path,
                                                                   class_name, name))
            
            fname, _ = os.path.splitext(name)
            _, mat, conf, num, *rest = fname.split("_")
            num = int(num)
            if num % 2 == 0:
                camera = "Ch"
            else:
                camera = "Mi"
            metadata_line[CLASS] = MATURATION_CLASSES[mat.lower()]
            metadata_line[CAMERA] = camera
            metadata_line[CONF] = CONFIGURATIONS[conf.lower()]

            line = CSV_SEP.join(metadata_line.values())
            metadata_file.write(line + "\n")
    print('{} DONE!'.format(dataset_folder_name))


def process_SO3P_dataset(metadata_file, dataset_folder_name,
                            dataset_folder_path):
    """
    Function to gather information for the SO3P dataset ONLY.
    These datasets have images collected in different folders, one per
    each class, and metadata infos are hard coded in image file names.
    However, these photos do NOT have VARIETY information.
    Moreover, the CAMERA information is pre-set (i.e. "Prosumer")

    The format of the image file is (for each maturation class):
        dsname_maturation_configuration_number1_number2.jpg
    """

    classes_folders = os.listdir(dataset_folder_path)
    classes_folders.sort()
    # print(dataset_folder_name, classes_folders)
    for class_name in classes_folders:
        image_files = os.listdir(os.path.join(dataset_folder_path, class_name))
        image_files.sort()
        for name in image_files:
            metadata_line = OrderedDict(METADATA_LINE)
            # Updates Metadata with Defaults
            metadata_line.update(DATASET_METADATA[dataset_folder_name])
            # Fill infos specific for file
            metadata_line[FILENAME] = name
            metadata_line[FILEPATH] = os.path.abspath(os.path.join(dataset_folder_path,
                                                                   class_name, name))
            _, mat, conf, *rest = name.split("_")
            metadata_line[CLASS] = MATURATION_CLASSES[mat.lower()]
            metadata_line[CONF] = CONFIGURATIONS[conf.lower()]

            line = CSV_SEP.join(metadata_line.values())
            metadata_file.write(line + "\n")
    print('{} DONE!'.format(dataset_folder_name))


def process_SOS_dataset(metadata_file, dataset_folder_name,
                            dataset_folder_path):
    """
    Function to gather information for all the SOS datasets.
    These datasets have images collected in different folders, one per
    each class, and metadata infos are hard coded in image file names.

    For **this** particular dataset, no information is hardcoded in the
    name of files (only in folders names, i.e. TYPE)
    """

    classes_folders = os.listdir(dataset_folder_path)
    classes_folders.sort()
    # print(dataset_folder_name, classes_folders)
    for class_name in classes_folders:
        image_files = os.listdir(os.path.join(dataset_folder_path, class_name))
        image_files.sort()
        for name in image_files:
            metadata_line = OrderedDict(METADATA_LINE)
            # Updates Metadata with Defaults
            metadata_line.update(DATASET_METADATA[dataset_folder_name])
            # Fill infos specific for file
            metadata_line[FILENAME] = name
            metadata_line[FILEPATH] = os.path.abspath(os.path.join(dataset_folder_path,
                                                                   class_name, name))
            metadata_line[SOSQ] = class_name.upper()

            line = CSV_SEP.join(metadata_line.values())
            metadata_file.write(line + "\n")
    print('{} DONE!'.format(dataset_folder_name))


def process_SOM_dataset(metadata_file, dataset_folder_name, dataset_folder_path):
    """
    Function to gather information for all the SOM datasets.
    These datasets have images collected in different folders, one per
    each specific STOCK_CODE.
    For this particular dataset, metadata information are extracted from a
    separate CSV file and then collected from file system.
    Image names embed information related to the SO Category

    For **this** particular dataset, no information is hardcoded in the
    name of files (only in folders names, i.e. TYPE)
    """

    som_metadata_filepath = os.path.join(dataset_folder_path, SOM_METADATA_FILENAME)

    def collect_statistics():
        """Collect Statistics for the SOM dataset.

        This script gathers information from an existing
        `metadata.csv` file and collects information
        organised by Raspberry varieties, SO Qualities,
        and SO Categories.
        """
        statistics = OrderedDict()
        with open(som_metadata_filepath) as csv_file:
            for i, line in enumerate(csv_file):
                if i == 0:  # Skip the header line
                    continue
                line = line.strip()
                stock_code, _, rtype, rclass, *rest = line.split(',')
                statistics.setdefault(rtype, {})
                statistics[rtype].setdefault(rclass, dict())

                target_folder = os.path.join(dataset_folder_path, stock_code)
                files_in_folder = os.listdir(target_folder)
                for filename in files_in_folder:
                    if filename.endswith('.jpg'):
                        fname, _ = os.path.splitext(filename)
                        statistics[rtype][rclass].setdefault(fname, list())
                        statistics[rtype][rclass][fname].append(os.path.join(target_folder, filename))
        return statistics

    som_ds_stats = collect_statistics()

    for variety in som_ds_stats:  # iterate over varieties
        for somq in som_ds_stats[variety]:  # iterate over SOMQ classes
            for som_category in som_ds_stats[variety][somq]:  # iterate over categories
                for image_filepath in som_ds_stats[variety][somq][som_category]:
                    metadata_line = OrderedDict(METADATA_LINE)
                    # Updates Metadata with Defaults
                    metadata_line.update(DATASET_METADATA[dataset_folder_name])

                    # Fill infos specific for file
                    image_fname, _ = os.path.splitext(image_filepath)
                    _, fname = os.path.split(image_fname)
                    metadata_line[FILENAME] = fname
                    metadata_line[FILEPATH] = image_filepath
                    metadata_line[SOMQ] = somq.upper()
                    metadata_line[VARIETY] = variety.title()
                    metadata_line[CAT] = som_category.upper()

                    line = CSV_SEP.join(metadata_line.values())
                    metadata_file.write(line + "\n")
    print('{} DONE!'.format(dataset_folder_name))


if __name__ == '__main__':

    with open("metadata_so1_so2_so3_so4_sos_som.csv", "w") as metadata_file:
        # Write Headings (first line)
        metadata_file.write(HEADING_LINE + "\n")

        dataset_folders = os.listdir(DATASET_PATH)
        dataset_folders.sort()  # Sort Folder Names in Lexicographic Order

        for dataset_folder_name in dataset_folders:
            dataset_folder_path = os.path.join(DATASET_PATH, dataset_folder_name)

            if dataset_folder_name == SO1:
                process_SO_dataset(metadata_file, dataset_folder_name,
                                   dataset_folder_path)
            if dataset_folder_name == SO2:
                training_folder_path = os.path.join(DATASET_PATH, dataset_folder_name,
                                                    'train')
                process_SO_dataset(metadata_file, dataset_folder_name, training_folder_path)
                validation_folder_path = os.path.join(DATASET_PATH, dataset_folder_name,
                                                      'validation')
                process_SO_dataset(metadata_file, dataset_folder_name, validation_folder_path)
            if dataset_folder_name == SO3:
                process_SO3_dataset(metadata_file, dataset_folder_name, dataset_folder_path)
            if dataset_folder_name == SO3P:
                process_SO3P_dataset(metadata_file, dataset_folder_name, dataset_folder_path)
            if dataset_folder_name == SO4:
                process_SO_dataset(metadata_file, dataset_folder_name,
                                   dataset_folder_path)
            if dataset_folder_name == SOS:
                process_SOS_dataset(metadata_file, dataset_folder_name, dataset_folder_path)
            if dataset_folder_name == SOM:
                process_SOM_dataset(metadata_file, dataset_folder_name, dataset_folder_path)

    print('Processing Complete')
