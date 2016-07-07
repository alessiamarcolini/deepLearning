"""
Collect Statistics on the SOM dataset
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

import os

statistics = dict()

SOM_ROOT_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'SOM')
SOM_IMAGES_ROOT_FOLDER = os.path.join(SOM_ROOT_FOLDER, 'images')
SOM_METADATA_FILE = os.path.join(SOM_ROOT_FOLDER, 'metadata.csv')


def collect_statistics():
    """Collect Statistics for the SOM dataset.

    This script gathers information from an existing
    `metadata.csv` file and collects information
    organised by Raspberry varieties, SO Qualities,
    and SO Categories.
    """

    with open(SOM_METADATA_FILE) as csv_file:
        for i, line in enumerate(csv_file):
            if i == 0:  # Skip the header line
                continue
            line = line.strip()
            dir_name, _, rtype, rclass, *rest = line.split(',')
            statistics.setdefault(rtype, {})
            statistics[rtype].setdefault(rclass, dict())

            target_folder = os.path.join(SOM_IMAGES_ROOT_FOLDER, dir_name)
            files_in_folder = os.listdir(target_folder)
            for filename in files_in_folder:
                if filename.endswith('.jpg'):
                    fname, ext = os.path.splitext(filename)
                    statistics[rtype][rclass].setdefault(fname, list())
                    statistics[rtype][rclass][fname].append(os.path.join(target_folder, filename))
    return statistics
