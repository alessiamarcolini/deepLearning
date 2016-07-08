__author__ = 'zarbo'

import argparse
import os

class myArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(myArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg


parser = myArgumentParser(description='Create the mapping files from a folder with labels as subfolders.',
        fromfile_prefix_chars='@')
parser.add_argument('--folder', type=str, help='Folder with labels as subfolder, and images inside')
parser.add_argument('--validation', action='store_true', help='Create validation mapping files')
parser.add_argument('--prefix', type=str, help='Prefix for the output file(s)')
args = parser.parse_args()

FOLDER = os.path.abspath(args.folder+"/")+"/"
VALIDATION = args.validation
PREFIX = args.prefix

labels = os.listdir(FOLDER)

paths = []
lbl = []

for i in range(len(labels)):
    paths.extend([FOLDER+labels[i]+"/"+elem for elem in os.listdir(FOLDER+labels[i]+"/")])
    lbl.extend([i for k in range(len(os.listdir(FOLDER+labels[i]+"/")))])

if VALIDATION:
    out_list = open(PREFIX+"_file_list.txt","w")
    out_lbl = open(PREFIX+"_file_labels.txt","w")
else:
    out = open(PREFIX+"_file_map.txt","w")

for i in range(len(paths)):
    if VALIDATION:
        out_list.write(paths[i])
        out_lbl.write(str(lbl[i]))
    else:
        out.write("\t".join([paths[i],str(lbl[i])])+"\n")

if VALIDATION:
    out_list.close()
    out_lbl.close()
else:
    out.close()