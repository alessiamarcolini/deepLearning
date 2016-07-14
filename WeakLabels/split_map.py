import numpy as np
import argparse

class MyArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(MyArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg

parser = MyArgumentParser(
    description='Split MAP in two files: File List and Labels.',
    fromfile_prefix_chars='@')
parser.add_argument('--map_file', type=str, help='Map file to split')

args = parser.parse_args()
MAP_FILE = args.map_file
FILE_LIST = MAP_FILE[:MAP_FILE.rfind(".")]+"_FileList.txt"
LABELS = MAP_FILE[:MAP_FILE.rfind(".")]+"_Labels.txt"

map = np.loadtxt(MAP_FILE, dtype=str)

np.savetxt(FILE_LIST, map[:,0], fmt='%s')
np.savetxt(LABELS, map[:,1], fmt='%s')


