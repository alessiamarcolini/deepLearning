import os


dataset_path = "/home/webvalley/deepLearning/datasets_cal"


dataset_list = os.listdir(dataset_path)
dataset_list.sort()

out = open("so1_so3_labels.csv", "w")

for dataset in dataset_list:
    l_dir = os.listdir("/".join([dataset_path, dataset]))
    l_dir.sort()
    for d in l_dir:
        l_images = (os.listdir("/".join([dataset_path, dataset, d])))
        l_images.sort()

        if d == "early":
            l = 0
        elif d == "good":
            l = 1
        elif d == "late":
            l = 2

        for name in l_images:
            abs_path = "/".join([dataset_path, dataset, d, name])
            out.write("\t".join([abs_path, str(l)]) + "\n")

out.close()
