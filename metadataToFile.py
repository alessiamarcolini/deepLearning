import os


dataset_path = "datasets_new"


dataset_list = os.listdir(dataset_path)

metadata_file = open("metadata_so1_so2_so3.csv", "w")
metadata_file.write("\t".join(["FILENAME","DATASET", "E-G-L", "CAMERA", "S-B-G", "VARIETY", "S-QUALITY"]))

lEGL = ["E", "G", "L"]
lSBG = ["S", "B", "G", "GB"]
lVARIETY = ["Lagorai", "Vajolet", "Tulameen"]
lSQUALITY = ["A", "AB", "B", "BC", "C"]

FILENAME = ""
DATASET = ""
EGL = ""
CAMERA = ""
SBG = ""
VARIETY = ""
SQUALITY = ""


for dataset in dataset_list:

    if dataset == "so1":
        CAMERA = "NA"
        DATASET = "SO1"
        SQUALITY = "NA"
        l_dir = os.listdir("/".join([dataset_path, dataset])
        print(dataset, l_dir)
        for d in l_dir:
            l_images = os.listdir("/".join([dataset_path, dataset, d])

            for name in l_images:
                FILENAME = name

                meta = name.split("_")
                var = meta[1]
                matur = meta[2]
                form = meta[3]

                if var == "L":
                    VARIETY = lVARIETY[0]
                elif var == "V":
                    VARIETY = lVARIETY[1]

                if matur == "early":
                    EGL = l-EGL[0]
                elif matur == "good":
                    EGL = l-EGL[1]
                elif matur == "late":
                    EGL = l-EGL[2]

                if form == "s":
                    SBG = lSBG[0]
                elif form == "b":
                    SBG = lSBG[1]
                elif form == "g":
                    SBG = lSBG[2]
                elif form == "gb":
                    SBG = lSBG[3]
                line = "\t".join(FILENAME, DATASET, EGL, CAMERA, SBG, VARIETY, SQUALITY)
                metadata_file.write(line)


    if dataset == "so2":
        CAMERA = "Francisca"
        DATASET = "SO2"
        SQUALITY = "NA"
        l_dir1 = os.listdir("/".join(dataset_path, dataset,"train"))
        l_dir2 = os.listdir("/".join(dataset_path, dataset,"validation"))
        l_dirs = [l_dir1, l_dir2]

        for l_dir in l_dirs:
            for d in l_dir:
                l_images = os.listdir("/".join([dataset_path, dataset, d])


                for name in l_images:
                    FILENAME = name

                    meta = name.split("_")
                    var = meta[1]
                    matur = meta[3]
                    form = meta[4]

                    if var == "L":
                        VARIETY = lVARIETY[0]
                    elif var == "V":
                        VARIETY = lVARIETY[1]

                    if matur == "early":
                        EGL = l-EGL[0]
                    elif matur == "good":
                        EGL = l-EGL[1]
                    elif matur == "late":
                        EGL = l-EGL[2]

                    if form == "s":
                        SBG = lSBG[0]
                    if form == "b":
                        SBG = lSBG[1]
                    if form == "g":
                        SBG = lSBG[2]
                    if form == "gb":
                        SBG = lSBG[3]
                    line = "\t".join(FILENAME, DATASET, EGL, CAMERA, SBG, VARIETY, SQUALITY)
                    metadata_file.write(line)

    if dataset == "so3":
        DATASET = "SO3"
        SQUALITY = "NA"
        l_dir = os.listdir("/".join([dataset_path, dataset])
        VARIETY = "NA"

        for d in l_dir:
            l_images = os.listdir("/".join([dataset_path, dataset, d])

            for name in l_images:
                FILENAME = name

                meta = name.split("_")
                matur = meta[1]
                form = meta[2]
                num = int(meta[3])

                if num%2==0:
                    CAMERA = "Chiara"
                else:
                    CAMERA = "Michele"


                if matur == "early":
                    EGL = l-EGL[0]
                elif matur == "good":
                    EGL = l-EGL[1]
                elif matur == "late":
                    EGL = l-EGL[2]

                if form == "s":
                    SBG = lSBG[0]
                elif form == "g":
                    SBG = lSBG[1]
                elif form == "b":
                    SBG = lSBG[2]
                elif form == "gb":
                    SBG = lSBG[3]
                line = "\t".join(FILENAME, DATASET, EGL, CAMERA, SBG, VARIETY, SQUALITY)
                metadata_file.write(line)

    if dataset == "so3_p":
        DATASET = "SO3"
        SQUALITY = "NA"
        VARIETY = ""
        CAMERA = "Prosumer"
        SBG = lSBG[1]
        l_dir = os.listdir("/".join([dataset_path, dataset])
        print(dataset, l_dir)


        for d in l_dir:
            l_images = os.listdir("/".join([dataset_path, dataset, d])

            for name in l_images:
                FILENAME = name

                meta = name.split("_")
                matur = meta[1]
                form = meta[2]
                num = int(meta[3])


                if matur == "early":
                    EGL = l-EGL[0]
                elif matur == "good":
                    EGL = l-EGL[1]
                elif matur == "late":
                    EGL = l-EGL[2]

                line = "\t".join(FILENAME, DATASET, EGL, CAMERA, SBG, VARIETY, SQUALITY)
                metadata_file.write(line)
metadata_file.close()
