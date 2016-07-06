__author__ = 'zarbo'
import os

lag_e = "./BerryPhotos_LV/L/train/early/"
lag_g = "./BerryPhotos_LV/L/train/good/"
lag_l = "./BerryPhotos_LV/L/train/late/"

vaj_e = "./BerryPhotos_LV/V/train/early/"
vaj_g = "./BerryPhotos_LV/V/train/good/"
vaj_l = "./BerryPhotos_LV/V/train/late/"

label_dict = {
                "L_E":0,
                "L_G":1,
                "L_L":2,
                "V_E":3,
                "V_G":4,
                "V_L":5,
                "E":0,
                "G":1,
                "L":2
            }

weak_dict_lbl = {"L_E":[],
                "L_G":[],
                "L_L":[],
                "V_E":[],
                "V_G":[],
                "V_L":[]
                }
hard_dict_lbl = {"E":[],
                 "G":[],
                 "L":[]
                }

img = os.listdir(lag_e)
for m in img:
    weak_dict_lbl['L_E'].append(lag_e+m)
    hard_dict_lbl['E'].append(lag_e+m)

img = os.listdir(lag_g)
for m in img:
    weak_dict_lbl['L_G'].append(lag_g+m)
    hard_dict_lbl['G'].append(lag_g+m)

img = os.listdir(lag_l)
for m in img:
    weak_dict_lbl['L_L'].append(lag_l+m)
    hard_dict_lbl['L'].append(lag_l+m)

img = os.listdir(vaj_e)
for m in img:
    weak_dict_lbl['V_E'].append(vaj_e+m)
    hard_dict_lbl['E'].append(vaj_e+m)

img = os.listdir(vaj_g)
for m in img:
    weak_dict_lbl['V_G'].append(vaj_g+m)
    hard_dict_lbl['G'].append(vaj_g+m)

img = os.listdir(vaj_l)
for m in img:
    weak_dict_lbl['V_L'].append(vaj_l+m)
    hard_dict_lbl['L'].append(vaj_l+m)


out = open("weak_map.txt", "w")
for k in weak_dict_lbl.keys():
    for p in weak_dict_lbl[k]:
        out.write("\t".join([p,str(label_dict[k])])+"\n")
out.close()

out = open("hard_map.txt", "w")
for k in hard_dict_lbl.keys():
    for p in hard_dict_lbl[k]:
        out.write("\t".join([p,str(label_dict[k])])+"\n")
out.close()



