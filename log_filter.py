import sys

log = open(sys.argv[1]).readlines()
out_log = open(sys.argv[1]+"_filt.txt","w")
filt_1 = []
for line in log:
	if not line.startswith('Epoch') and line.find('=') == -1:
		if line.startswith("\t#") or line.startswith("\t\t#") or line.startswith("0."):
			filt_1.append(line)
trials_dict = {}
trial_config = {}
for line in filt_1:
#	print line
#	raw_input()
	if line.startswith("0."):
		trials_dict[float(line.strip())] = trial_config
		trial_config = {}
	else:
		line = line.replace("#Chosen ", "").strip().split(":")
		line[0] = line[0].strip()
		line[1] = line[1].strip()
		trial_config[line[0]] = line[1]

keys = trials_dict.keys()
keys.sort()
keys.reverse()

for mcc in keys:
	config = trials_dict[mcc]
	out_line = ["MCC:" + str(mcc)]
	config_values  = config.keys()
	for rec in config_values:
		out_line.append(rec + ":"+str(config[rec]))
	out_log.write("\t".join(out_line)+"\n")
	
out_log.close()
