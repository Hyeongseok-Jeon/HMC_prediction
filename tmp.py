import csv
import os

data_dir = 'G:/17. aimmo 라벨링(모라이)\라벨링결과\최종/tracksMeta'
data_list = os.listdir(data_dir)

veh_num_tot = 0
for i in range(len(data_list)):
    with open(data_dir+'/'+data_list[i], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        veh_num = 0
        for row in spamreader:
            veh_num = veh_num + 1
        veh_num_tot = veh_num_tot + veh_num-1