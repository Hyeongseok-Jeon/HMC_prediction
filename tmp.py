import csv
import os
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'G:/17. aimmo 라벨링(모라이)\라벨링결과\최종/tracksMeta'
data_list = os.listdir(data_dir)

veh_num_tot = 0
for i in range(len(data_list)):
    with open(data_dir + '/' + data_list[i], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        veh_num = 0
        for row in spamreader:
            veh_num = veh_num + 1
        veh_num_tot = veh_num_tot + veh_num - 1

argoverse_training_straight = 191024
argoverse_training_left_turn = 7860
argoverse_training_right_turn = 4757
argoverse_training_others = 2301
argoverse_training_tot = 0.01 * (191024 + 7860 + 4757 + 2301)

argoverse_val_straight = 34958
argoverse_val_left_turn = 1880
argoverse_val_right_turn = 1238
argoverse_val_others = 468
argoverse_val_tot = 0.01 * (34958 + 1880 + 1238 + 468)

kaist_training_straight = 50176
kaist_training_left_turn = 21120
kaist_training_right_turn = 34560
kaist_training_others = 128
kaist_training_tot = 0.01 * (50176 + 21120 + 34560 + 128)

kaist_val_straight = 13824
kaist_val_left_turn = 6400
kaist_val_right_turn = 8320
kaist_val_others = 256
kaist_val_tot = 0.01 * (13824 + 6400 + 8320 + 256)

kaist_training_straight_1 = 75264
kaist_training_left_turn_1 = 2112
kaist_training_right_turn_1 = 3456
kaist_training_others_1 = 192
kaist_training_tot_1 = 0.01 * (75264 + 2112 + 3456 + 192)

kaist_val_straight_1 = 20736
kaist_val_left_turn_1 = 640
kaist_val_right_turn_1 = 832
kaist_val_others_1 = 384
kaist_val_tot_1 = 0.01 * (20736 + 640 + 832 + 384)

x = np.arange(4)
years = ['Go Straight', 'Left Turn', 'Right Turn', 'Others']
values_kaist_training = [kaist_training_straight / kaist_training_tot,
                         kaist_training_left_turn / kaist_training_tot,
                         kaist_training_right_turn / kaist_training_tot,
                         kaist_training_others / kaist_training_tot]

values_kaist_training_1 = [kaist_training_straight_1 / kaist_training_tot_1,
                           kaist_training_left_turn_1 / kaist_training_tot_1,
                           kaist_training_right_turn_1 / kaist_training_tot_1,
                           kaist_training_others_1 / kaist_training_tot_1]

values_argoverse_training = [argoverse_training_straight / argoverse_training_tot,
                             argoverse_training_left_turn / argoverse_training_tot,
                             argoverse_training_right_turn / argoverse_training_tot,
                             argoverse_training_others / argoverse_training_tot]

values_kaist_val = [kaist_val_straight / kaist_val_tot,
                    kaist_val_left_turn / kaist_val_tot,
                    kaist_val_right_turn / kaist_val_tot,
                    kaist_val_others / kaist_val_tot]

values_kaist_val_1 = [kaist_val_straight_1 / kaist_val_tot_1,
                      kaist_val_left_turn_1 / kaist_val_tot_1,
                      kaist_val_right_turn_1 / kaist_val_tot_1,
                      kaist_val_others_1 / kaist_val_tot_1]

values_argoverse_val = [argoverse_val_straight / argoverse_val_tot,
                        argoverse_val_left_turn / argoverse_val_tot,
                        argoverse_val_right_turn / argoverse_val_tot,
                        argoverse_val_others / argoverse_val_tot]

values_kaist_tot = [(kaist_training_straight + kaist_val_straight) / (kaist_training_tot + kaist_val_tot),
                    (kaist_training_left_turn + kaist_val_left_turn) / (kaist_training_tot + kaist_val_tot),
                    (kaist_training_right_turn + kaist_val_right_turn) / (kaist_training_tot + kaist_val_tot),
                    (kaist_training_others + kaist_val_others) / (kaist_training_tot + kaist_val_tot)]

values_kaist_tot_1 = [(kaist_training_straight_1 + kaist_val_straight_1) / (kaist_training_tot_1 + kaist_val_tot_1),
                      (kaist_training_left_turn_1 + kaist_val_left_turn_1) / (kaist_training_tot_1 + kaist_val_tot_1),
                      (kaist_training_right_turn_1 + kaist_val_right_turn_1) / (kaist_training_tot_1 + kaist_val_tot_1),
                      (kaist_training_others_1 + kaist_val_others_1) / (kaist_training_tot_1 + kaist_val_tot_1)]

values_argoverse_tot = [(argoverse_training_straight + argoverse_val_straight) / (argoverse_training_tot + argoverse_val_tot),
                        (argoverse_training_left_turn + argoverse_val_left_turn) / (argoverse_training_tot + argoverse_val_tot),
                        (argoverse_training_right_turn + argoverse_val_right_turn) / (argoverse_training_tot + argoverse_val_tot),
                        (argoverse_training_others + argoverse_val_others) / (argoverse_training_tot + argoverse_val_tot)]

####
plt.figure()
bar_width = 0.35
alpha = 0.5
index = np.arange(4)
p1 = plt.bar(index, values_kaist_tot,
             bar_width,
             color='b',
             alpha=1.5*alpha,
             label='Original dataset - balanced')

p2 = plt.bar(index + bar_width, values_kaist_tot_1,
             bar_width,
             color='b',
             alpha=0.5*alpha,
             label='Modified dataset - imbalanced')

for value in p1:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

for value in p2:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

plt.title('Data distribution', fontsize=15, fontname='Times New Roman')
plt.ylabel('Percentage', fontsize=12, fontname='Times New Roman')
plt.ylim(0, 100)
plt.xticks(index + 0.5 * bar_width, years, fontsize=10, fontname='Times New Roman')
plt.legend((p1[0], p2[0]), ('Original dataset - balanced', 'Modified dataset - imbalanced'), fontsize=10)
plt.savefig('data_modification.png', dpi=5000)
plt.show()

####
plt.figure()
bar_width = 0.35
alpha = 0.5
index = np.arange(4)
p1 = plt.bar(index, values_kaist_training,
             bar_width,
             color='b',
             alpha=alpha,
             label='In-house BEV dataset')

p2 = plt.bar(index + bar_width, values_argoverse_training,
             bar_width,
             color='r',
             alpha=alpha,
             label='Argoverse dataset')

for value in p1:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

for value in p2:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

plt.title('Data distribution of training dataset', fontsize=15, fontname='Times New Roman')
plt.ylabel('Percentage', fontsize=12, fontname='Times New Roman')
plt.ylim(0, 100)
plt.xticks(index + 0.5 * bar_width, years, fontsize=10, fontname='Times New Roman')
plt.legend((p1[0], p2[0]), ('In-house BEV dataset', 'Argoverse dataset'), fontsize=10)
plt.savefig('train.png')
plt.show()

####
plt.figure()
bar_width = 0.35
alpha = 0.5
index = np.arange(4)
p1 = plt.bar(index, values_kaist_val,
             bar_width,
             color='b',
             alpha=alpha,
             label='In-house BEV dataset')

p2 = plt.bar(index + bar_width, values_argoverse_val,
             bar_width,
             color='r',
             alpha=alpha,
             label='Argoverse dataset')

for value in p1:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

for value in p2:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

plt.title('Data distribution of validation dataset', fontsize=15, fontname='Times New Roman')
plt.ylabel('Percentage', fontsize=12, fontname='Times New Roman')
plt.ylim(0, 100)
plt.xticks(index + 0.5 * bar_width, years, fontsize=10, fontname='Times New Roman')
plt.legend((p1[0], p2[0]), ('In-house BEV dataset', 'Argoverse dataset'), fontsize=10)
plt.savefig('val.png')
plt.show()

####
plt.figure()
bar_width = 0.35
alpha = 0.5
index = np.arange(4)
p1 = plt.bar(index, values_kaist_tot,
             bar_width,
             color='b',
             alpha=alpha,
             label='In-house BEV dataset')

p2 = plt.bar(index + bar_width, values_argoverse_tot,
             bar_width,
             color='r',
             alpha=alpha,
             label='Argoverse dataset')

for value in p1:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

for value in p2:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width() / 2.,
             1.002 * height, '%.1f%%' % height, ha='center', va='bottom')

plt.title('Data distribution of dataset', fontsize=15, fontname='Times New Roman')
plt.ylabel('Percentage', fontsize=12, fontname='Times New Roman')
plt.ylim(0, 100)
plt.xticks(index + 0.5 * bar_width, years, fontsize=10, fontname='Times New Roman')
plt.legend((p1[0], p2[0]), ('In-house BEV dataset', 'Argoverse dataset'), fontsize=10)
plt.savefig('total.png', dpi=5000)
plt.show()
