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
             alpha=1.5 * alpha,
             label='Original dataset - balanced')

p2 = plt.bar(index + bar_width, values_kaist_tot_1,
             bar_width,
             color='b',
             alpha=0.5 * alpha,
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
plt.savefig('plots/data_modification.png', dpi=5000)
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
plt.savefig('plots/train.png')
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
plt.savefig('plots/val.png')
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
plt.savefig('plots/total.png', dpi=5000)
plt.show()


##############

def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert (d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.))


###################
import csv
import os
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

sdv_x_LT_bal = 9.47
sdv_y_LT_bal = 10.94
sdv_x_RT_bal = 13.65
sdv_y_RT_bal = 10.63
sdv_x_GO_bal = 9.25
sdv_y_GO_bal = 13.75
stv_bal = [sdv_x_LT_bal, sdv_y_LT_bal, sdv_x_RT_bal, sdv_y_RT_bal, sdv_x_GO_bal, sdv_y_GO_bal]

sdv_x_LT_imbal = 18.8
sdv_y_LT_imbal = 16.23
sdv_x_RT_imbal = 16.07
sdv_y_RT_imbal = 22.33
sdv_x_GO_imbal = 13.34
sdv_y_GO_imbal = 12.17
stv_imbal = [sdv_x_LT_imbal, sdv_y_LT_imbal, sdv_x_RT_imbal, sdv_y_RT_imbal, sdv_x_GO_imbal, sdv_y_GO_imbal]
years = ['Balanced training set', 'Imbalanced training set']

plt.figure(figsize=[10, 2.5])
bar_width = 0.3
alpha = [0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
index = 3*np.arange(2)
p1 = []
p2 = []
for i in range(len(stv_bal)):
    bar = plt.bar(index[0]+bar_width*i, stv_bal[i],
                  bar_width,
                  color='b',
                  alpha=alpha[i],
                  label='In-house BEV dataset',
                  edgecolor='k')
    p1.append(bar)

for i in range(len(stv_bal)):
    bar = plt.bar(index[1]+bar_width*i, stv_imbal[i],
                  bar_width,
                  color='r',
                  alpha=alpha[i],
                  label='In-house BEV dataset',
                  edgecolor='k')
    p2.append(bar)

for i in range(len(p1)):
    value = p1[i]
    for val in value:
        height = val.get_height()
        plt.text(val.get_x() + val.get_width() / 2.,
                 1.002 * height, '%.1f' % height, ha='center', va='bottom')
    if i == 0:
        maneuver = 'x_LT'
    elif i == 1:
        maneuver = 'y_LT'
    elif i == 2:
        maneuver = 'x_RT'
    elif i == 3:
        maneuver = 'y_RT'
    elif i == 4:
        maneuver = 'x_ST'
    elif i == 5:
        maneuver = 'y_ST'
    plt.text(val.get_x() + val.get_width() / 2.,
             6, maneuver, ha='center', va='bottom')

for i in range(len(p2)):
    value = p2[i]
    for val in value:
        height = val.get_height()
        plt.text(val.get_x() + val.get_width() / 2.,
                 1.002 * height, '%.1f' % height, ha='center', va='bottom')
    if i == 0:
        maneuver = 'x_LT'
    elif i == 1:
        maneuver = 'y_LT'
    elif i == 2:
        maneuver = 'x_RT'
    elif i == 3:
        maneuver = 'y_RT'
    elif i == 4:
        maneuver = 'x_ST'
    elif i == 5:
        maneuver = 'y_ST'
    plt.text(val.get_x() + val.get_width() / 2.,
             8, maneuver, ha='center', va='bottom')

plt.title('Standard deviation of each maneuver class for training dataset', fontsize=15, fontname='Times New Roman')
plt.xticks([index[0]+bar_width*2.5,index[1]+bar_width*2.5], years, fontsize=13, fontname='Times New Roman')
plt.legend((p1[0], p2[0]), (years[0], years[1]), fontsize=10)
plt.ylim(0, 25)
plt.savefig('plots/sdv.png')
plt.show()


###################
import csv
import os
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

LT_ST_bal = 7.8905
ST_RT_bal = 7.6795
RT_LT_bal = 8.503

LT_ST_imbal = 7.886
ST_RT_imbal = 7.3525
RT_LT_imbal = 9.465

