import numpy as np


def data_distribution_bar_graph():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('tkagg')

    x = [0, 2, 4, 6, 8, 10]
    years = ['Go straight', 'Left turn', 'Right turn', 'Left lane \n change', 'Right lane \n change', 'U-turn']
    values_argo_train = [92.8, 3.8, 2.3, 0.5, 0.6, 0.0]
    values_argo_val = [90.7, 4.9, 3.2, 0.7, 0.5, 0.0]
    values_KAIST_train = [47.3, 19.9, 32.6, 0.0, 0.0, 0.1]
    values_KAIST_val = [48.0, 22.2, 28.9, 0.0, 0.0, 0.9]

    plt.figure()
    plt.bar(x, values_argo_train)
    plt.xticks(x, years)
    plt.title('data distribution - argoverse training')
    plt.ylim(0, 100)
    for i, v in enumerate(x):
        plt.text(v, values_argo_train[i], str(values_argo_train[i])+'%',
                 fontsize = 9,
                 color='black',
                 horizontalalignment='center',
                 verticalalignment='bottom')
    plt.show()

    plt.figure()
    plt.bar(x, values_argo_val)
    plt.xticks(x, years)
    plt.title('data distribution - argoverse validation')
    plt.ylim(0, 100)
    for i, v in enumerate(x):
        plt.text(v, values_argo_val[i], str(values_argo_val[i])+'%',
                 fontsize = 9,
                 color='black',
                 horizontalalignment='center',
                 verticalalignment='bottom')
    plt.show()

    plt.figure()
    plt.bar(x, values_KAIST_train)
    plt.xticks(x, years)
    plt.title('data distribution - KAIST training (128x augmentation)')
    plt.ylim(0, 100)
    for i, v in enumerate(x):
        plt.text(v, values_KAIST_train[i], str(values_KAIST_train[i])+'%',
                 fontsize = 9,
                 color='black',
                 horizontalalignment='center',
                 verticalalignment='bottom')
    plt.show()

    plt.figure()
    plt.bar(x, values_KAIST_val)
    plt.xticks(x, years)
    plt.title('data distribution - KAIST validation (128x augmentation)')
    plt.ylim(0, 100)
    for i, v in enumerate(x):
        plt.text(v, values_KAIST_val[i], str(values_KAIST_val[i])+'%',
                 fontsize = 9,
                 color='black',
                 horizontalalignment='center',
                 verticalalignment='bottom')
    plt.show()

def trajectory_plot():
    from data.drone_data import pred_loader_1, collate_fn
    from model.representation_learning.config_enc import config
    from torch.utils.data import DataLoader
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.use('tkagg')

    config["splicing_num"] = 1
    config["occlusion_rate"] = 0
    config["batch_size"] = 1
    config["LC_multiple"] = 1

    dataset_tot = pred_loader_1(config, 'orig', mode='vis')
    dataloader_tot = DataLoader(dataset_tot,
                                batch_size=config["batch_size"],
                                shuffle=True)

    traj_tot_bag = []
    traj_mod_bag = []
    outlet_bag = []

    for i, data in enumerate(dataloader_tot):
        total_traj, mod_traj, outlet_node_state = data
        total_traj = total_traj[0].numpy()
        mod_traj = mod_traj[0].numpy()
        outlet_node_state = outlet_node_state[0].numpy()

        traj_tot_bag.append(total_traj)
        traj_mod_bag.append(mod_traj)
        outlet_bag.append(outlet_node_state)

    plt.figure()
    outlet_poses = np.concatenate(outlet_bag)
    plt.scatter(outlet_poses[:,0], outlet_poses[:,1], color='b',zorder=10)
    plt.scatter(0, 0, color='r',zorder=10)
    for i in range(len(traj_mod_bag)):
        plt.plot(traj_mod_bag[i][:,0], traj_mod_bag[i][:,1])

    for i in range(len(traj_tot_bag)):
        plt.plot(traj_tot_bag[i][:, 0], traj_tot_bag[i][:, 1])

    plt.xlim(-55, 90)
    plt.ylim(-70, 70)