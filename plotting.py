import os

import torch
from matplotlib import pyplot as plt

def array2csv_str(nd_array):
    nd_array = nd_array.squeeze()
    nd_array = nd_array.tolist()
    return ','.join(map(str,nd_array))


def plot_gait(gait, global_steps, save_dir=None):
    with torch.no_grad():
        nb_actuators = gait.nb_actuators

        period = int(gait.period()) + 1
        all_frames = torch.linspace(0, period, period * 100).unsqueeze(-1).cuda()

        gaits = gait(all_frames).cpu().numpy()
        period = round(gait.period())

        all_frames = all_frames.cpu().numpy()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gait_csv_str = '\n'.join([array2csv_str(all_frames), *list(map(array2csv_str, gaits.T))])
    with open(f"{save_dir}/gaits.csv", "w") as f:
        f.write(gait_csv_str)

    #plots = {}
    for actuator_id in range(nb_actuators):
        y1 = gaits[:, actuator_id]

        if save_dir is not None:
            plt.plot(all_frames, y1)
            plt.title(f'A{actuator_id} cycle @ {global_steps} steps')
            plt.xlabel(f'Period is {period} frames')
            plt.ylabel('activation')

            plt.savefig(f'{save_dir}/image_actuator{actuator_id}.png')
            plt.clf()

        #tag = f"eval_gait/A{actuator_id}_S{global_steps}_P{period}"
        #plots[tag] = (all_frames.squeeze(), y1.squeeze())
    #return plots
