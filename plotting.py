import torch
from matplotlib import pyplot as plt


def plot_gait(gait, global_steps, as_tb=True, save_dir=None):
    with torch.no_grad():
        nb_actuators = gait.nb_actuators

        period = int(gait.period()) + 1
        all_frames = torch.linspace(0, period, period * 100).unsqueeze(-1).cuda()

        gaits = gait(all_frames).cpu().numpy()
        period = round(gait.period())

        all_frames = all_frames.cpu().numpy()

    DEBUG = False

    plots = {}
    for actuator_id in range(nb_actuators):
        y1 = gaits[:, actuator_id]

        # print(y1)

        if DEBUG or not as_tb:
            plt.plot(all_frames, y1)
            plt.title(f'A{actuator_id} cycle @ {global_steps} steps')
            plt.xlabel(f'Period is {period} frames')
            plt.ylabel('activation')

            if DEBUG:
                plt.show()

            plt.savefig(f'{save_dir}/image_actuator{actuator_id}.png')
            plt.clf()
            """
            plt.gcf().canvas.get_renderer()
            fig = plt.gcf()
            img = Image.frombytes(
                'RGB',
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb()
            )

            #img = np.array(img)
            #img = torch.tensor(img)
            #img = img.permute(2,0,1)"""

            # plots[f"image_actuator{actuator_id}"] = img
        else:
            tag = f"A{actuator_id}_S{global_steps}_P{period}"
            plots[tag] = (all_frames.squeeze(), y1.squeeze())
    return plots
