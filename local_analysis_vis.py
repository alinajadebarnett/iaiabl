import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import os
import argparse
import re
import shutil

classname_dict = dict()
classname_dict[0] = "circumscribed"
classname_dict[1] = "indistinct"
classname_dict[2] = "spiculated"

def main():
    # get dir
    parser = argparse.ArgumentParser()
    parser.add_argument('-local_analysis_directory', nargs=1, type=str, default='0')
    args = parser.parse_args()

    source_dir = args.local_analysis_directory[0]

    os.makedirs(os.path.join(source_dir, 'visualizations_of_expl'), exist_ok=True)

    pred, truth = read_local_analysis_log(os.path.join(source_dir + 'local_analysis.log'))

    anno_opts_cen = dict(xy=(0.4, 0.5), xycoords='axes fraction',
                    va='center', ha='center')
    anno_opts_symb = dict(xy=(1, 0.5), xycoords='axes fraction',
                    va='center', ha='center')
    anno_opts_sum = dict(xy=(0, -0.1), xycoords='axes fraction',
            va='center', ha='left')
    
    ###### all classes, one expl
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 12)

    ncols, nrows = 7, 3
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    plt.rcParams.update({'font.size': 14})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc='left', color = "k")
        elif ax_num == 1:
            ax.set_title("Test image activation\nby prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 3:
            ax.set_title("Self-activation of\nprototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc='left', color = "k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc='left', color = "k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc='left', color = "k")
        else:
            pass

    plt.rcParams.update({'font.size': 22})

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate('x', **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate('=', **anno_opts_symb)

    # get and plot data from source directory

    orig_img = Image.open(os.path.join(source_dir + 'original_img.png'))

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(orig_img)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    top_p_dir = os.path.join(source_dir + 'most_activated_prototypes')
    for top_p in range(3):
        # put info in place
        p_info_file = open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_prototype.txt'), 'r')
        sim_score, cc_dict, class_str, top_cc_str = read_info(p_info_file)
        p_info_file.close()
        for ax in [f_axes[top_p][4]]:
            ax.annotate(sim_score, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][5]]:
            ax.annotate(top_cc_str + "\n" + class_str, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][6]]:
            tc = float(top_cc_str) * float(sim_score)
            ax.annotate('{0:.3f}'.format(tc) + "\n" + class_str, **anno_opts_cen)
            ax.set_axis_off()
        # put images in place
        p_img = Image.open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_prototype_full_size.png'))
        for ax in [f_axes[top_p][2]]:
            ax.imshow(p_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        p_act_img = Image.open(os.path.join(top_p_dir, f'top-{top_p+1}_activated_prototype_self_act.png'))
        for ax in [f_axes[top_p][3]]:
            ax.imshow(p_act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        act_img = Image.open(os.path.join(top_p_dir, f'prototype_activation_map_by_top-{top_p+1}_prototype_normed.png'))
        for ax in [f_axes[top_p][1]]:
            ax.imshow(act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    #summary
    f_axes[2][4].annotate(f"This {classname_dict[int(truth)]} lesion is classified as {classname_dict[int(pred)]}.", **anno_opts_sum)

    save_loc1 = os.path.join(source_dir, 'visualizations_of_expl') + f'/all_class.png'
    plt.savefig(save_loc1, bbox_inches='tight', pad_inches=0)
    os.makedirs('./visualizations_of_expl/', exist_ok=True)
    save_loc2 = './visualizations_of_expl/' + str(source_dir.replace('/', '__'))[len('__usr__xtmp__IAIABL__saved_models__0129_pushonall_topkk=9_fa=0.001_random=4__pruned_prototypes_epoch50_k6_pt3__'):] + f'all_class.png'
    shutil.copy2(save_loc1, save_loc2)
    print(f"Saved in {save_loc2}")
    return

def read_local_analysis_log(file_loc):
    log_file = open(file_loc, 'r')
    for _ in range(30):
        line = log_file.readline()
        if line[0:len("Predicted: ")] == "Predicted: ":
            pred = line[len("Predicted: "):]
        elif line[0:len("Actual: ")] == "Actual: ":
            actual = line[len("Actual: "):]
    # pred = log_file.readline()[len("Predicted: "):]
    # actual = log_file.readline()[len("Actual: "):]
    log_file.close()
    return pred, actual


def read_info(info_file, per_class=False):
    sim_score_line = info_file.readline()
    connection_line = info_file.readline()
    proto_index_line = info_file.readline()
    cc_0_line = info_file.readline()
    cc_1_line = info_file.readline()
    cc_2_line = info_file.readline()

    sim_score = sim_score_line[len("similarity: "):-1]
    if per_class:
        cc = connection_line[len('last layer connection: '):-1]
    else:
        cc = connection_line[len('last layer connection with predicted class: '):-1]
    circ_cc_str = cc_0_line[len('proto connection to class 0:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+1)]
    circ_cc = float(circ_cc_str)
    indst_cc_str = cc_1_line[len('proto connection to class 1:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+1)]
    indst_cc = float(indst_cc_str)
    spic_cc_str = cc_2_line[len('proto connection to class 2:tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+1)]
    spic_cc = float(spic_cc_str)

    cc_dict = dict()
    cc_dict[0] = circ_cc
    cc_dict[1] = indst_cc
    cc_dict[2] = spic_cc
    class_of_p = max(cc_dict, key=lambda k: cc_dict[k])
    top_cc = cc_dict[class_of_p]

    class_str = classname_dict[class_of_p]
    if class_of_p == 0:
        top_cc_str = circ_cc_str
    elif class_of_p == 1:
        top_cc_str = indst_cc_str
    elif class_of_p == 2:
        top_cc_str = spic_cc_str
    else:
        print("Error. The maximum value class is not found.")
    
    return sim_score, cc_dict, class_str, top_cc_str

def test():

    im = Image.open('./visualizations_of_expl/' + 'original_img.png')

    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 12)

    ncols, nrows = 7, 3
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    plt.rcParams.update({'font.size': 15})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc='left', color = "k")
        elif ax_num == 1:
            ax.set_title("Test image activation by prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 3:
            ax.set_title("Self-activation of prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc='left', color = "k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc='left', color = "k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc='left', color = "k")
        else:
            pass

    plt.rcParams.update({'font.size': 22})

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(im)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


    anno_opts = dict(xy=(0.4, 0.5), xycoords='axes fraction',
                    va='center', ha='center')

    anno_opts_symb = dict(xy=(1, 0.5), xycoords='axes fraction',
                    va='center', ha='center')

    for ax in [f_axes[r][s] for r in range(nrows) for s in range(4,7)]:
        ax.annotate('Number', **anno_opts)
        ax.set_axis_off()

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate('x', **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate('=', **anno_opts_symb)

    os.makedirs('./visualizations_of_expl/', exist_ok=True)
    plt.savefig('./visualizations_of_expl/' + 'test.png')

    # Refs: https://stackoverflow.com/questions/40846492/how-to-add-text-to-each-image-using-imagegrid
    # https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

if __name__ == "__main__":
    main()