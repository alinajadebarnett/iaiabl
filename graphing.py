import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from PIL import Image
from dataHelper import DatasetFolder, DatasetFolder_WithReplacement
from load_run import accu, auroc, bootstrapped_test, calc_kappa
from highlighting_precision import hp
from vanilla_vgg import Vanilla_VGG
import re
import numpy as np
import os
import train_and_test as tnt
from sklearn.metrics import roc_curve, auc
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse

# line styles
linestyle_tuple = [
                    ('solid',                 'solid'),

                    ('densely dashed',        (0, (5, 1))),
                    ('densely dotted',        (0, (1, 1))),
                    ('densely dashdotted',    (0, (3, 1, 1, 1))),
                    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
                    ('densely dashdashdotted', (0, (3, 1, 3, 1, 1, 1))),

                    ('dashed',        (0, (5, 2))),
                    ('dotted',        (0, (1, 2))),
                    ('dashdotted',    (0, (3, 2, 1, 2))),
                    ('dashdotdotted', (0, (3, 2, 1, 2, 1, 2))),
                    ('dashdashdotted', (0, (3, 2, 3, 2, 1, 2))),

                #  ('loosely dashed',        (0, (5, 10))),
                #  ('loosely dashdotted',    (0, (3, 10, 1, 10))),
                #  ('loosely dotted',        (0, (1, 5))),

                #  ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                #  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))
                ]

marker_style = ["x", "+", "o", "s"]

def auroc_curves(test_dir, model_paths, model_labels, save_path, target_class, greyscale_support=True, kappa_vals=False, confusion=False, default_col=False):
    # test loader set up
    test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
    num_classes=len(test_dataset.classes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False)

    # colors, ref: https://stackoverflow.com/a/33905962/7521428
    start = 0.1
    stop = 0.9
    number_of_colors = len(model_labels)
    cm_subsection = np.linspace(start, stop, number_of_colors) 
    colors = [ cm.nipy_spectral(x) for x in cm_subsection ]

    #linestyles from global var
    # linestyle_tuple = linestyle_tuple

    plt.figure(figsize=(5.0, 3.6), dpi=200)
    # plot a line for each model
    for model_idx, model_path in enumerate(model_paths):
        model = torch.load(model_path)

        total_one_hot_label, total_output = [], []

        if kappa_vals or confusion:
            confusion_matrix = np.zeros((num_classes, num_classes))

        for i, (image, label, patient_id) in enumerate(test_loader):
            input = image.cuda()
            target = label.cuda()

            grad_req = torch.no_grad()
            with grad_req:
                if 'vanilla' in model_path:
                    output = model.forward(input)
                else:
                    output, min_distances, _ = model.forward(input)

                # one hot label for AUC
                one_hot_label = np.zeros(shape=(len(target), num_classes))
                for k in range(len(target)):
                    one_hot_label[k][target[k].item()] = 1

                prob = torch.nn.functional.softmax(output, dim=1)
                total_output.extend(prob.data.cpu().numpy())
                total_one_hot_label.extend(one_hot_label)

                # confusion matrix
                if kappa_vals or confusion:
                    _, predicted = torch.max(output.data, 1)
                    for t_idx, t in enumerate(label):
                        confusion_matrix[predicted[t_idx]][t] += 1 #row is predicted, col is true
                    if kappa_vals:
                        kappa, oneclass_confusion_matrix = calc_kappa(confusion_matrix, target_class)
                    
        
        if confusion:
            print('The confusion matrix for model {0} on class {1} is: \n{2}'.format(model_path, target_class, confusion_matrix))
        if kappa_vals:
            print('The kappa value for model {0} on class {1} is: {2}. Matrix: \n{3}'.format(model_path, target_class, kappa, oneclass_confusion_matrix))

        total_output = np.array(total_output)
        total_one_hot_label = np.array(total_one_hot_label)
        # print(total_output[:5])
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(total_one_hot_label[:, i], total_output[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        print("Plotting model {} of {}.".format(model_idx+1, len(model_paths)))
        lw = 1
        label = '{} (area = {:.2f})'.format(model_labels[model_idx], roc_auc[target_class])
        if (greyscale_support) and (not default_col):
            plt.plot(fpr[target_class], tpr[target_class], color=colors[model_idx],
                    lw=lw, label=label, linestyle=linestyle_tuple[model_idx][1])
        elif (not greyscale_support) and (not default_col):
            plt.plot(fpr[target_class], tpr[target_class], color=colors[model_idx],
                    lw=lw, label=label, linestyle='solid')
        elif (not greyscale_support) and (default_col):
            plt.plot(fpr[target_class], tpr[target_class],
                    lw=lw, label=label, linestyle='solid')
        else:
            plt.plot(fpr[target_class], tpr[target_class],
                    lw=lw, label=label, linestyle=linestyle_tuple[model_idx][1])
    
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.title('AUC')
    plt.legend(loc="lower right")
    print('Saving to:', save_path)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def comparison_plot(test_dir, model_paths, legend_labels, plot_xs, save_path, greyscale_support=False, y_func=[(auroc, 'AUROC')]):
    if len(y_func) == 1:
        y_func = y_func[0]
        assert len(model_paths) == len(legend_labels)
        assert len(model_paths[0]) == len(plot_xs)

        # colors, ref: https://stackoverflow.com/a/33905962/7521428
        start = 0.1
        stop = 0.9
        number_of_colors = len(legend_labels)
        cm_subsection = np.linspace(start, stop, number_of_colors) 
        colors = [ cm.nipy_spectral(x) for x in cm_subsection ]

        #linestyles from global var
        # linestyle_tuple = linestyle_tuple

        # graphing
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        for group_idx, model_group in enumerate(model_paths):
            xs = [item+1 for item in range(len(plot_xs))]
            ys = []
            for model_path in model_group:
                auroc_ = y_func[0](test_dir, model_path)
                ys.append(auroc_)

            if greyscale_support:
                ax.plot(xs, ys, label=legend_labels[group_idx], 
                        linewidth=1, linestyle=linestyle_tuple[group_idx][1], marker="None")
            else:
                ax.plot(xs, ys, label=legend_labels[group_idx], 
                        linewidth=1.5, linestyle="solid", marker="None")

        # ax.set_xscale('log')  
        # fig.canvas.draw()
        ax.set_xticks(xs)
        ax.set_xticklabels(plot_xs)
        # ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) # https://stackoverflow.com/questions/14530113/set-ticks-with-logarithmic-scale
        # ax.set_xlim([0.5, 4.5])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Fine annotation coefficient')
        ax.set_ylabel(y_func[1])
        # plt.title('AUC')
        if len(legend_labels) > 1:
            ax.legend(loc="lower right")
        print('Saving to:', save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)
    else:
        assert len(model_paths) == 1
        assert len(legend_labels) == 1
        assert len(model_paths[0]) == len(plot_xs)

        # colors, ref: https://stackoverflow.com/a/33905962/7521428
        start = 0.1
        stop = 0.9
        number_of_colors = len(y_func)
        cm_subsection = np.linspace(start, stop, number_of_colors) 
        colors = [ cm.nipy_spectral(x) for x in cm_subsection ]

        #linestyles from global var
        # linestyle_tuple = linestyle_tuple

        # graphing
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        for y_func_idx, (y_function, y_function_name) in enumerate(y_func):
            xs = [item+1 for item in range(len(plot_xs))]
            ys = []
            for group_idx, model_group in enumerate(model_paths):
                for model_path in model_group:
                    auroc_ = y_function(test_dir, model_path)
                    ys.append(auroc_)

            if greyscale_support:
                ax.plot(xs, ys, label=y_function_name, 
                        linewidth=1, linestyle=linestyle_tuple[y_func_idx][1], marker="None")
            else:
                ax.plot(xs, ys, label=y_function_name, 
                        linewidth=1.5, linestyle="solid", marker="None")

        # ax.set_xscale('log')  
        # fig.canvas.draw()
        ax.set_xticks(xs)
        ax.set_xticklabels(plot_xs)
        # ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) # https://stackoverflow.com/questions/14530113/set-ticks-with-logarithmic-scale
        # ax.set_xlim([0.5, 4.5])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Fine annotation coefficient')
        ax.set_ylabel('AUROC / Lesion Activation Precision')
        # plt.title('AUC')
        ax.legend(loc="lower right")
        print('Saving to:', save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)

if __name__=="__main__":
    
    plot = 4 # choose which plot you want

    if plot==1:
        comparison_plot(test_dir='/usr/xtmp/mammo/Lo1136i_with_fa/validation/',
                        model_paths=[["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.0_random=4/100_9push0.4500.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.001_random=4/90_1push0.8786.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.002_random=4/80_9push0.5317.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.005_random=4/50_5push0.5000.pth"], 
                                    ["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.0_random=4/100_9push0.9430.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.001_random=4/100_9push0.8982.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.002_random=4/100_9push0.3970.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.005_random=4/100_9push0.5317.pth"],
                                    ["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.0_random=4/100_9push0.9245.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.001_random=4/100_9push0.9146.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.002_random=4/100_9push0.8573.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.005_random=4/100_9push0.4471.pth"]
                                    ],
                        legend_labels=["maxpool (top 0.5%)", 
                                        "top 2%", 
                                        "top 5%"
                                        ],
                        plot_xs=[0,
                                0.001,
                                0.002,
                                0.005
                                ],
                        save_path="./visualizations/auroc_over_fa.png",
                        y_func=[(auroc, 'AUROC')]
                        )

    if plot==2:
        comparison_plot(test_dir='/usr/xtmp/mammo/Lo1136i_with_fa/validation/',
                        model_paths=[["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.0_random=4/100_9push0.4500.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.001_random=4/90_1push0.8786.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.002_random=4/80_9push0.5317.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.005_random=4/50_5push0.5000.pth"], 
                                    ["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.0_random=4/100_9push0.9430.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.001_random=4/100_9push0.8982.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.002_random=4/100_9push0.3970.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.005_random=4/100_9push0.5317.pth"],
                                    ["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.0_random=4/100_9push0.9245.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.001_random=4/100_9push0.9146.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.002_random=4/100_9push0.8573.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.005_random=4/100_9push0.4471.pth"]
                                    ],
                        legend_labels=["maxpool (top 0.5%)", 
                                        "top 2%", 
                                        "top 5%"
                                        ],
                        plot_xs=[0,
                                0.001,
                                0.002,
                                0.005
                                ],
                        save_path="./visualizations/hp_over_fa.png",
                        y_func=[(hp, 'Lesion Activation Precision')]
                        )

    if plot==3:
        comparison_plot(test_dir='/usr/xtmp/mammo/Lo1136i_finer/by_margin/validation/',
                        model_paths=[["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.0_random=4/100_9push0.4500.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.001_random=4/90_1push0.8786.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.002_random=4/80_9push0.5317.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=1_fa=0.005_random=4/50_5push0.5000.pth"], 
                                    ["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.0_random=4/100_9push0.9430.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.001_random=4/100_9push0.8982.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.002_random=4/100_9push0.3970.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=3_fa=0.005_random=4/100_9push0.5317.pth"],
                                    ["/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.0_random=4/100_9push0.9245.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.001_random=4/100_9push0.9146.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.002_random=4/100_9push0.8573.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0112_topkk=9_fa=0.005_random=4/100_9push0.4471.pth"]
                                    ],
                        legend_labels=["maxpool (top 0.5%)", 
                                        "top 2%", 
                                        "top 5%"
                                        ],
                        plot_xs=[0,
                                0.001,
                                0.002,
                                0.005
                                ],
                        save_path="./visualizations/finehp_over_fa.png",
                        y_func=[(hp, 'Fine Activation Precision')]
                        )
    
    if plot==4:
        for target_class in range(3):
            auroc_curves(test_dir='/usr/xtmp/mammo/Lo1136i/test/',
                        model_paths=["/usr/xtmp/mammo/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/50_4push0.9546.pth",
                                    "/usr/xtmp/mammo/saved_models/vgg16/0125_topkk=1_fa=0.0_random=4/50_5push0.9209.pth",
                                    "/usr/xtmp/mammo/saved_models/vanilla/0125_vanilla_3margin_vgg16_latent512_baseline3_random=4/0.9384582045743842_at_epoch_136"
                                    ],
                        model_labels=["IAIA-BL (our model)",
                                        "Original ProtoPNet",
                                        "Blackbox VGG-16"
                                        ],
                        save_path=f'./visualizations/roc_curves_ep50_defaultcol_class{target_class}.png',
                        target_class=target_class,
                        greyscale_support=False,
                        kappa_vals=True,
                        confusion=True,
                        default_col=True
                        )