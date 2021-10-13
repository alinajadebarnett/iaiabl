import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from PIL import Image
from dataHelper import DatasetFolder, DatasetFolder_WithReplacement
from helpers import silent_print
import re
import numpy as np
import os
import train_and_test as tnt
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from vanilla_vgg import Vanilla_VGG
from delong import print_delong_AUROCs
from delong_2 import delong_roc_test
import argparse

def calc_kappa(confusion_matrix, target_class):
    num_classes = confusion_matrix.shape[0]
    oneclass_confusion_matrix = np.zeros((2,2))
    for pred in range(num_classes):
        for truth in range(num_classes):
            if pred==target_class and truth==target_class:
                oneclass_confusion_matrix[0][0] = confusion_matrix[pred][truth] #TP
            if pred!=target_class and truth!=target_class:
                oneclass_confusion_matrix[1][1] += confusion_matrix[pred][truth] #TN
            if pred!=target_class and truth==target_class:
                oneclass_confusion_matrix[1][0] += confusion_matrix[pred][truth] #FN
            if pred==target_class and truth!=target_class:
                oneclass_confusion_matrix[0][1] += confusion_matrix[pred][truth] #FP

    a, b, c, d = oneclass_confusion_matrix[0][0], oneclass_confusion_matrix[0][1], oneclass_confusion_matrix[1][0], oneclass_confusion_matrix[1][1]
    a, b, c, d = float(a), float(b), float(c), float(d)
    po = (a + d) / (a+b+c+d)
    pe = ( (a+b)*(a+c) + (c+d)*(b+d) ) / (a+b+c+d)**2
    kappa = (po-pe) / (1-pe)

    return kappa, oneclass_confusion_matrix

def accu(test_dir, model_path, save_logits=False, verbose=False, topk_k=None):
    ''' retrieves the value of the current test function of tnt, not neccessarily accuracy'''
    # load the model
    check_test_accu = True

    ppnet = torch.load(model_path)
    if topk_k is not None:
        ppnet.set_topk_k(topk_k)
        if verbose:
            print(f'Set the topk_k to: {ppnet.topk_k}')
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    class_specific = True

    # load the test data and check test accuracy
    if check_test_accu:
        test_batch_size = 100

        test_dataset = DatasetFolder(
            test_dir,
            augmentation=False,
            loader=np.load,
            extensions=("npy",),
            transform=transforms.Compose([
                torch.from_numpy,
            ])
            )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True,
            num_workers=4, pin_memory=False)
        if verbose:
            print('test set size: {0}'.format(len(test_loader.dataset)))

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=print, save_logits=save_logits)
        if verbose:
            print(accu)

    return accu

def auroc(test_dir, model_path, verbose=False, topk_k=None):
    # load the model and test dataloader

    ppnet = torch.load(model_path)
    if topk_k is not None:
        ppnet.set_topk_k(topk_k)
        if verbose:
            print(f'Set the topk_k to: {ppnet.topk_k}')
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    class_specific = True

    test_batch_size = 75

    test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    if verbose:
        print('test set size: {0}'.format(len(test_loader.dataset)))

    # calc
    auroc = calc_auroc(model=ppnet, test_loader=test_loader, num_classes=len(test_dataset.classes))
    if verbose:
        print(auroc)

    return auroc

def calc_auroc(model, test_loader, num_classes, per_class=False, kappa_vals=False, vanilla=False):
    total_one_hot_label, total_output = [], []
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for i, (image, label, patient_id) in enumerate(test_loader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.no_grad()
        with grad_req:
            if vanilla:
                output = model(input)
            else:
                output, _, _ = model(input)

            # one hot label for AUC
            one_hot_label = np.zeros(shape=(len(target), num_classes))
            for k in range(len(target)):
                one_hot_label[k][target[k].item()] = 1

            prob = torch.nn.functional.softmax(output, dim=1)
            total_output.extend(prob.data.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)

            if kappa_vals:
                _, predicted = torch.max(output.data, 1)
                for t_idx, t in enumerate(label):
                    confusion_matrix[predicted[t_idx]][t] += 1 #row is predicted, col is true
            kappa_val_allclass = cohen_kappa_score(predicted.cpu().numpy(), label.cpu().numpy())

    if kappa_vals:
        kappas_dict = dict()
        for target_class in range(num_classes):
            kappa, _ = calc_kappa(confusion_matrix, target_class)
            kappas_dict[target_class] = kappa
        kappas_dict[num_classes] = kappa_val_allclass

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

    per_class_aurocs = list(roc_auc.values())
    auroc = np.mean(per_class_aurocs)

    if per_class and kappa_vals:
        return auroc, roc_auc, kappas_dict

    if per_class:
        return auroc, roc_auc

    return auroc

def bootstrapped_test(test_dir, model_path, verbose=False, vanilla=False):
    # load model
    save_logits = False
    ppnet = torch.load(model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # this dataset class grabs a random sample with replacement instead of all samples iteratively
    test_dataset = DatasetFolder_WithReplacement(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
        )

    # set the bootstrap parameters
    test_batch_size = len(test_dataset.samples) # we decided to use a sample size equal to the size of the test set.
    CI = 0.95 # confidence interval
    num_iterations = 5000 
    accus = [0]*num_iterations # doing this instead of a list append marginally improves computational efficiency
    aurocs = [0]*num_iterations
    class_0_aurocs = [0]*num_iterations
    class_1_aurocs = [0]*num_iterations
    class_2_aurocs = [0]*num_iterations
    class_0_kappas = [0]*num_iterations
    class_1_kappas = [0]*num_iterations
    class_2_kappas = [0]*num_iterations
    all_class_kappas = [0]*num_iterations

    for iteration_index in range(num_iterations):

        # to look for convergence in slurm for early stopping possibility
        if verbose and iteration_index > 0 and iteration_index % 25 == 0:
            accus_ = accus[:iteration_index]
            aurocs_ = aurocs[:iteration_index]
            accu_mean = np.mean(np.asarray(accus_))
            accu_std = np.std(np.asarray(accus_))
            lower, upper = 100 * ( (1.0 - CI)/2. ), 100 *  ( 1.0 - ((1.0 - CI)/2.) )
            accu_lower, accu_upper = np.percentile(accus_, [lower, upper])
            auroc_mean = np.mean(np.asarray(aurocs_))
            auroc_lower, auroc_upper = np.percentile(aurocs_, [lower, upper])

            print(f'{iteration_index} of {num_iterations}:')
            print(f"So far, accuracy mean is {accu_mean} and {CI*100}% confidence iterval accuracy is {accu_lower} to {accu_upper}.")
            print(f"So far, AUROC mean is {auroc_mean} and {CI*100}% confidence iterval AUROC is [{auroc_lower}, {auroc_upper}].")

        # test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True,
            num_workers=4, pin_memory=False)
        
        if not vanilla:
            # calc accu
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=silent_print, save_logits=save_logits)

        # calc aurocs, kappas
        auroc, auroc_dict, kappa_dict = calc_auroc(model=ppnet, test_loader=test_loader, num_classes=len(test_dataset.classes), per_class=True, kappa_vals=True, vanilla=vanilla)

        # add to list
        aurocs[iteration_index] = auroc
        if not vanilla:
            accus[iteration_index] = accu
        class_0_aurocs[iteration_index] = auroc_dict[0]
        class_1_aurocs[iteration_index] = auroc_dict[1]
        class_2_aurocs[iteration_index] = auroc_dict[2]
        class_0_kappas[iteration_index] = kappa_dict[0]
        class_1_kappas[iteration_index] = kappa_dict[1]
        class_2_kappas[iteration_index] = kappa_dict[2]
        all_class_kappas[iteration_index] = kappa_dict[3]

    # calc stats
    print(f"Model {model_path} tested on {test_dir}:")

    if vanilla:
        vois = zip([aurocs, class_0_aurocs, class_1_aurocs, class_2_aurocs, \
                class_0_kappas, class_1_kappas, class_2_kappas, all_class_kappas],\
                ['AUROC', 'Circumscribed AUROC', 'Indistinct AUROC', 'Spiculated AUROC', \
                'Circumscribed Kappa', 'Indistinct Kappa', 'Spiculated Kappa', 'All-class Kappa'])
    else:
        vois = zip([aurocs, accus, class_0_aurocs, class_1_aurocs, class_2_aurocs, \
                class_0_kappas, class_1_kappas, class_2_kappas, all_class_kappas],\
                ['AUROC', 'Test function', 'Circumscribed AUROC', 'Indistinct AUROC', 'Spiculated AUROC', \
                'Circumscribed Kappa', 'Indistinct Kappa', 'Spiculated Kappa', 'All-class Kappa'])
    lower, upper = 100 * ( (1.0 - CI)/2. ), 100 *  ( 1.0 - ((1.0 - CI)/2.) )
    for valueofinterest, valueofinterest_str in vois:
        voi_mean = np.mean(np.asarray(valueofinterest))
        voi_std = np.std(np.asarray(valueofinterest))
        voi_lower, voi_upper = np.percentile(valueofinterest, [lower, upper])
        print(f"Final mean {valueofinterest_str} is {voi_mean}, std {voi_std} {CI*100}% confidence iterval accuracy is {voi_lower} to {voi_upper}.")

def confusion_matrix(model_path, data_path, num_classes=3):
    # predicted * true
    model = torch.load(model_path)
    test_dataset = DatasetFolder(
        data_path,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False)

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i, (image, label, patient_id) in enumerate(test_loader):
        input = image.cuda()\

        grad_req = torch.no_grad()
        with grad_req:
            output, min_distances, _ = model(input)
            res = torch.argmax(output, dim=1)
            for j in range(len(res)):
                confusion_matrix[res[j]][label[j]] += 1 # cm[predicted][true] += 1

    print("confusion matrix is", confusion_matrix)

def delong_it(test_dir, model_path, vanilla=False, save_ys=None):
    kappa_vals = True

    model = torch.load(model_path)
    test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False)

    num_classes = len(test_dataset.classes)

    total_one_hot_label, total_output = [], []
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for i, (image, label, patient_id) in enumerate(test_loader):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:
            if vanilla:
                output = model(input)
            else:
                output, _, _ = model(input)

            # one hot label for AUC
            one_hot_label = np.zeros(shape=(len(target), num_classes))
            for k in range(len(target)):
                one_hot_label[k][target[k].item()] = 1

            prob = torch.nn.functional.softmax(output, dim=1)
            total_output.extend(prob.data.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)

            if kappa_vals:
                _, predicted = torch.max(output.data, 1)
                for t_idx, t in enumerate(label):
                    confusion_matrix[predicted[t_idx]][t] += 1 #row is predicted, col is true
            kappa_val_allclass = cohen_kappa_score(predicted.cpu().numpy(), label.cpu().numpy())

    if kappa_vals:
        kappas_dict = dict()
        for target_class in range(num_classes):
            kappa, _ = calc_kappa(confusion_matrix, target_class)
            kappas_dict[target_class] = kappa
        kappas_dict[num_classes] = kappa_val_allclass

    total_output = np.array(total_output)
    total_one_hot_label = np.array(total_one_hot_label)
    if save_ys is not None:
        np.save(save_ys + '0225_outputs.npy', total_output[:, i], allow_pickle=False)
        np.save(save_ys + '0225_labels.npy', total_one_hot_label[:, i], allow_pickle=False)
        print(f"Saved predictions and labels to {save_ys}")
    # print(total_output[:5])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_by_class = []
    ci_by_class = []
    auc_cov_by_class = []
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(total_one_hot_label[:, i], total_output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"Class {i}: ")
        auc_by_class_, auc_cov_by_class_, ci_by_class_ = print_delong_AUROCs(total_one_hot_label[:, i], total_output[:, i])
        auc_by_class.append(auc_by_class_)
        ci_by_class.append(ci_by_class_)
        auc_cov_by_class.append(auc_by_class_)

    print("")
    weights = [25, 34, 19]
    weights = np.asarray(weights)
    img_weighted_avg_auroc = np.average(np.asarray(auc_by_class), weights=weights)
    ci_by_class = np.asarray(ci_by_class)
    # print(f'ci by class array shape: {ci_by_class.shape} ci by class array:\n{ci_by_class}')
    # print(f'middle of cis: \n{0.5*(ci_by_class[:,1]+ci_by_class[:,0])}')
    # print(f'halfrange cis: \n{0.5*(ci_by_class[:,1]-ci_by_class[:,0])}')

    new_halfrange_ci = (1. / np.sum(weights))**2 * np.sum(np.square(np.dot(0.5*(ci_by_class[:,1]-ci_by_class[:,0]), weights)))
    new_halfrange_ci = np.sqrt(new_halfrange_ci)

    print(f'Image weighted all classes: {img_weighted_avg_auroc} CI [{img_weighted_avg_auroc - new_halfrange_ci} {img_weighted_avg_auroc + new_halfrange_ci}].')
    

    per_class_aurocs = list(roc_auc.values())
    auroc = np.mean(per_class_aurocs)

    print("Kappas: ", kappas_dict)

def delong_compare(test_dir, model_path1, model_path2, vanilla=[False,False]):
    model1 = torch.load(model_path1)
    model2 = torch.load(model_path2)

    test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False,
        num_workers=4, pin_memory=False)
    num_classes = len(test_dataset.classes)

    m_total_one_hot_label, m_total_output = dict(), dict()
    for model_idx, model in enumerate([model1, model2]):
        total_one_hot_label, total_output = [], []

        for i, (image, label, patient_id) in enumerate(test_loader):
            input = image.cuda()
            target = label.cuda()

            grad_req = torch.no_grad()
            with grad_req:
                if vanilla[model_idx]:
                    output = model(input)
                else:
                    output, _, _ = model(input)

                # one hot label for AUC
                one_hot_label = np.zeros(shape=(len(target), num_classes))
                for k in range(len(target)):
                    one_hot_label[k][target[k].item()] = 1

                prob = torch.nn.functional.softmax(output, dim=1)
                total_output.extend(prob.data.cpu().numpy())
                total_one_hot_label.extend(one_hot_label)

        m_total_output[model_idx] = np.array(total_output)
        m_total_one_hot_label[model_idx] = np.array(total_one_hot_label)

    # Compute and print p-val for each class
    print(f'Comparing models {model_path1} and {model_path2} on the dataset {test_dir}:')
    for i in range(num_classes):
        p_value = delong_roc_test(m_total_one_hot_label[1][:, i], m_total_output[0][:, i], m_total_output[1][:, i])
        print(f'\tFor class {i}, p_value of the two model coming from different distributions is {np.power(10,p_value)}.')

def Welshs_t_test(mean1, mean2, std1, std2, N1, N2):
    # source https://en.wikipedia.org/wiki/Welch%27s_t-test
    t = (mean1 - mean2) / (np.sqrt( ((std1**2)/N1) + ((std2**2)/N2) ))

    mu = ( ((std1**2)/N1) + ((std2**2)/N2) )**2 / ( (std1**4)/((N1**2)*(N1-1)) + (std2**4)/((N2**2)*(N2-1)) )

    return t, mu

if __name__=="__main__":
    which_to_run = 5

    if which_to_run==0:
        test_dir='/usr/xtmp/mammo/Lo1136i/train_plus_val_augmented/'
        model_path='/usr/xtmp/mammo/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/pruned_prototypes_epoch50_k6_pt3/50_4prune0.9533.pth'
        print(accu(test_dir, model_path, verbose=True))

        print("Trained on top 2%, test with 2% model.")
        print(auroc(test_dir, model_path, verbose=True, topk_k=3))

        print("Trained on top 2%, test with 5% model.")
        print(auroc(test_dir, model_path, verbose=True, topk_k=9))

    if which_to_run==1:
        test_dir='/usr/xtmp/mammo/Lo1136i/test/'
        model_path='/usr/xtmp/mammo/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/pruned_prototypes_epoch50_k6_pt3/50_4prune0.9533.pth'
        print("\nPruned IAIA-BL Delong.")
        print(delong_it(test_dir, model_path, vanilla=False))

        test_dir='/usr/xtmp/mammo/Lo1136i/test/'
        model_path='/usr/xtmp/mammo/saved_models/vgg16/0125_topkk=1_fa=0.0_random=4/50_5push0.9209.pth'
        print("\nOriginal ProtoPNet Delong.")
        print(delong_it(test_dir, model_path, vanilla=False))

        test_dir='/usr/xtmp/mammo/Lo1136i/test/'
        model_path='/usr/xtmp/mammo/saved_models/vanilla/0125_vanilla_3margin_vgg16_latent512_baseline3_random=4/0.9384582045743842_at_epoch_136'
        print("\nVanilla Delong.")
        print(delong_it(test_dir, model_path, vanilla=True))

    if which_to_run==2:
        test_dir='/usr/xtmp/mammo/Lo1136i/bymal/test/'
        model_path='/usr/xtmp/mammo/saved_models/vanilla/0202_vanilla_2mal_vgg16_latent512_random=4/0.8686868686868687_at_epoch_20'
        print("\nEnd-to-end Mal Vanilla Delong.")
        print(delong_it(test_dir, model_path, vanilla=True, save_ys='./logit_csvs/'))

        print("Bootstrapped.")
        print(bootstrapped_test(test_dir, model_path, verbose=False, vanilla=True))

    if which_to_run==3:
        print("######### original protop ##########")
        test_dir='/usr/xtmp/mammo/Lo1136i/test/'
        model_path="/usr/xtmp/mammo/saved_models/vgg16/0125_topkk=1_fa=0.0_random=4/100_0push0.9194.pth"
        # print("Check once.")
        # print(accu(test_dir, model_path, save_logits=False, verbose=False))
        print("Bootstrapped.")
        print(bootstrapped_test(test_dir, model_path, verbose=False))

    if which_to_run==4:
        delong_compare(test_dir='/usr/xtmp/mammo/Lo1136i/test/',\
                        model_path1='/usr/xtmp/mammo/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/pruned_prototypes_epoch50_k6_pt3/50_4prune0.9533.pth',
                        model_path2='/usr/xtmp/mammo/saved_models/vgg16/0125_topkk=1_fa=0.0_random=4/50_5push0.9209.pth',\
                        vanilla=[False, False])

        delong_compare(test_dir='/usr/xtmp/mammo/Lo1136i/test/',\
                        model_path1='/usr/xtmp/mammo/saved_models/vgg16/0129_pushonall_topkk=9_fa=0.001_random=4/pruned_prototypes_epoch50_k6_pt3/50_4prune0.9533.pth',
                        model_path2='/usr/xtmp/mammo/saved_models/vanilla/0125_vanilla_3margin_vgg16_latent512_baseline3_random=4/0.9384582045743842_at_epoch_136',\
                        vanilla=[False, True])
        
        delong_compare(test_dir='/usr/xtmp/mammo/Lo1136i/test/',\
                        model_path1='/usr/xtmp/mammo/saved_models/vgg16/0125_topkk=1_fa=0.0_random=4/50_5push0.9209.pth',
                        model_path2='/usr/xtmp/mammo/saved_models/vanilla/0125_vanilla_3margin_vgg16_latent512_baseline3_random=4/0.9384582045743842_at_epoch_136',\
                        vanilla=[False, True])
    
    if which_to_run==5:
        mean1 = 0.951
        mean2 = 0.911
        mean3 = 0.947
        # ref https://handbook-5-1.cochrane.org/chapter_7/7_7_3_2_obtaining_standard_deviations_from_standard_errors_and.htm
        std1 = -np.sqrt(78)*(0.905 - 0.996)/3.92
        std2 = -np.sqrt(78)*(0.905 - 0.996)/3.92
        std3 = -np.sqrt(78)*(0.905 - 0.996)/3.92
        N1 = 3
        N2 = 3
        N3 = 3
        print(Welshs_t_test(mean1, mean2, std1, std2, N1, N2))
        print(Welshs_t_test(mean1, mean3, std1, std3, N1, N3))
        print(Welshs_t_test(mean3, mean2, std3, std2, N3, N2))

        
