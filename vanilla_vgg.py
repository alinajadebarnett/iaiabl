import matplotlib
import matplotlib.pyplot as plt
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
import argparse
import torch.nn as nn
from dataHelper import DatasetFolder
from torchvision import transforms
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import random

# build model
class Vanilla_VGG(nn.Module):
    def __init__(self, myfeatures, num_classes):
        super(Vanilla_VGG, self).__init__()

        self.features = myfeatures
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=0)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str)
    parser.add_argument("-train_dir", type=str, default="/usr/project/xtmp/mammo/")
    parser.add_argument("-test_dir", type=str, default="/usr/project/xtmp/mammo/")
    parser.add_argument("-name", type=str)
    parser.add_argument("-lr", type=lambda x: float(x))
    parser.add_argument("-wd", type=lambda x: float(x))
    parser.add_argument("-num_classes", type=lambda x: int(x))
    args = parser.parse_args()
    model_name = args.model
    train_dir = args.train_dir
    test_dir = args.test_dir
    task_name = args.name
    num_classes = args.num_classes

    save_loc = '/usr/xtmp/mammo/saved_models/vanilla/'
    os.makedirs(save_loc + task_name, exist_ok = True)
    lr = args.lr
    wd = args.wd
    print(wd, lr)
    print("Saving to: ", save_loc + task_name)

    random_seed_number = 12
    print("Random seed: ", random_seed_number)
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    writer = SummaryWriter()

    base_architecture_to_features = {'vgg11': vgg11_features,
                                    'vgg11_bn': vgg11_bn_features,
                                    'vgg13': vgg13_features,
                                    'vgg13_bn': vgg13_bn_features,
                                    'vgg16': vgg16_features,
                                    'vgg16_bn': vgg16_bn_features,
                                    'vgg19': vgg19_features,
                                    'vgg19_bn': vgg19_bn_features}

    features = base_architecture_to_features[model_name](pretrained=True)

    model = Vanilla_VGG(features, num_classes)

    # load data
    # train set
    train_dataset = DatasetFolder(
        train_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        target_size=(224, 224),
        transform = transforms.Compose([
            torch.from_numpy,
        ]))
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False)

    # test set
    test_dataset =DatasetFolder(
        test_dir,
        loader=np.load,
        target_size=(224, 224),
        extensions=("npy",),
        transform = transforms.Compose([
            torch.from_numpy,
        ]))
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False,
        num_workers=4, pin_memory=False)


    # start training
    epochs = 250

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    device = torch.device("cuda")
    model.to(device)

    train_losses = []
    test_losses = []
    train_auc = []
    test_auc = []
    curr_best = 0


    for epoch in range(epochs):
        # train
        confusion_matrix = np.zeros((num_classes, num_classes))
        total_output = []
        total_one_hot_label  = []
        running_loss = 0
        model.train()
        for inputs, labels, id in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            one_hot_label = np.zeros(shape=(len(labels), num_classes))
            for k in range(len(labels)):
                one_hot_label[k][labels[k].item()] = 1
            # roc_auc_score()
            total_output.extend(logps.cpu().detach().numpy())
            total_one_hot_label.extend(one_hot_label)
            # confusion matrix
            _, predicted = torch.max(logps.data, 1)
            for t_idx, t in enumerate(labels):
                confusion_matrix[predicted[t_idx]][t] += 1 #row is predicted, col is true
                # if predicted[t_idx] == t:  # correct label
                #     confusion_matrix[t][t] += 1
                # elif t == 0 and predicted[t_idx] == 1:
                #     confusion_matrix[1] += 1  # false positives
                # elif t == 1 and predicted[t_idx] == 0:
                #     confusion_matrix[2] += 1  # false negative
                # else:
                #     confusion_matrix[3] += 1

        auc_score = roc_auc_score(np.array(total_one_hot_label), np.array(total_output))

        train_losses.append(running_loss / len(trainloader))
        train_auc.append(auc_score)
        print("=======================================================")
        print("\t at epoch {}".format(epoch))
        print("\t train loss is {}".format(train_losses[-1]))
        print("\t train auc is {}".format(auc_score))
        print('\tthe confusion matrix is: \n{0}'.format(confusion_matrix))
        # test
        confusion_matrix = np.zeros((num_classes, num_classes))
        test_loss = 0
        total_output = []
        total_one_hot_label  = []
        model.eval()
        with torch.no_grad():
            for inputs, labels, id in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()
                one_hot_label = np.zeros(shape=(len(labels), num_classes))
                for k in range(len(labels)):
                    one_hot_label[k][labels[k].item()] = 1
                # roc_auc_score()
                total_output.extend(logps.cpu().numpy())
                total_one_hot_label.extend(one_hot_label)
                # confusion matrix
                _, predicted = torch.max(logps.data, 1)
                for t_idx, t in enumerate(labels):
                    confusion_matrix[predicted[t_idx]][t] += 1 #row is predicted, col is true
                    # if predicted[t_idx] == t and predicted[t_idx] == 1:  # true positive
                    #     confusion_matrix[0] += 1
                    # elif t == 0 and predicted[t_idx] == 1:
                    #     confusion_matrix[1] += 1  # false positives
                    # elif t == 1 and predicted[t_idx] == 0:
                    #     confusion_matrix[2] += 1  # false negative
                    # else:
                    #     confusion_matrix[3] += 1
        auc_score = roc_auc_score(np.array(total_one_hot_label), np.array(total_output))
        test_losses.append(test_loss / len(testloader))
        test_auc.append(auc_score)
        print("===========================")
        if auc_score > curr_best:
            curr_best = auc_score
        print("\t test loss is {}".format(test_losses[-1]))
        print("\t test auc is {}".format(auc_score))
        print("\t current best is {}".format(curr_best))
        print('\tthe confusion matrix is: \n{0}'.format(confusion_matrix))
        print("=======================================================")

        # save model
        if auc_score > 0:
            torch.save(model, save_loc + task_name + "/" + str(auc_score) + "_at_epoch_" + str(epoch))

        # plot graphs
        plt.plot(train_losses, "b", label="train")
        plt.plot(test_losses, "r", label="test")
        #plt.ylim(0, 4)
        plt.legend()
        plt.savefig(save_loc + task_name + '/train_test_loss_vanilla' + ".png")
        plt.close()

        plt.plot(train_auc, "b", label="train")
        plt.plot(test_auc, "r", label="test")
        #plt.ylim(0.4, 1)
        plt.legend()
        plt.savefig(save_loc + task_name + '/train_test_auc_vanilla' + ".png")
        plt.close()

    writer.close()

if __name__=="__main__":
    main()