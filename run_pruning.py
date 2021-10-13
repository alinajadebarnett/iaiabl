import os
import shutil
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse
from dataHelper import DatasetFolder
from helpers import makedir
import model
import last_layer
import push
import prune
import find_nearest
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-train_dir', nargs=1, type=str)
parser.add_argument('-test_dir', nargs=1, type=str)
parser.add_argument('-push_dir', nargs=1, type=str)
args = parser.parse_args()

optimize_last_layer = True

proto_to_keep = [0,1,5,9,10,11] #for model /usr/xtmp/mammo/saved_models/vgg16/0125_topkk=9_fa=0.001_random=4/50_9push0.9645.pth
# pruning parameters

k = 5
prune_threshold = 3

original_model_dir = args.modeldir[0]
original_model_name = args.model[0]
train_dir, test_dir, train_push_dir = args.train_dir[0], args.test_dir[0], args.push_dir[0]

need_push = ('nopush' in original_model_name)
if need_push:
    assert(False) # pruning must happen after push
else:
    epoch = original_model_name.split('push')[0]

if '_' in epoch:
    epoch = int(epoch.split('_')[0])
else:
    epoch = int(epoch)

model_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch,
                                          k,
                                          prune_threshold))
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'prune.log'))

ppnet = torch.load(original_model_dir + original_model_name)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

train_batch_size = 80
test_batch_size = 100
img_size = 224
train_push_batch_size = 80

# all datasets
# train set
train_dataset = DatasetFolder(
    train_dir,
    augmentation=False,
    loader=np.load,
    extensions=("npy",),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

# push set
train_push_dataset = DatasetFolder(
    root = train_push_dir,
    loader = np.load,
    extensions=("npy",),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# test set
test_dataset =DatasetFolder(
    test_dir,
    loader=np.load,
    extensions=("npy",),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

    
log('push set size: {0}'.format(len(train_push_loader.dataset)))

tnt.test(model=ppnet_multi, dataloader=test_loader,
         class_specific=class_specific, log=log)
print(find_nearest.find_k_nearest_patches_to_prototypes(dataloader=train_push_loader,
                                                prototype_network_parallel=ppnet_multi,
                                                k=5,
                                                preprocess_input_function=preprocess_input_function,
                                                full_save=False,
                                                log=log))
print("last layer trasnpose: \n", last_layer.show_last_layer_connections_T(ppnet))

# prune prototypes
log('========================================================prune======================================================')
prune.prune_prototypes(dataloader=train_push_loader,
                       prototype_network_parallel=ppnet_multi,
                       k=k,
                       prune_threshold=prune_threshold,
                       preprocess_input_function=preprocess_input_function, # normalize
                       original_model_dir=original_model_dir,
                       epoch_number=epoch,
                       #model_name=None,
                       log=log,
                       copy_prototype_imgs=True,
                       prototypes_to_keep=proto_to_keep)
accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                class_specific=class_specific, log=log)
print(find_nearest.find_k_nearest_patches_to_prototypes(dataloader=train_push_loader,
                                                          prototype_network_parallel=ppnet_multi,
                                                          k=5,
                                                          preprocess_input_function=preprocess_input_function,
                                                          full_save=False,
                                                          log=log))
print("last layer trasnpose: \n", last_layer.show_last_layer_connections_T(ppnet))
save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                            model_name=original_model_name.split('push')[0] + 'prune',
                            accu=accu,
                            target_accu=0.70, log=log)

# last layer optimization
if optimize_last_layer:
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    
    from settings import coefs

    log('optimize last layer')
    tnt.last_only(model=ppnet_multi, log=log)
    for i in range(25):
        log('iteration: \t{0}'.format(i))
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        print("last layer trasnpose: \n", last_layer.show_last_layer_connections_T(ppnet))
        save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                    model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune',
                                    accu=accu,
                                    target_accu=0.70, log=log)
logclose()
