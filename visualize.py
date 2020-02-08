from dataset_video import return_dataset
from models import VideoModel
import torchvision
import torch
from dataset import VideoDataset
import argparse
from opts import parser
import os
import sys
import time
import torch.backends.cudnn as cudnn
from transforms import *
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
import shutil
import CosineAnnealingLR
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt

def main():

    global args

    args = parser.parse_args()
  
    train_videofolder, val_videofolder, args.root_path, _ = return_dataset(args.dataset)

    num_class = 174
    rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"

    model = VideoModel(num_class=num_class, modality=args.modality,
                        num_segments=args.num_segments, base_model=args.arch, consensus_type=args.consensus_type,
                        dropout=args.dropout, partial_bn=not args.no_partialbn, gsm=args.gsm, target_transform=None)

    print("parameters", sum(p.numel() for p in model.parameters()))

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    train_augmentation = model.get_augmentation()
    policies = model.get_optim_policies()
    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    normalize = GroupNormalize(input_mean, input_std)



    dataset = VideoDataset(args.root_path, train_videofolder, num_segments=8,
                   new_length=1,
                   modality="RGB",
                   image_tmpl=rgb_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize
                   ]))

    def normalize_output(img):
        img = img - img.min()
        img = img / img.max()
        return img
    data = dataset[0][0].unsqueeze_(0).cuda()
    output = model(data)

    #print(model)
    #.exit(1)

    # Plot some images
    idx = torch.randint(0, output.size(0), ())
    #pred = normalize_output(output[idx, 0])
    img = data[idx, 0]

    #fig, axarr = plt.subplots(1, 2)
    plt.imshow(img.cpu().detach().numpy())
    #axarr[1].imshow(pred.cpu().detach().numpy())

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

  

    model.base_model.conv1_7x7_s2.register_forward_hook(get_activation('conv1'))
    data, _ = dataset[0]
    data.unsqueeze_(0)
    output = model(data.cuda())

    kernels = model.base_model.conv1_7x7_s2.weight.cpu().detach()

    fig, axarr = plt.subplots(kernels.size(0)-40, figsize=(15,15))
    for idx in range(kernels.size(0)-40):
        axarr[idx].imshow(np.transpose(kernels[idx].squeeze(), (1,2,0)))
        

    act = activation['conv1'].squeeze()
    fig, axarr = plt.subplots(act.size(0), figsize=(15,15))
    for idx in range(act.size(0)):
        axarr[idx].imshow(np.transpose(act[idx][:3].cpu(), (1,2,0)))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
