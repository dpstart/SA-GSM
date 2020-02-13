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
from MulticoreTSNE import MulticoreTSNE as TSNE
import base64
ssl._create_default_https_context = ssl._create_unverified_context

import random
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import argparse
sns.set()

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def main():

    parser = argparse.ArgumentParser(
    description="TRN testing on the full validation set")
    parser.add_argument('dataset', type=str, choices=['something-v1','diving48'])
    parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('weights', type=str)
    parser.add_argument('--arch', type=str, default="BNInception")
    parser.add_argument('--save_scores', default=False, action="store_true")
    parser.add_argument('--test_segments', type=int, default=8)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--num_clips',type=int, default=1,help='Number of clips sampled from a video')
    parser.add_argument('--softmax', type=int, default=0)
    parser.add_argument('--gsm', default=False, action="store_true")

    args = parser.parse_args()

    train_videofolder, args.val_list, args.root_path, _ = return_dataset("diving48")
    #train_videofolder, val_videofolder, _, _ = return_dataset("diving48")

    #num_class = 174
    num_class = 48



    net = VideoModel(num_class=174, modality="RGB",
                       num_segments=8, base_model="BNInception", consensus_type="avg",  gsm=True, target_transform=None)

    print("parameters", sum(p.numel() for p in net.parameters()))



    checkpoint = torch.load("experiments/diving48/BNInception/avg-RGB/13/log_best.pth.tar")
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    #print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))


    args.rgb_prefix = ''
    args.rgb_read_format = "{:05d}.jpg"

    args.rgb_prefix = 'frames'
    #rgb_read_format = "{:05d}.jpg"
    
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])

    data_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.val_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix+args.rgb_read_format,
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ]), num_clips=args.num_clips),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))


    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    data_gen = enumerate(data_loader)


    targets = [video.label for video in data_loader.dataset.video_list]
    targets = torch.tensor(targets)
    target_idx = (targets<50).nonzero()


    sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx)


    net.consensus = Identity()
    net.new_fc = Identity()
    net = net.cuda()

    embs = None
    targets = None

    net.eval()
    data_gen = enumerate(data_loader)

    #print(next(data_gen))

    for i, (input, target) in data_gen:

        if  target==5 or target == 3 or target==2:

            out = net(input.cuda())
            temp = out.cpu().detach()
            del out

            if embs is None and targets is None:
                embs = temp
                targets = target
            else:
                embs = torch.cat((embs,temp))
                targets = torch.cat((targets, target))
            print(embs.shape)

    
    embs = tsne(embs)

    import seaborn as sns
    N = len(np.unique(targets))

    palette = sns.color_palette("bright", N)

    sns.scatterplot(embs[:,0], embs[:,1], hue=targets, legend='full', palette=palette)
    plt.show()
    sys.exit(1)
    fig, ax = plt.subplots(1,1, figsize=(6,6))

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    scat = ax.scatter(embs[:,0], embs[:,1], s=10, rasterized=True, c=targets,cmap=cmap)
    # create the colorbar
    #cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    #cb.set_label('Custom cbar')
    ax.set_title('Discrete color mappings')
    plt.show()
    sys.exit(1)

    thumbnails, labels = zip(*dataset)
    embeddings = torch.cat(thumbnails).view(len(thumbnails), -1)

    #embeddings, labels, thumbnails = (zip(embeddings, labels, thumbnails))
    print(embeddings.shape)
    #embeddings = torch.stack(embeddings)

    embeddings = tsne(embeddings)
    
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    N = len(np.unique(labels))

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    scat = ax.scatter(embeddings[:,0], embeddings[:,1],c=labels,cmap=cmap)
    # create the colorbar
    #cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    #cb.set_label('Custom cbar')
    ax.set_title('Discrete color mappings')
    plt.show()
    #plt.scatter(embeddings[:,0], embeddings[:,1], label=labels)
    #plt.show()

def tsne(embeddings):
    return torch.from_numpy(TSNE(n_iter = 2000, perplexity=20, n_components=2, verbose=5).fit_transform(embeddings.numpy()))

if __name__ == "__main__":
    main()
