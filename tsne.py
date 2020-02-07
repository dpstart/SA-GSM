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


def svg(embeddings, labels, thumbnails, viewbox = (200, 200), legend_size = (10, 10), legend_font_size = 3, border_width = 3, circle_radius = 1):
    embeddings = (embeddings - embeddings.min(0)[0]) / (embeddings.max(0)[0] - embeddings.min(0)[0]) * torch.Tensor(viewbox).type(torch.DoubleTensor)
    legend_offset = viewbox[0] - legend_size[0], viewbox[1] - legend_size[1]
    class_index = sorted(set(labels))
    class_colors = [360 / (i + 1) for i in range(len(class_index))]
    colors = [class_colors[class_index.index(label)] for label in labels]
    #thumbnails_base64 = [base64.b64encode(cv2.imencode('.jpg', img.mul(255).permute(1, 2, 0).numpy()[..., ::-1])[1]) for img in thumbnails]

    return '<svg style="border: {}px solid" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">'.format(border_width, *viewbox) + \
        ''.join(map('<circle onmouseover="show_tooltip(evt)" cx="{}" cy="{}" title="{}" fill="hsl({}, 50%, 50%)" r="{}" />'.format, embeddings[:, 0], embeddings[:, 1], labels, colors, [circle_radius] * len(embeddings))) + \
        '''<script>
        function show_tooltip(e)
        {{
            e.target.ownerDocument.getElementById("preview").setAttribute("href", e.target.getAttribute("desc"));
            e.target.ownerDocument.getElementById("label").textContent = e.target.getAttribute("title");
        }}
        </script>
        <image id="preview" x="0" width="{legend_size[0]}" height="{legend_size[1]}" y="{legend_offset[0]}"/>
        <text id="label" x="0" y="{legend_offset[1]}" font-size="{legend_font_size}" />
        </svg>'''.format(legend_size = legend_size, legend_offset = legend_offset, legend_font_size = legend_font_size)

def main():

    train_videofolder, val_videofolder, _, _ = return_dataset("something-v1")

    model = VideoModel(num_class=174, modality="RGB",
                        num_segments=8, base_model="BNInception", consensus_type="avg",
                        dropout=0.5, partial_bn=False, gsm=True, target_transform=None)

    #print("parameters", sum(p.numel() for p in model.parameters()))

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    train_augmentation = model.get_augmentation()


    normalize = GroupNormalize(input_mean, input_std)

    rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"

    dataset = VideoDataset("dataset/something-v1/20bn-something-something-v1", train_videofolder, num_segments=8,
                   new_length=1,
                   modality="RGB",
                   image_tmpl=rgb_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=True),
                       ToTorchFormatTensor(div=True),
                       normalize
                   ]))

    thumbnails, labels = zip(*dataset)
    embeddings = torch.cat(thumbnails).view(len(thumbnails), -1)

    #embeddings, labels, thumbnails = (zip(embeddings, labels, thumbnails))
    print(embeddings.shape)
    #embeddings = torch.stack(embeddings)

    embeddings = tsne(embeddings)
    #open('svg.xml', 'w').write(svg(embeddings, labels, thumbnails))
    
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

    print(labels)

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
    return torch.from_numpy(TSNE(n_iter = 250, verbose=5).fit_transform(embeddings.numpy()))

if __name__ == "__main__":
    main()
