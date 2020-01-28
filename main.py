from dataset_video import return_dataset
from models import VideoModel
import torchvision
import torch
from dataset import VideoDataset
import argparse
import os
import sys

def main():

    global args 

    args = parser.parse_args()


    if args.dataset == 'something-v1':
        num_class = 174
        rgb_prefix = ''
        rgb_read_format = "{:05d}.jpg"
    else:
        ValueError("Unknown dataset"+args.dataset)

    model_dir = os.path.join('experiments', args.dataset, arg.arch. args.consensus_type+'-'+args.modality, str(args.run_iter))
    if not args.resume:
        if os.path.exists(model_dir):
            print(f"Dir {model_dir} already exists!")
        else:
            os.makedir(model_dir)
            os.makedirs(os.path.join(model_dir, args.root_log))

        

    train_videofolder, _, _, _ = return_dataset("something-v1")

    model = VideoModel(num_class=num_class, modality=args.modality, 
                        num_segments=args.num_segments, base_model=args.arch, consensus_type=args.consensus_type,
                        dropout=args.dropout=. partial_bn=not args.no_partial_bn, gsm=args.gsm, target_transform=None)


    #model = torch.nn.DataParallel(model, deviceids=args.gpus).cuda()

    


    

    train_loader = torch.utils.data.DataLoader(
        VideoDataset("something-v1", train_videofolder, num_segments=8,
                   new_length=1,
                   modality="RGB",
                   image_tmpl=rgb_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                   ])),
        batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True)

if __name__ == "__main__":
    main()