from dataset_video import return_dataset
import torchvision
import torch
from dataset import VideoDataset

def main():

    train_videofolder, _, _, _ = return_dataset("something-v1")

    rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"

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