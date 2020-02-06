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
from transforms import *
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

best_prec1 = 0

def main():

    global args

    args = parser.parse_args()


    if args.dataset == 'something-v1':
        num_class = 174
        rgb_prefix = ''
        rgb_read_format = "{:05d}.jpg"
    else:
        ValueError("Unknown dataset"+args.dataset)

    model_dir = os.path.join('experiments', args.dataset, args.arch, args.consensus_type+'-'+args.modality, f"{args.run_iter}")
    if not args.resume:
        if os.path.exists(model_dir):
            print(f"Dir {model_dir} already exists!")
        else:
            os.makedirs(model_dir)
            os.makedirs(os.path.join(model_dir, args.root_log))

    writer = SummaryWriter(model_dir)

    train_videofolder, val_videofolder, _, _ = return_dataset("something-v1")

    model = VideoModel(num_class=num_class, modality=args.modality,
                        num_segments=args.num_segments, base_model=args.arch, consensus_type=args.consensus_type,
                        dropout=args.dropout, partial_bn=not args.no_partialbn, gsm=args.gsm, target_transform=None)


    train_augmentation = model.get_augmentation()
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model).cuda()



    train_loader = torch.utils.data.DataLoader(
        VideoDataset("dataset/something-v1/20bn-something-something-v1", train_videofolder, num_segments=8,
                   new_length=1,
                   modality="RGB",
                   image_tmpl=rgb_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']))
                   ])),
        batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        VideoDataset("dataset/something-v1/20bn-something-something-v1", val_videofolder, num_segments=8,
                   new_length=1,
                   modality="RGB",
                   image_tmpl=rgb_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3']))
                   ])),
        batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies,
                                args.lr)

    args.start_epoch = 0
    log_training = open(os.path.join(model_dir, args.root_log, '%s.csv' % args.store_name), 'a')

    for epoch in range(args.start_epoch, args.epochs):

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)

        train_prec1 = train(train_loader, model, criterion, optimizer, epoch, log_training, writer=writer)

        #lr_scheduler_clr.step()

        # evaluate on validation set
        if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training,
                             writer=writer, epoch=epoch)

            
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'current_prec1': prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, is_best, model_dir)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'current_prec1': train_prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, False, model_dir)
       


def train(train_loader, model, criterion, optimizer, epoch, log, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    """
    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)
    """

    # switch to train mode
    model.train()
    loss_summ = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / args.iter_size
        loss_summ += loss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss_summ.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
        if (i+1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            optimizer.step()
            optimizer.zero_grad()
            loss_summ = 0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            writer.add_scalar('train/batch_loss', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_top1Accuracy', top1.avg, epoch * len(train_loader) + i)
            log.write(output + '\n')
            log.flush()
    writer.add_scalar('train/loss', losses.avg, epoch + 1)
    writer.add_scalar('train/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('train/top5Accuracy', top5.avg, epoch + 1)
    return top1.avg

def validate(val_loader, model, criterion, iter, log, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))

            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                print(output)
                log.write(output + '\n')
                log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    writer.add_scalar('test/loss', losses.avg, epoch + 1)
    writer.add_scalar('test/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('test/top5Accuracy', top5.avg, epoch + 1)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    main()
