import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet import resnet50
from mobilenet import mobilenet_v2
import numpy as np

def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

model_names = ['resnet50', 'mobilenet_v2']
modes = ['uniform', 'progressive']
	
parser = argparse.ArgumentParser(description='Testing script')
parser.add_argument('data-path', type=str, metavar='PATH', help='path to ImageNet')
parser.add_argument('checkpoint', type=str, metavar='PATH', help='name of the model file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: '+' | '.join(model_names)+' (default: resnet50)')
parser.add_argument('--mode', default='uniform', choices=modes, help='compression mode: '+' | '.join(modes)+' (default: uniform)')
parser.add_argument('-g', '--groups', default=4, type=int)
parser.add_argument('-r', '--compression-rate', default=2.0, type=float)

args = parser.parse_args()

model = resnet50(pretrained=False, g=args.g, r=args.r) if args.arch == 'resnet50' else mobilenet_v2(pretrained=False, g=args.g, r=args.r)
model.load_state_dict(torch.load(args.checkpoint))
model.cuda()
model = nn.DataParallel(model)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(os.path.join(args.data_path, "val"), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])),
    batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)

top1, top5 = validate(val_loader, model)