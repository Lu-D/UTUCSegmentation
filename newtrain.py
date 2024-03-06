
import os


from collections import OrderedDict
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
# from torchvision.transforms import Compose
import matplotlib

matplotlib.use('Agg')

from model import get_net

from dataloader import *
from dataloader.FrameDataset import FrameDataset

from torch.utils.data import DataLoader, random_split

from util import   get_parser
from util.Dice import iou_score

import pandas as pd
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Flip, ColorJitter, GaussianBlur, RandomBrightnessContrast, Resize
import torch.backends.cudnn as cudnn

from util.AverageMeter import AverageMeter


# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_args():
    parser = get_parser(train=True)
    return parser.parse_args()


def train(args, train_loader, model, criterion, optimizer):
    '''
    Train for one epoch
    '''

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for batch in train_loader:
        input = batch['image'].cuda()
        target = batch['mask'].cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        iou, dice = iou_score(output, target)

        # gradient and optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(args, val_loader, model, criterion):
    '''
    Validate for one epoch
    '''
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for batch in val_loader:
            input = batch['image'].cuda()
            target = batch['mask'].cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            iou, dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])



def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define loss
    # criterion = BCEDiceLoss().cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    # improves speed when input size does not change
    cudnn.benchmark = True

    # create model
    model = get_net(args, device)
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=1, min_lr=args.min_lr)
    # load data

    train_transform = Compose([
        Resize(args.height, args.width),
        RandomRotate90(),
        Flip(),
        ColorJitter(p=0.2),
        GaussianBlur(p=0.2),
        RandomBrightnessContrast(p=0.2),
        transforms.Normalize()
    ])

    val_transform = Compose([
        Resize(args.height, args.width),
        transforms.Normalize()
    ])

    train_dataset = FrameDataset(args.inputs, args.labels, transform=train_transform, args=args)

    n_val = int(len(train_dataset) * args.val)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val])
    val_set.transform = val_transform
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    dir_checkpoint = os.path.join(args.save_dir + '/' + args.name, 'cp')
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    best_iou = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))

        # train one epoch

        train_log = train(args, train_loader, model, criterion, optimizer)
        # validate
        val_log = validate(args, val_loader, model, criterion)

        scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(args.lr)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(os.path.join(dir_checkpoint, f'log.csv'), index=False)

        # save model on improvement
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(),
                       os.path.join(dir_checkpoint, f'{args.name}_CP_epoch{epoch + 1}.pth'))
            best_iou = val_log['iou']
            print("=> saved best model")


        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
