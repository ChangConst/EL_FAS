import os
import argparse
import time
import yaml
import random
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from dataLoader.loadOULU_NPU import OULU_NPU
from dataLoader.loadSiW import SiW
from dataLoader.loadSiW2OULU import SiW2OULU
from utils.assistModules import AverageMeter, save_checkpoint
from utils.metrics import *
from models.ELFAS_Model import elfas_model
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="PyTorch FAS Project")
parser.add_argument('--config', default='cfgs/test.yaml', help='path to config file')
parser.add_argument('--j', '--workers', default=4, type=int, metavar='N',
                    help='number of data laoding workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number(useful on restarts)')
parser.add_argument('--random_seed', default=20, type=int,
                    help='fix seed to provide (near-)reproducibility')
parser.add_argument('--gpus', default='0', type=str,
                    help='specify which gpu to use, e.g. --gpus 0, 1')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='M',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum to use')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--val', '--evaluate', default=False, type=bool,
                    help='evaluate model on validation set')
parser.add_argument('--val_save', default=False, type=bool,
                    help='whether to save evaluate result')
parser.add_argument('--test', default=False, type=bool,
                    help='test model on test set')
parser.add_argument('--test_save', default=False, type=bool,
                    help='whether to save test result')
parser.add_argument('--input_size', default=224, type=int,
                    help='image size feed to the network')
parser.add_argument('--model_name', default='ELFAS', type=str,
                    help='name of the model')
parser.add_argument('--every_decay', default=40, type=int,
                    help='how many epoch lr decays')

parser.add_argument('--save_path', default='./checkpoints', type=str,
                    help='path to save checkpoints')
parser.add_argument('--log_path', default='./logs', type=str,
                    help='path to save log files')

parser.add_argument('--dataset', default='CASIA', type=str,
                    help='which dataset to use [CASIA, OULU_NPU, SiW, ReplyAttack]')
parser.add_argument('--protocol', default='1', type=str,
                    help='which protocol to use is there is any')
parser.add_argument('--color_space', default='rgb', type=str,
                    help='transform image to specified color space(rgb, ycbcr, hsv)')
parser.add_argument('--scale', default='1.0', type=str,
                    help='face scale to choose training image')

parser.add_argument('--se', default=False, type=bool,
                    help='whether to use se in unet')

best_prec1 = 0
USE_GPU = torch.cuda.is_available()

def main():
    global best_prec1, USE_GPU, args, device, datetime
    args = parser.parse_args()

    local_time = time.localtime()
    datetime = '_'.join([str(local_time.tm_year), str(local_time.tm_mon), str(local_time.tm_mday), str(local_time.tm_hour), str(local_time.tm_min)])
    device = torch.device('cuda:' + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")

    # config model with parameters from args.config
    with open(args.config, 'r') as f:
        config = yaml.load(f)
        # config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config['common'].items():
        setattr(args, k, v)

    # load specified dataset
    if str(args.dataset).lower() == 'oulu_npu':
        DATASET = OULU_NPU
        print('Running with dataset OULU_NPU')
    elif str(args.dataset).lower() == 'siw':
        DATASET = SiW
        print('Running with dataset SiW')
    elif str(args.dataset).lower() == 'siw2oulu':
        DATASET = SiW2OULU
        print('Running with dataset siw2oulu')
    else:
        print("error dataset is :", str(args.dataset))
        print('dataset error!')
        exit(-1)

    # seed everything
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    # create model
    model = elfas_model()

    input_size = int(args.input_size)

    if USE_GPU:
        cudnn.benchmark = True

        torch.cuda.manual_seed_all(args.random_seed)
        gpus = [int(i) for i in args.gpus.split(',')]
        model = torch.nn.DataParallel(model, device_ids=gpus)
        model.to(device)

    # print the total number of parameters

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("Parameters NUM: ", num_params)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch{})".format(
                args.resume, checkpoint['epoch']
            ))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('Running with protocol: ', args.protocol)

    if str(args.dataset).lower() == 'siw':
        phase_val = 'test'
    else:
        phase_val = 'eval'

    train_dataset = DATASET(input_size, phase='train', color_space=args.color_space, protocol=args.protocol, scale=args.scale)
    val_dataset = DATASET(input_size, phase=phase_val, color_space=args.color_space, protocol=args.protocol, scale=args.scale)
    test_dataset = DATASET(input_size, phase='test', color_space=args.color_space, protocol=args.protocol, scale=args.scale)

    train_sampler = None
    val_sampler = None
    test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=(train_sampler is None), sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=test_sampler)
    if args.val:
        validate(val_loader, model, args.start_epoch)
        return
    if args.test:
        test(test_loader, model, args.start_epoch, thr=0.8)
        return

    # training process
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)

        is_best = False
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_name = '{}/{}_{}.pth.tar'.format(args.save_path, args.model_name, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, filename=save_name)

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var = Variable(input).float().to(device)
        target_var = Variable(target).long().to(device)

        spoof_noise = model(input_var)

        loss = calc_loss(spoof_noise, target_var, device=device)

        scores = []
        for idx in range(spoof_noise.shape[0]):
            score = spoof_noise[idx, ...].mean().cpu()
            scores.append(score.detach().numpy())

        scores = np.array(scores)
        target = target.long().cpu().numpy()

        metrics_, best_thr, acc, predict = eval_from_scores(scores, target)

        top1.update(acc)

        reduced_loss = loss.data.clone()
        losses.update(reduced_loss)

        # compute gradient and do SDG step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]['lr']

        if i % args.print_freq == 0:
            if not os.path.isdir(args.log_path):
                os.mkdir(args.log_path)

            with open('{}/{}_{}.log'.format(args.log_path, datetime, args.model_name), 'a+') as flog:
                line = 'Epoch: [{0}][{1}/{2}]\t lr:{3:.5f}\t' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                       .format(epoch, i, len(train_loader),lr,
                        batch_time=batch_time, loss=losses, top1=top1)
                print(line)
                flog.write('{}\n'.format(line))

def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    result_list = []
    label_list = []
    predicted_list = []

    model.eval()

    end = time.time()
    for i, (input, target, image_dir) in enumerate(val_loader):
        with torch.no_grad():
            input_var = Variable(input).float().to(device)
            target_var = Variable(target).long().to(device)

            # compute output
            spoof_noise = model(input_var)

            loss = calc_loss(spoof_noise, target_var, device=device)

            scores = []
            for idx in range(spoof_noise.shape[0]):
                score = spoof_noise[idx, ...].mean().cpu()
                scores.append(score.detach().numpy())

            scores = np.array(scores)

            metrics_, best_thr, acc, predict = eval_from_scores(scores, target)

            losses.update(loss.data, input.size(0))
            top1.update(acc, input.size(0))

            label = target.long().cpu().numpy()

            for i_batch in range(len(predict)):
                result_list.append(scores[i_batch])
                label_list.append(label[i_batch])
                predicted_list.append(predict[i_batch])

                if args.val_save:
                    if not os.path.isdir(args.log_path):
                        os.mkdir(args.log_path)
                    f = open('{}/{}_{}_{}_val_detail.txt'.format(args.log_path, datetime, args.model_name, epoch), 'a+')
                    write_image_dir = image_dir[i_batch]
                    f.write(write_image_dir + ' ' + str(predict[i_batch, 1]) + '\n')

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                line = 'Val: [{0}/{1}]\t' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                         loss=losses, top1=top1)
                if not os.path.isdir(args.log_path):
                    os.mkdir(args.log_path)
                with open('{}/{}_{}.log'.format(args.log_path, datetime, args.model_name), 'a+') as flog:
                    flog.write('{}\n'.format(line))
                    print(line)
    metrics_, best_thr, acc, predict = eval_from_scores(np.array(result_list), np.array(label_list))
    tn, fp, fn, tp, acer, apcer, npcer = metrics_


    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    with open('{}/val_result_{}_{}.txt'.format(args.log_path, datetime,args.model_name),'a+') as f_result:
        result_line = 'epoch: {} APCER: {:.6f} NPCER: {:.6f} Acc: {:.3f} TN: {} FP: {} FN: {} TP: {}  ACER: {:.8f} Best_thr:{:.8f}'.format(epoch,apcer,npcer, top1.avg, tn, fp, fn,tp,acer,best_thr)
        f_result.write('{}\n'.format(result_line))
        print(result_line)
    return top1.avg


def test(test_loader, model, epoch, thr):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    result_list = []
    label_list = []
    predicted_list = []

    model.eval()

    end = time.time()
    for i, (input, target, image_dir) in enumerate(test_loader):
        with torch.no_grad():
            input_var = Variable(input).float().to(device)
            target_var = Variable(target).long().to(device)

            # compute output
            x5, spoof_noise = model(input_var)
            loss = calc_loss(spoof_noise, target_var, device=device)

            scores = []
            for idx in range(spoof_noise.shape[0]):
                score = spoof_noise[idx, ...].mean().cpu()
                scores.append(score.detach().numpy())

            scores = np.array(scores)

            metrics_, best_thr, acc, predict = eval_from_scores_thr(scores, target, thr)

            losses.update(loss.data, input.size(0))
            top1.update(acc, input.size(0))

            label = target.long().cpu().numpy()

            # save predict result for every input image if args.test_save is True
            for i_batch in range(len(predict)):
                result_list.append(scores[i_batch])
                label_list.append(label[i_batch])
                predicted_list.append(predict[i_batch])

                if args.test_save:
                    if not os.path.isdir(args.log_path):
                        os.mkdir(args.log_path)
                    with open('{}/{}_{}_{}_test_detail.txt'.format(args.log_path, datetime, args.model_name, epoch), 'a+') as f:
                        write_image_dir = image_dir[i_batch]
                        f.write(write_image_dir + ' ' + str(predict[i_batch]) + " " + str(scores[i_batch]) + '\n')

            # print and save batch test results
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                if not os.path.isdir(args.log_path):
                    os.mkdir(args.log_path)
                line = 'Test: [{0}/{1}]\t' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(i, len(test_loader), batch_time=batch_time,
                                                                         loss=losses, top1=top1)
                with open('{}/{}_{}.log'.format(args.log_path, datetime, args.model_name), 'a+') as flog:
                    flog.write('{}\n'.format(line))
                    print(line)

    # Final test results
    metrics_, best_thr, acc, predict = eval_from_scores_thr(np.array(result_list), np.array(label_list), thr)
    tn, fp, fn, tp, acer, apcer, npcer = metrics_

    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    with open('{}/test_result_{}_{}.txt'.format(args.log_path, datetime, args.model_name),'a+') as f_result:
        result_line = 'epoch: {} APCER:{:.6f} NPCER:{:.6f} Acc:{:.3f} TN:{} FP : {} FN:{} TP:{}  ACER:{:.8f} thr:{:.8f}'.format(epoch,apcer,npcer, top1.avg, tn, fp, fn,tp,acer,thr)
        f_result.write('{}\n'.format(result_line))
        print(result_line)
    return top1.avg



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.every_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calc_loss(spoof_noise, target, device):
    '''
    计算整体的loss值
    '''

    bce_loss = nn.BCELoss(reduction='mean')

    target_map = [torch.zeros_like(spoof_noise[0]) if ele == 0 else torch.ones_like(spoof_noise[0]) for ele in target]
    target_map = [torch.unsqueeze(ele, dim=0) for ele in target_map]
    target_map = torch.cat(target_map, dim=0)

    return bce_loss(spoof_noise, target_map)


if __name__ == '__main__':
    main()





