import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
import torch
import math

import visdom
import torch.utils.data as Data
import argparse
import numpy as np
from tqdm import tqdm
from distutils.version import LooseVersion
from Datasets.ISIC2018 import ISIC2018_dataset
from utils.transform import ISIC2018_transform
from utils.transformer1 import ISIC2018_transform1
from Models.attention_unet import AttU_Net
from Models.unet___ import R2U_Net
from Models.unet___ import R2AttU_Net
from Models.networks.unet_model import UNet
from Models.unet___ import NestedUNet
from Models.deeplab_all import DeepLab
from Models.resunet import ResUnet
from Models.resunetplusplus import ResUnetPlusPlus
from Models.FAT import FAT_Net
from Models.swin_unet import SwinTransformerSys
from Models.vit import ViT
from Models.trans_unet import TransUNet
from Models.MsRed import Ms_red_v1

from Models.networks.res2net_mixer import res2MixerUNet
from Models.networks.res2last import res2MixerUNet_last

from utils.dice_loss import SoftDiceLoss, get_soft_label, val_dice_isic
from utils.dice_loss import Intersection_over_Union_isic
from utils.dice_loss import Accuracy_Sensitivity_Specificity_isic

from utils.evaluation import AverageMeter
from utils.binary import assd
from torch.optim.lr_scheduler import StepLR

Test_Model = {
              'attunet': AttU_Net,
              'unet': UNet,
              'r2unet': R2U_Net,
              'nestedunet':NestedUNet,
              'deeplab':DeepLab,
              'resunet':ResUnet,
              'resunetplus':ResUnetPlusPlus,
              'fat':FAT_Net,
              'swin_unet':SwinTransformerSys,
              'vit': ViT,
              'transunet':TransUNet, 
              'reslast1':res2MixerUNet_last,
              'msred1':Ms_red_v1,
              'res2mixer': res2MixerUNet
              }

Test_Dataset = {'ISIC2018': ISIC2018_dataset}

Test_Transform = {'ISIC2018': ISIC2018_transform}

def train(train_loader, model, criterion, optimizer, args, epoch):
    losses = AverageMeter()

    model.train()
    for step, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = x.float().cuda()
        target = y.float().cuda()
        output = model(image)
        target_soft = get_soft_label(target, args.num_classes)  # get soft label


        loss = criterion(output, target_soft, args.num_classes)  # the dice losses
        losses.update(loss.data, image.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % (math.ceil(float(len(train_loader.dataset)) / args.batch_size)) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                epoch, step * len(image), len(train_loader.dataset),
                       100. * step / len(train_loader), losses=losses))

    print('The average loss:{losses.avg:.4f}'.format(losses=losses))

    curFilename = args.ckpt + '/' + str("current_epoch_model") + '_' + args.data + '_checkpoint.pth.tar'
    print('the current model will be saved at {}'.format(curFilename))
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
    torch.save(state, curFilename)
    # 需要保存epoch数
    current_epoch = open(args.ckpt + '/' + 'current_epoch.txt', 'w')
    print("epoch: " + str(epoch), file=current_epoch)
    current_epoch.close()

    return losses.avg


def valid_isic(valid_loader, model, criterion, optimizer, args, epoch, minloss, maxdice):
    val_losses = AverageMeter()
    val_isic_dice = AverageMeter()

    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        image = t.float().cuda()
        target = k.float().cuda()
        with torch.no_grad():
            output = model(image)  # model output

            output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
            output_soft = get_soft_label(output_dis, args.num_classes)
            target_soft = get_soft_label(target, args.num_classes)  # get soft label

        val_loss = criterion(output, target_soft, args.num_classes)  # the dice losses
        val_losses.update(val_loss.data, image.size(0))

        isic = val_dice_isic(output_soft, target_soft, args.num_classes)  # the dice score
        val_isic_dice.update(isic.data, image.size(0))

        if step % (math.ceil(float(len(valid_loader.dataset)) / args.batch_size)) == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                epoch, step * len(image), len(valid_loader.dataset), 100. * step / len(valid_loader),
                losses=val_losses))
    # 训练的时候打印相关的信息
    print('The ISIC Mean Average Dice score: {isic.avg: .4f}; '
          'The Average Loss score: {loss.avg: .4f}'.format(
        isic=val_isic_dice, loss=val_losses))

    # 这里是保存loss最小的模型
    if val_losses.avg < min(minloss):
        minloss.append(val_losses.avg)
        print(minloss)
        modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    if val_isic_dice.avg > max(maxdice):
        maxdice.append(val_isic_dice.avg)
        print(maxdice)
        modelname = args.ckpt + '/' + 'max_dice' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best dice score model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    ml = minloss
    md = maxdice

    for idx, item in enumerate(ml):
        if torch.is_tensor(item):
            ml[idx] = item.cpu()

    for idx, item in enumerate(md):
        if torch.is_tensor(item):
            md[idx] = item.cpu()

    minlist = np.array(ml, dtype=object)
    maxlist = np.array(md, dtype=object)
    np.save('./minloss.npy', minlist)
    np.save('./maxdice.npy', maxlist)

    return val_losses.avg, val_isic_dice.avg



def test_isic_min_loss(test_loader, model, args):
    isic_dice = []
    isic_iou = []
    isic_assd = []

    isic_accuracy = []
    isic_sensitivity = []
    isic_specificity = []

    modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best loss model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    model.eval()
    for step, (img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = img.float().cuda()
        target = lab.float().cuda()

        output = model(image)  # model output
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_soft = get_soft_label(output_dis, args.num_classes)
        target_soft = get_soft_label(target, args.num_classes)  # get soft label

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

        isic_b_dice = val_dice_isic(output_soft, target_soft, args.num_classes)  # the dice accuracy
        isic_b_iou = Intersection_over_Union_isic(output_soft, target_soft, args.num_classes)  # the iou accuracy
        isic_b_asd = assd(output_arr[:, :, :, 1], label_arr[:, :, :, 1])  # the assd

        accuracy, sensitivity, specificity = Accuracy_Sensitivity_Specificity_isic(target_soft, output_soft)

        dice_np = isic_b_dice.data.cpu().numpy()
        iou_np = isic_b_iou.data.cpu().numpy()

        isic_accuracy.append(accuracy)
        isic_sensitivity.append(sensitivity)
        isic_specificity.append(specificity)

        isic_dice.append(dice_np)
        isic_iou.append(iou_np)
        isic_assd.append(isic_b_asd)

    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)

    isic_assd_mean = np.average(isic_assd)
    isic_assd_std = np.std(isic_assd)

    isic_accuracy_mean = np.average(isic_accuracy)
    isic_accuracy_std = np.std(isic_accuracy)

    isic_sensitivity_mean = np.average(isic_sensitivity)
    isic_sensitivity_std = np.std(isic_sensitivity)

    isic_specificity_mean = np.average(isic_specificity)
    isic_specificity_std = np.std(isic_specificity)
    min_txt = open(args.ckpt + '/' + 'min_loss.txt', 'w')
    print("=> the min loss model as the result :", file=min_txt)
    print('The ISIC mean DICE: {isic_dice_mean: .4f}; The Placenta DICE std: {isic_dice_std: .4f}'.format(
        isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std), file=min_txt)
    print('The ISIC mean IoU: {isic_iou_mean: .4f}; The ISIC IoU std: {isic_iou_std: .4f}'.format(
        isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std), file=min_txt)
    print('The ISIC mean assd: {isic_asd_mean: .4f}; The ISIC assd std: {isic_asd_std: .4f}'.format(
        isic_asd_mean=isic_assd_mean, isic_asd_std=isic_assd_std), file=min_txt)

    print('The ISIC mean accarucy: {isic_accuracy_mean: .4f}; The ISIC accuracy std: {isic_accuracy_std: .4f}'.format(
        isic_accuracy_mean=isic_accuracy_mean, isic_accuracy_std=isic_accuracy_std), file=min_txt)
    print(
        'The ISIC mean sensitivity: {isic_sensitivity_mean: .4f}; The ISIC sensitivity std: {isic_sensitivity_std: .4f}'.format(
            isic_sensitivity_mean=isic_sensitivity_mean, isic_sensitivity_std=isic_sensitivity_std), file=min_txt)
    print(
        'The ISIC mean specificity: {isic_specificity_mean: .4f}; The ISIC specificity std: {isic_specificity_std: .4f}'.format(
            isic_specificity_mean=isic_specificity_mean, isic_specificity_std=isic_specificity_std), file=min_txt)
    min_txt.close()


def test_isic_max_dice(test_loader, model, args):
    isic_dice = []
    isic_iou = []
    isic_assd = []

    isic_accuracy = []
    isic_sensitivity = []
    isic_specificity = []

    modelname = args.ckpt + '/' + 'max_dice' + '_' + args.data + '_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best dice model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    model.eval()
    for step, (img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = img.float().cuda()
        target = lab.float().cuda()

        output = model(image)  # model output
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_soft = get_soft_label(output_dis, args.num_classes)
        target_soft = get_soft_label(target, args.num_classes)  # get soft label

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

        isic_b_dice = val_dice_isic(output_soft, target_soft, args.num_classes)  # the dice accuracy
        isic_b_iou = Intersection_over_Union_isic(output_soft, target_soft, args.num_classes)  # the iou accuracy
        isic_b_asd = assd(output_arr[:, :, :, 1], label_arr[:, :, :, 1])  # the assd

        accuracy, sensitivity, specificity = Accuracy_Sensitivity_Specificity_isic(target_soft, output_soft)

        dice_np = isic_b_dice.data.cpu().numpy()
        iou_np = isic_b_iou.data.cpu().numpy()

        isic_accuracy.append(accuracy)
        isic_sensitivity.append(sensitivity)
        isic_specificity.append(specificity)

        isic_dice.append(dice_np)
        isic_iou.append(iou_np)
        isic_assd.append(isic_b_asd)

    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)

    isic_assd_mean = np.average(isic_assd)
    isic_assd_std = np.std(isic_assd)

    isic_accuracy_mean = np.average(isic_accuracy)
    isic_accuracy_std = np.std(isic_accuracy)

    isic_sensitivity_mean = np.average(isic_sensitivity)
    isic_sensitivity_std = np.std(isic_sensitivity)

    isic_specificity_mean = np.average(isic_specificity)
    isic_specificity_std = np.std(isic_specificity)

    # 设置文件将结果保存到max_dice_txt
    max_txt = open(args.ckpt + '/' + 'max_dice.txt', 'w')
    print("=> the max dice model as the result :", file=max_txt)
    print('The ISIC mean DICE: {isic_dice_mean: .4f}; The Placenta DICE std: {isic_dice_std: .4f}'.format(
        isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std), file=max_txt)
    print('The ISIC mean IoU: {isic_iou_mean: .4f}; The ISIC IoU std: {isic_iou_std: .4f}'.format(
        isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std), file=max_txt)
    print('The ISIC mean assd: {isic_asd_mean: .4f}; The ISIC assd std: {isic_asd_std: .4f}'.format(
        isic_asd_mean=isic_assd_mean, isic_asd_std=isic_assd_std), file=max_txt)
    print('The ISIC mean accarucy: {isic_accuracy_mean: .4f}; The ISIC accuracy std: {isic_accuracy_std: .4f}'.format(
        isic_accuracy_mean=isic_accuracy_mean, isic_accuracy_std=isic_accuracy_std), file=max_txt)
    print(
        'The ISIC mean sensitivity: {isic_sensitivity_mean: .4f}; The ISIC sensitivity std: {isic_sensitivity_std: .4f}'.format(
            isic_sensitivity_mean=isic_sensitivity_mean, isic_sensitivity_std=isic_sensitivity_std), file=max_txt)
    print(
        'The ISIC mean specificity: {isic_specificity_mean: .4f}; The ISIC specificity std: {isic_specificity_std: .4f}'.format(
            isic_specificity_mean=isic_specificity_mean, isic_specificity_std=isic_specificity_std), file=max_txt)
    max_txt.close()


def main(args):
    if args.start_epoch > 1:
        print("=> Now starting recovery the minloss or maxdice list!")
        minloss = np.load('./minloss.npy', allow_pickle=True).tolist()
        maxdice = np.load('./maxdice.npy', allow_pickle=True).tolist()
    else:
        minloss = [1.0]
        maxdice = [0.1]

    start_epoch = args.start_epoch

    # loading the dataset
    print('loading the {0},{1},{2} dataset ...'.format('train', 'validation', 'test'))
    trainset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='train',
                                       transform=Test_Transform[args.data])
    validset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='validation',
                                       transform=Test_Transform[args.data])
    testset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                      transform=Test_Transform[args.data])
    trainloader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=8)
    validloader = Data.DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=8)
    testloader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=8)
    print('Loading is done\n')

    # Define model
    if args.data == 'Fetus':
        args.num_input = 1
        args.num_classes = 3
        args.out_size = (256, 256)
    elif args.data == 'ISIC2018':
        args.num_input = 3
        args.num_classes = 2
        args.out_size = (256, 256)
    model = Test_Model[args.id](args)

    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()

    # collect the number of parameters in the network
    print("------------------------------------------")
    print("Network Architecture of Model CMU_Net:")
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    print(model)
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))
    print("------------------------------------------")

    # Define optimizers and loss function
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr_rate,
                                  weight_decay=args.weight_decay)
    criterion = SoftDiceLoss()
    scheduler = StepLR(optimizer, step_size=192, gamma=0.5)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    print("Start training ...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        scheduler.step()
        train_avg_loss = train(trainloader, model, criterion, optimizer, args, epoch)


        if args.data == 'Fetus':
            val_avg_loss, val_placenta_dice, val_brain_dice = valid_fetus(validloader, model, criterion,
                                                                          optimizer, args, epoch, minloss, maxdice)

        elif args.data == 'ISIC2018':
            val_avg_loss, val_isic_dice = valid_isic(validloader, model, criterion, optimizer, args, epoch, minloss,
                                                     maxdice)

        # save models
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
                print('the model will be saved at {}'.format(filename))
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                torch.save(state, filename)

    print('Training Done! Start testing')
    if args.data == 'Fetus':
        test_fetus(testloader, model, args)
    elif args.data == 'ISIC2018':

        test_isic_min_loss(testloader, model, args)
        test_isic_max_dice(testloader, model, args)

    print('Testing All Done!')

    final_filename = args.ckpt + '/' + str("final_epoch_model") + '_' + args.data + '_checkpoint.pth.tar'
    print('the final model will be saved at {}'.format(final_filename))
    state = {'epoch': args.epochs, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
    torch.save(state, final_filename)


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='CMUNet',
                        help='a name for identitying the model. Choose from the following options: Unet')

    # device choise gpuu
    parser.add_argument('--device', type=str, default='2', help='the choices of GPU')
    # Path related arguments
    parser.add_argument('--root_path', default='./data/ISIC2017_Task1_npy_all',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')

    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weights regularizer')

    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=200, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    # other arguments
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')
    # parser.add_argument('--backbone', default='resnet')
    parser.add_argument('--out_size', default=(256, 256), help='the output image size')
    parser.add_argument('--val_folder', default='folder6', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str('current_epoch_model') + '_' + args.data + '_checkpoint.pth.tar'
    main(args)

