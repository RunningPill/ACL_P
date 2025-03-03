from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from data import ModelNet40
from kneedata import BonePointNet2, BoneEvaluateSet, BonePointNet, FemurTibiaPointNet
from model import Pct, Pct_FT
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

import time 

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    # 原始训练集
    train_loader = DataLoader(BonePointNet2(partition='train', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('train len:', len(train_loader) * args.batch_size)
    test_loader = DataLoader(BonePointNet2(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    print('test len:', len(test_loader) * args.test_batch_size)

    # 混合数据集
    # train_loader = DataLoader(BonePointNet(partition='train', num_points=args.num_points),
    #                         batch_size=args.batch_size, shuffle=True, drop_last=True)
    # print('train len:', len(train_loader) * args.batch_size)

    # test_loader = DataLoader(BonePointNet(partition='test', num_points=args.num_points),
    #                         batch_size=args.batch_size, shuffle=True, drop_last=True)
    # print('test len:', len(test_loader) * args.batch_size)

    # 两路融合
    # train_loader = DataLoader(FemurTibiaPointNet(partition='train', num_points=args.num_points),
    #                         batch_size=args.batch_size, shuffle=True, drop_last=True)
    # print('train len:', len(train_loader) * args.batch_size)

    # test_loader = DataLoader(FemurTibiaPointNet(partition='test', num_points=args.num_points),
    #                         batch_size=args.batch_size, shuffle=True, drop_last=True)
    # print('test len:', len(test_loader) * args.batch_size)

    device = torch.device("cuda" if args.cuda else "cpu")

    # 股骨胫骨平移单路
    model = Pct(args).to(device)

    # 两路融合
    # model = Pct_FT(args).to(device)

    # 多块GPU
    print(str(model))
    model = nn.DataParallel(model)

    if args.use_sgd == True:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        # scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        for data, label in (train_loader):
            data, label = data.to(device), label.to(device).squeeze() 
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            start_time = time.time()
            logits = model(data)

        # for data1,data2, label in (train_loader):
        #     data1 , data2, label = data1.to(device), data2.to(device), label.to(device).squeeze()
        #     data1 = data1.permute(0, 2, 1)
        #     data2 = data2.permute(0, 2, 1)
        #     batch_size = data1.size()[0]
        #     opt.zero_grad()
        #     start_time = time.time()
        #     logits = model(data1, data2)


            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            
        print ('train total time is',total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                train_loss*1.0/count,
                                                                                metrics.accuracy_score(
                                                                                train_true, train_pred),
                                                                                metrics.balanced_accuracy_score(
                                                                                train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)

        # for data1,data2, label in (test_loader):
        #     data1 , data2, label = data1.to(device), data2.to(device), label.to(device).squeeze()
        #     data1 = data1.permute(0, 2, 1)
        #     data2 = data2.permute(0, 2, 1)
        #     batch_size = data1.size()[0]
        #     start_time = time.time()
        #     logits = model(data1 , data2)

            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print ('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            io.cprint('Best checkpoint\n ' + outstr)
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

        # save final epoch
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_final.t7' % args.exp_name)
        # after each epoch, adjust the learning rate
        scheduler.step()

# 手动混淆矩阵计算特异度spec
def specificityCalc(Predictions, Labels):
    MCM = metrics.confusion_matrix(Labels, Predictions)
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]

    tp_sum = MCM[1, 1]
    fn_sum = MCM[1, 0]

    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    Sensitivity = tp_sum / (tp_sum + fn_sum + 1e-6)

    return Specificity, Sensitivity

def test(args, io):
    # test_loader = DataLoader(FemurTibiaPointNet(partition='val', num_points=args.num_points),
    #                         batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(BonePointNet2(partition='test', num_points=args.num_points), num_workers=8,
                        batch_size=args.test_batch_size, shuffle=False, drop_last=True)
    # test_loader = DataLoader(BonePointNet(partition='val', num_points=args.num_points),
    #                         batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    # bone_data_path = '/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/6016'
    # test_loader = DataLoader(BoneEvaluateSet(data_path=bone_data_path ,num_points=args.num_points),
    #                        batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    print('val len:', len(test_loader) * args.batch_size)
    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args).to(device)
    # model = Pct_FT(args).to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    test_for_auc = []
    # for data1, data2, label, pid in test_loader:
    #     data1, data2, label = data1.to(device), data2.to(device), label.to(device).squeeze()
    #     data1 = data1.permute(0, 2, 1)
    #     data2 = data2.permute(0, 2, 1)
    #     logits = model(data1, data2)
    # for data, label, pid in test_loader:
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)

        pred_for_auc = nn.functional.softmax(logits, dim=1)
        preds = logits.max(dim=1)[1] 
        print(preds)
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            test_for_auc.append(pred_for_auc[:, 1].detach().cpu().numpy())

        # pid 和t1概率逐行写入txt
        # with open('checkpoints/' + args.exp_name + '/pid_pred_1.txt', 'a') as f:
        #     for i in range(len(pid)):
        #         f.write(str(pid[i]) + ' ' + str(pred_for_auc[i][0].item()) + '\n')
        
    # test_true = np.concatenate(test_true)
    # test_pred = np.concatenate(test_pred)
    # test_for_auc = np.concatenate(test_for_auc)
    # test_acc = metrics.accuracy_score(test_true, test_pred)
    # outstr = 'Test :: test acc: %.6f'%(test_acc)
    # io.cprint(outstr)

    test_true = np.concatenate(test_true)
    gtstr = f'gt: {test_true}'
    io.cprint(gtstr)
    test_pred = np.concatenate(test_pred)
    predstr = f'pred: {test_pred}'
    io.cprint(predstr)
    test_for_auc = np.concatenate(test_for_auc)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    test_auc = metrics.roc_auc_score(test_true, test_for_auc, average='macro', multi_class='ovo')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_spec,test_sens = specificityCalc(test_pred, test_true)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f,test auc: %.6f, test avg acc: %.6f, test precision: %.6f, \
        test recall(sens): %.6f,test sens: %.6f, test spec: %.6f, test f1: %.6f'%(test_acc,test_auc, avg_per_class_acc, test_precision, test_recall,test_sens,test_spec, test_f1)
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()
    # now time
    nowtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    io = IOStream('checkpoints/' + args.exp_name + f'/run_{nowtime}.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
