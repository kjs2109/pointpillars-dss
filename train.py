import argparse   
import os 
import torch 
from tqdm import tqdm 

import random 
import numpy as np 
import matplotlib.pyplot as plt
from dataset import DssDataset, get_dataloader 
from model import PointPillars 
from loss import Loss 
from torch.utils.tensorboard import SummaryWriter 


def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None): 
    for k, v in loss_dict.items(): 
        writer.add_scalar(f'{tag}/{k}', v, global_step) 
    
    if lr is not None: 
        writer.add_scalar('lr', lr, global_step) 
    
    if momentum is not None: 
        writer.add_scalar('momentum', momentum, global_step) 


def check_and_mkdir(target_path):
    target_path = os.path.abspath(target_path)
    path_to_targets = os.path.split(target_path)

    if '.' in path_to_targets[-1]: 
        path_to_targets = path_to_targets[:-1] 
    
    path_history = '/'
    for path in path_to_targets:
        path_history = os.path.join(path_history, path)
        if not os.path.exists(path_history):
            os.mkdir(path_history)


def save_plot(dice_score_log, save_path): 

    check_and_mkdir(save_path)

    plt.figure(figsize=(10, 8))
    plt.plot(dice_score_log)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(save_path)
    plt.close()


def main(args): 

    setup_seed() 

    # dataset 
    train_dataset = DssDataset(data_root=args.data_root, split='train') 
    val_dataset = DssDataset(data_root=args.data_root, split='val') 

    # dataloader 
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    # model 
    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses).cuda()
    else:
        pointpillars = PointPillars(nclasses=args.nclasses)

    # loss 
    loss_func = Loss()

    # optimizer & scheduler 
    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    
    # log & save checkpoint 
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True) 

    # train loop 
    total_loss_log = [] 
    for epoch in range(args.max_epoch): 
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0 

        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:

                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda() 

            optimizer.zero_grad() 

            batched_pts = data_dict['batched_pts'] 
            batched_gt_bboxes = data_dict['batched_gt_bboxes'] 
            batched_labels = data_dict['batched_labels'] 
            batched_difficulty = data_dict['batched_difficulty'] 
            
            # inference 
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pointpillars(batched_pts=batched_pts, 
                                                                                            mode='train',
                                                                                            batched_gt_bboxes=batched_gt_bboxes, 
                                                                                            batched_gt_labels=batched_labels)
            
            # reshape  b x c x h x w -> b x h x w x c -> n x c 
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)  #  x, y, z, w, l, h, ry
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)     

            # anchor_target_dict  
            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7) 
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)

            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)  

            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)  
            bbox_pred = bbox_pred[pos_idx] 
            batched_bbox_reg = batched_bbox_reg[pos_idx] 

            
            ########################################################## loss ##########################################################
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
            batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
            #############################################################################################################################
            
            loss = loss_dict['total_loss']
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.ckpt_freq_epoch == 0: 
                save_summary(writer, loss_dict, global_step, 'train', 
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0]) 
                
            train_step += 1  

        total_loss_log.append(loss_dict['total_loss'].item()) 
        save_plot(total_loss_log, os.path.join(saved_logs_path, 'loss_curve.png')) 

        if (epoch + 1) % args.ckpt_freq_epoch == 0: 
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'pointpillars_{epoch}.pth'))

        if epoch % 2 == 0: 
            continue 

        pointpillars.eval()
        with torch.no_grad(): 
            for i, data_dict in enumerate(tqdm(val_dataloader)): 
                if not args.no_cuda: 
                    for key in data_dict: 
                        for j, item in enumerate(data_dict[key]): 
                            if torch.is_tensor(item): 
                                data_dict[key][j] = data_dict[key][j].cuda() 

                # TODO: calculate the metrics 

                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                
                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1
        pointpillars.train() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/workspace/DssDataset', help='your data root for DssDataset')
    parser.add_argument('--saved_path', default='pillar_logs/test2_nclases-3')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true',help='whether to use cuda')
    args = parser.parse_args()

    main(args)





