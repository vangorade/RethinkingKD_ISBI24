import os
import torch
import argparse
from train import SegPL
from networks import get_model
from utils.loss_functions import *
from torch.utils.data import DataLoader
from utils.base_pl_model import BasePLModel
from datasets.midataset import SliceDataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import seed
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


seed.seed_everything(123)
parser = argparse.ArgumentParser('train_kd')
parser.add_argument('--train_data_path', type=str, default='/home/vandangorade/ISBI24/DKD/code/data/lits19/train/train')
parser.add_argument('--test_data_path', type=str, default='/home/vandangorade/ISBI24/DKD/code/data/lits19/test/test')
parser.add_argument('--checkpoint_path', type=str, default='/home/vandangorade/ISBI24/DKD/code/final_ckpts/lits/student/wo_feedback')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--tckpt', type=str, default='/home/vandangorade/ISBI24/DKD/code/final_ckpts/lits/teacher/checkpoint_lits_tumor_unet++_epoch=78.ckpt', help='teacher model checkpoint path')
parser.add_argument('--smodel', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, default='lits', choices=['kits', 'lits'])
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--lr', type=float, default=1e-2)
# loss_type parser
parser.add_argument('--loss_type', type=str, default='wo_feedback')

# KD loss para
alpha = 0.1
beta1 = 0.9
beta2 = 0.9


class KDPL(BasePLModel):
    def __init__(self, params):
        super(KDPL, self).__init__()
        self.save_hyperparameters(params)

        # load and freeze teacher net
        self.t_net = SegPL.load_from_checkpoint(checkpoint_path=self.hparams.tckpt)
        self.t_net.freeze()

        # student net
        self.net = get_model(self.hparams.smodel, channels=2)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        self.t_net.eval()
        t_e1, t_e2, t_e3, t_e4, t_e5, t_d5, t_d4, t_d3, t_d2, t_output = self.t_net.net(ct)
        s_e1, s_e2, s_e3, s_e4, s_e5, s_d5, s_d4, s_d3, s_d2, s_output, = self.net(ct)

        loss_seg = calc_loss(s_output, mask)

    
        if args.loss_type == "attn_kd": #baseline
            loss_imd = importance_maps_distillation(s_e5, t_e5)
            loss = loss_seg + loss_imd 
        elif args.loss_type == "deep_kd":  #DKD
            loss_pmd = prediction_map_distillation(s_output, t_output)  + \
                    prediction_map_distillation(s_d5, s_d5) + \
                    prediction_map_distillation(s_d4, t_d4) + \
                    prediction_map_distillation(s_d3, t_d3) + \
                    prediction_map_distillation(s_d2, t_d2)
            loss_imd = importance_maps_distillation(s_e5, t_e5)  + \
                    importance_maps_distillation(s_e1, t_e1) + \
                    importance_maps_distillation(s_e2, t_e2) + \
                    importance_maps_distillation(s_e3, t_e3) + \
                    importance_maps_distillation(s_e4, t_e4)
            loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd 

        elif args.loss_type == "indv_feedback":  # HIFD
            loss_pmd = prediction_map_distillation(s_output, t_output)  + \
                    prediction_map_distillation(s_d4, t_output) + \
                    prediction_map_distillation(s_d2, t_output)
            loss_imd = importance_maps_distillation(s_e5, t_e5)  + \
                    importance_maps_distillation(s_e2, t_e5) + \
                    importance_maps_distillation(s_e4, t_e5)
            loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd 

        elif args.loss_type == "wo_feedback": # SKD
            loss_pmd = prediction_map_distillation(s_output, t_output)  
            loss_imd = importance_maps_distillation(s_e5, t_e5) 
            
            loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd 
        
        elif args.loss_type == "cat_feedback_ours": # IFD
            loss_pmd = prediction_map_distillation(s_output, t_output)  + \
                    cat_prediction_map_distillation([s_d5, s_d4, s_d3, s_d2], t_output) 
            loss_imd = importance_maps_distillation(s_e5, t_e5)  + \
                    cat_importance_maps_distillation([s_e1, s_e2, s_e3, s_e4], t_e5) 

            loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd 

        elif args.loss_type == "recursive_feedback_ours": # HLFD
            loss_pmd = 0.5 * rec_prediction_map_distillation(s_d5, [t_d5, t_d4, t_d3])  + \
            0.5 * (prediction_map_distillation(s_d4, t_output) + 
                   prediction_map_distillation(s_d3, t_output) + 
                   prediction_map_distillation(s_d2, t_output))
            
            loss_imd = 0.5 * recur_cat_importance_maps_distillation(s_e1, [t_e2, t_e3, t_e4]) + \
            0.5 * (importance_maps_distillation(s_e2, t_e5) + 
                   importance_maps_distillation(s_e3, t_e5) + 
                   importance_maps_distillation(s_e4, t_e5)) 

            loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd

        else:
            print("select correct loss type.")            


        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        s_e1, s_e2, s_e3, s_e4, s_e5, s_d5, s_d4, s_d3, s_d2, s_output, = self.net(ct)

        self.measure(batch, s_output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.train_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=32, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.test_data_path,
            dataset=self.hparams.dataset,
            task=self.hparams.task,
            train=False
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]


def main():
    args = parser.parse_args()
    model = KDPL(args) #.load_from_checkpoint(checkpoint_path=os.path.join(args.checkpoint_path, 'model_path.ckpt'))

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path),
        filename='checkpoint_%s_%s_kd_%s_{epoch}_%s' % (args.dataset, args.task, args.smodel, args.loss_type),
        save_top_k=1,  # Save the best checkpoint
        
    )
    logger = TensorBoardLogger('log', name='%s_%s_kd_%s' % (args.dataset, args.task, args.smodel))
    trainer = Trainer.from_argparse_args(args, 
                                        # resume_from_checkpoint=os.path.join(args.checkpoint_path, 'checkpoint_kits_tumor_kd_resnet18_epoch=59_recursive_feedback_ours_scat.ckpt'),
                                        max_epochs=args.epochs, 
                                        gpus=[6, 3],           
                                        strategy='dp',  
                                        precision=16,            
                                        callbacks=checkpoint_callback, 
                                        logger=logger)
    trainer.fit(model)


def test():
    args = parser.parse_args()
    model = KDPL.load_from_checkpoint(checkpoint_path=os.path.join(args.checkpoint_path, 'model_path.ckpt'))
    trainer = Trainer(gpus=[0,2],            
                    strategy='dp', 
                    precision=16)
    trainer.test(model)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()
