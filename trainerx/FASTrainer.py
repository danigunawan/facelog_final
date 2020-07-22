import os
from random import randint
import torch
import torchvision
from trainerx.base import BaseTrainer
from utilsx.meters import AvgMeter
from utilsx.eval import add_visualization_to_tensorboard, predict, calc_accuracy
from torchsummary import summary

class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, device):
        super(FASTrainer, self).__init__(cfg, network, optimizer, device)
        self.network = self.network.to(device)

    def load_model(self):
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name, map_location=self.device)
        print('Load: ',saved_name)
        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])
        summary(self.network, (3,256,256))


    def save_model(self, epoch):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state, saved_name)


    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        for i, (img, depth_map, label) in enumerate(self.trainloader):
            img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
            net_depth_map, _, _, _, _, _ = self.network(img)
            self.optimizer.zero_grad()
            loss = self.criterion(net_depth_map, depth_map)
            loss.backward()
            self.optimizer.step()
            print('mean: %.5f\nmax: %.5f'%(torch.mean(net_depth_map), torch.max(net_depth_map)))
            preds, score = predict(net_depth_map)

            targets, _ = predict(depth_map)
            accuracy = calc_accuracy(preds, targets)
            print('preds:\t{}\nscore:\t{}\naccu:\t{}'.format(preds, score, accuracy))
            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)

            print('Epoch: {}, iter: {}, loss: {}, acc: {}\n\n'.format(epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg, self.train_acc_metric.avg))


    def train(self, criterion, lr_scheduler, trainloader, valloader, writer ):
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.trainloader = trainloader
        self.valloader = valloader
        self.writer = writer
        
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            with torch.no_grad():
                epoch_acc = self.validate(epoch)
            # if epoch_acc > self.best_val_acc:
            #     self.best_val_acc = epoch_acc
            self.save_model(epoch)


    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        seed = randint(0, len(self.valloader)-1)
        for i, (img, depth_map, label) in enumerate(self.valloader):
            img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
            net_depth_map, _, _, _, _, _ = self.network(img)
            print('------>max: %.5f , mean: %.5f'%(torch.max(net_depth_map),torch.mean(net_depth_map)))
            loss = self.criterion(net_depth_map, depth_map)

            preds, score = predict(net_depth_map)
            targets, _ = predict(depth_map)
            print('preds:\t{}\nscores:\t{}'.format(preds, score))
            accuracy = calc_accuracy(preds, targets)
            print('acc:\t',accuracy)
            # Update metrics
            self.val_loss_metric.update(loss.item())
            self.val_acc_metric.update(accuracy)


            if i == seed:
                add_visualization_to_tensorboard(self.cfg, epoch, img, preds, targets, score, self.writer)
            print('Epoch: {}, loss: {}, acc: {}\n\n'.format(epoch, self.val_loss_metric.avg, self.val_acc_metric.avg))
        return self.val_acc_metric.avg

    def test(self, image):
        with torch.no_grad():
            self.network.eval()
            image = image.unsqueeze(0)
            image = image.to(self.device)
            net_depth_map, _,_,_,_,_ = self.network(image)
            preds, score = predict(net_depth_map, threshold=0.65)
        return preds, score