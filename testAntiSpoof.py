import os
import torch
from torchvision import transforms
from utilsx.utils import read_cfg, get_optimizer, get_device, build_network
from trainerx.FASTrainer import FASTrainer
from PIL import Image
import time

class AntiSpoof:
    def __init__(self):
        self.cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")
        self.device = get_device(self.cfg)
        self.network = build_network(self.cfg)
        self.optimizer = get_optimizer(self.cfg, self.network)
        self.val_transform = transforms.Compose([
            transforms.Resize(self.cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg['dataset']['mean'], self.cfg['dataset']['sigma'])
            ])

        self.model = FASTrainer(
            cfg=self.cfg, 
            network=self.network,
            optimizer=self.optimizer,
            device=self.device
            )

        self.model.load_model()

    def pre_(self,image):
        t1 = time.time()
        image = Image.fromarray(image)
        image = self.val_transform(image)
        preds, score = self.model.test(image)
        if int(preds[0]) == 0:
            print('\n\n----------->Spoof')
        else:
            print('\n\n----------->Real')
        print('------>>>>>>predict: {}, scores: {}'.format(preds, score))
        print('time', time.time() - t1)

if __name__ == "__main__":
    img = Image.open('image.jpg')
    a = AntiSpoof()
    a.pre_(img)