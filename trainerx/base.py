class BaseTrainer():
    def __init__(self, cfg, network, optimizer, device):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.device = device
        
    def load_model(self):
        raise NotImplementedError


    def save_model(self):
        raise NotImplementedError


    def train_one_epoch(self):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError


    def validate(self):
        raise NotImplementedError