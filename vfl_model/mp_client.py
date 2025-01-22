from torch import nn
from torch import optim
from vfl_model.stgcn_vfl import STGCNVFL
from InitParas import InitParameters
from codecarbon import EmissionsTracker


class PassiveClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.model = STGCNVFL(17, 2)
        self.output = None
        self.grad_from_label = None

    def forward(self, A_waves, inputs):
        # tracker = EmissionsTracker()
        # # Start energy monitoring
        # tracker.start()

        self.output = self.model(A_waves, inputs)
        output = self.output.detach().requires_grad_()
        #
        # # Stop monitor and collect the energy cost data
        # emissions = tracker.stop()
        # # print energy cost info
        # print(f"Estimated energy consumption: {emissions:.8f} kWh")

        return output

    def backward(self, grad_from_label):
        self.grad_from_label = grad_from_label
        self.output.backward(grad_from_label)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def get_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=2e-4, weight_decay=5e-3)
        self.optimizer = optimizer
        return optimizer

    def get_scheduler(self):
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        return scheduler

