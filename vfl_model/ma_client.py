import torch
from torch import nn, optim
from InitParas import InitParameters


class ActiveClient(nn.Module):
    def __init__(self, num_features, num_passive_clients):
        super().__init__()
        self.optimizer = None
        self.model = LabelNet(num_features, num_passive_clients)
        self.pc_outputs = None
        self.grads_from_label = None

    def forward(self, pc_outputs):
        self.pc_outputs = pc_outputs
        outputs = self.model(torch.cat(pc_outputs, dim=2))
        return outputs

    def backward(self) -> list[torch.Tensor]:
        self.grads_from_label = [pc_output.grad.clone() for pc_output in self.pc_outputs]
        return self.grads_from_label

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


class LabelNet(nn.Module):
    def __init__(self, num_features, num_passive_clients):
        super().__init__()
        self.fc1 = nn.Linear((InitParameters.NUM_STEPS_INPUT - 1 * 5) * 64 * num_passive_clients,
                             InitParameters.NUM_STEPS_OUTPUT * num_features)

    def forward(self, x):
        return self.fc1(x)
