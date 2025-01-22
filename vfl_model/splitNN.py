import torch


class SplitNN(torch.nn.Module):
    def __init__(self, A_waves, passive_clients, label_client):
        super().__init__()
        self.passive_clients = passive_clients
        self.label_client = label_client
        self.pc_optimizers = [pc.get_optimizer() for pc in self.passive_clients]
        self.pc_schedulers = [pc.get_scheduler() for pc in self.passive_clients]
        self.label_optimizer = label_client.get_optimizer()
        self.label_scheduler = label_client.get_scheduler()
        self.A_waves = A_waves

        self.pc_outputs = None

    def forward(self, inputs):
        self.pc_outputs = [pc(self.A_waves, input_pc) for input_pc, pc in zip(inputs, self.passive_clients)]
        outputs = self.label_client(self.pc_outputs)
        return outputs

    def backward(self):
        grads_to_pcs = self.label_client.backward()
        for grads_to_pc, pc in zip(grads_to_pcs, self.passive_clients):
            pc.backward(grads_to_pc)

    def zero_grads(self):
        for pc_optimizer in self.pc_optimizers:
            pc_optimizer.zero_grad()
        self.label_optimizer.zero_grad()

    def step(self):
        for pc_optimizer in self.pc_optimizers:
            pc_optimizer.step()
        self.label_optimizer.step()

    def learning_rate_decay(self):
        for pc_scheduler in self.pc_schedulers:
            pc_scheduler.step()
        self.label_scheduler.step()

    def model_to_device(self, device):
        for pc in self.passive_clients:
            pc.model.to(device=device)
        self.label_client.to(device=device)

    def train(self):
        for pc in self.passive_clients:
            pc.train()
        self.label_client.train()

    def eval(self):
        for pc in self.passive_clients:
            pc.eval()
        self.label_client.eval()
