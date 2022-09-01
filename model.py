import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size,out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)
    
    def save(self, filename="model.pth"):
        model_folder = "./model"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        filename = os.path.join(model_folder, filename)
        torch.save(self.state_dict, filename)

class QTrainer:

    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)
        
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(state)):
            
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma*torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        self.optimizer.step()
