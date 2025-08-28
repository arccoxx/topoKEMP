import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .utils import generate_random_braid, extract_features, is_unknot
import numpy as np
import networkx as nx

class KnotTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)  # Dummy for list input

    def forward(self, x):
        return self.fc(torch.tensor(x))

    def classify(self, state):
        with torch.no_grad():
            output = self.forward(state)
            prob = torch.softmax(output, dim=0)[1].item()
            return prob

class GNNRLPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.actor = nn.Linear(32, 5)
        self.critic = nn.Linear(32, 1)

    def forward(self, state):
        embed = torch.relu(self.fc1(torch.tensor(state)))
        return self.actor(embed), self.critic(embed)

    def classify(self, state):
        with torch.no_grad():
            output, _ = self.forward(state)
            prob = torch.softmax(output, dim=1)[0][1].item()
            return prob

    def predict_moves(self, state, k=5):
        output, _ = self.forward(state)
        return torch.topk(output, k).indices.tolist()

def train_ml_models(num_samples=1000, epochs=50):
    features, labels = [], []
    for _ in range(num_samples):
        knot = generate_random_braid()
        if knot:
            feats = extract_features(knot)
            label = 1 if is_unknot(knot) else 0
            features.append(feats)
            labels.append(label)
    
    X = torch.tensor(np.array(features))
    y = torch.tensor(labels, dtype=torch.long)
    
    model = KnotTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'transformer.pth')
    
    # Similar for GNNRLPolicy (PPO training placeholder)
    policy = GNNRLPolicy()
    torch.save(policy.state_dict(), 'gnn_rl.pth')
