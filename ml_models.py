import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .utils import generate_random_braid, extract_features, is_unknot
import numpy as np
import networkx as nx
import random

# Hard-coded real knot data from Rolfsen table (braids approx from standard notations)
real_knots = [
    {'braid': [], 'is_unknot': 1},  # Unknot
    {'braid': [1,1,1], 'is_unknot': 0},  # 3_1 trefoil
    {'braid': [1,2,1,-2], 'is_unknot': 0},  # 4_1 figure-eight
    {'braid': [1,1,1,1,1], 'is_unknot': 0},  # 5_1
    {'braid': [1,1,-2,1,-2], 'is_unknot': 0},  # 5_2
    {'braid': [1,2,3,2,1,3], 'is_unknot': 0},  # 6_1 approx
    {'braid': [1,1,1,1,1,1,1], 'is_unknot': 0},  # 7_1
    {'braid': [1, -2, 1, -2, 1, -2], 'is_unknot': 0},  # 6_3 approx
    {'braid': [1]*8, 'is_unknot': 0},  # 8_1 approx
    {'braid': [1,2,-1,2,-1,2], 'is_unknot': 0},  # 6_2 approx
    # Add more from tables as needed (up to ~250 for Rolfsen)
]

class KnotTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)  # Input 10 features

    def forward(self, x):
        return self.fc(x)

    def classify(self, state):
        with torch.no_grad():
            output = self.forward(state)
            prob = torch.softmax(output, dim=1)[0][1].item()
            return prob

class GNNRLPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.actor = nn.Linear(32, 5)
        self.critic = nn.Linear(32, 1)

    def forward(self, state):
        embed = torch.relu(self.fc1(state))
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
    # Add real knots
    for knot in real_knots:
        feats = extract_features(knot['braid'])  # Use proxy extract
        labels.append(knot['is_unknot'])
        features.append(feats)
    print(f"Added {len(real_knots)} real knots from hard-coded table.")
    # Add generated for more data (balance with unknots)
    for _ in range(num_samples):
        braid = generate_random_braid()
        feats = extract_features(braid)
        label = is_unknot(braid)
        features.append(feats)
        labels.append(label)
    
    X = torch.tensor(np.array(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    model = KnotTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'transformer.pth')
    
    # RL Meta-Learning for policy (using generated states as tasks)
    policy = GNNRLPolicy()
    meta_optimizer = optim.Adam(policy.parameters(), lr=0.001)
    inner_lr = 0.01
    tasks = [generate_random_braid() for _ in range(5)]  # Tasks as braids
    for task in tasks:
        adapted_policy = GNNRLPolicy()  # Copy
        adapted_policy.load_state_dict(policy.state_dict())
        inner_optimizer = optim.SGD(adapted_policy.parameters(), lr=inner_lr)
        # Inner loop
        for step in range(2):
            state = torch.tensor([random.random() for _ in range(10)]).float().unsqueeze(0)
            output, value = adapted_policy(state)
            loss = value.mean()  # Dummy loss (replace with real reward for full RL)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        # Outer update
        state = torch.tensor([random.random() for _ in range(10)]).float().unsqueeze(0)
        output, value = adapted_policy(state)
        meta_loss = value.mean()
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
    torch.save(policy.state_dict(), 'gnn_rl.pth')
    print("Training complete; models saved.")
