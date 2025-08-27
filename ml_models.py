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
        self.transformer = nn.Transformer(d_model=512, nhead=8)
        self.fc = nn.Linear(512, 2)  # Binary classify

    def forward(self, x):
        x = self.transformer(x, x)
        return self.fc(x.mean(dim=1))

    def classify(self, state):
        with torch.no_grad():
            output = self(state.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0][1].item()
            return prob

class GNNRLPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = nn.Linear(10, 32)  # Simple; use proper GNN like torch_geometric
        self.actor = nn.Linear(32, 5)  # 5 move types
        self.critic = nn.Linear(32, 1)

    def forward(self, graph):
        # Graph features
        feats = torch.tensor([1.0] * 10)  # Dummy
        embed = self.gnn(feats)
        return self.actor(embed), self.critic(embed)

    def predict_moves(self, state, k=5):
        actor, _ = self(state)
        return torch.topk(actor, k).indices.tolist()

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
            outputs = model(inputs.unsqueeze(1))  # Seq dim
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'transformer.pth')
    
    # Similar for GNNRLPolicy (PPO training placeholder)
    policy = GNNRLPolicy()
    torch.save(policy.state_dict(), 'gnn_rl.pth')
