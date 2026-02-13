import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

# ==========================================
# 1. CONFIGURATION & DATA GENERATION
# ==========================================

def generate_distributed_data(num_clients=3):
    """
    Simulates global transaction data and shards it into local 'Bank' datasets.
    Banks never share this raw data with each other or the server.
    """
    # Generate 5000 transactions. 20 features each.
    # Weights=[0.9, 0.1] means 10% of transactions are Fraud.
    X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, 
                               weights=[0.9, 0.1], random_state=42)
    
    # Standardize data (common in finance)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Hold out 20% for global server testing
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Split the remaining 80% among the 3 Banks
    chunk_size = len(X_train) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        dataset = TensorDataset(X_train[start:end], y_train[start:end])
        client_datasets.append(dataset)
        
    test_dataset = TensorDataset(X_test, y_test)
    
    return client_datasets, test_dataset

# ==========================================
# 2. MODEL DEFINITION
# ==========================================

class FraudModel(nn.Module):
    def __init__(self):
        super(FraudModel, self).__init__()
        # Input: 20 features -> Hidden: 64 neurons -> Output: 1 (Fraud Prob)
        self.layer1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer2(x))
        return x

# ==========================================
# 3. CLIENT (BANK) LOGIC
# ==========================================

class BankNode:
    def __init__(self, dataset, client_id):
        self.client_id = client_id
        self.dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model = FraudModel()
        self.criterion = nn.BCELoss() # Binary Cross Entropy for Fraud/Not-Fraud
        
    def client_update(self, global_weights, epochs=2):
        """
        Receives global weights, trains locally, returns UPDATED weights.
        """
        self.model.load_state_dict(global_weights)
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.model.train()
        
        epoch_loss = 0
        for _ in range(epochs):
            for X_batch, y_batch in self.dataloader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        # Return only the weights (State Dict), NOT the data
        return self.model.state_dict(), epoch_loss / len(self.dataloader)

# ==========================================
# 4. SERVER (AGGREGATOR) LOGIC
# ==========================================

def federated_average(weight_list):
    """
    Mathematically averages the weights from all banks.
    New_Global = (Bank_A + Bank_B + Bank_C) / 3
    """
    avg_weights = copy.deepcopy(weight_list[0])
    
    for key in avg_weights.keys():
        for i in range(1, len(weight_list)):
            avg_weights[key] += weight_list[i][key]
        # Divide by number of clients to get mean
        avg_weights[key] = torch.div(avg_weights[key], len(weight_list))
        
    return avg_weights

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            predicted = (outputs > 0.5).float()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Setup
    NUM_ROUNDS = 5
    NUM_BANKS = 3
    print(f"--- Setting up Federated System with {NUM_BANKS} Banks ---")
    
    # 1. Distribute Data
    bank_datasets, global_test_data = generate_distributed_data(NUM_BANKS)
    test_loader = DataLoader(global_test_data, batch_size=64)
    
    # 2. Initialize Nodes
    global_model = FraudModel()
    global_weights = global_model.state_dict() # The "Master" Model
    
    banks = [BankNode(data, i) for i, data in enumerate(bank_datasets)]
    
    # 3. Training Loop
    print(f"--- Starting Training for {NUM_ROUNDS} Rounds ---")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        local_weights = []
        local_losses = []
        
        # A. Send global weights to banks & train locally
        for bank in banks:
            # Deepcopy ensures server weights aren't modified by reference
            w_node, loss_node = bank.client_update(copy.deepcopy(global_weights), epochs=3)
            local_weights.append(w_node)
            local_losses.append(loss_node)
        
        # B. Server Aggregation (FedAvg)
        global_weights = federated_average(local_weights)
        
        # C. Update Global Model
        global_model.load_state_dict(global_weights)
        
        # D. Evaluate Global Model
        avg_loss = sum(local_losses) / len(local_losses)
        accuracy = evaluate(global_model, test_loader)
        
        print(f"Round {round_num}: Avg Training Loss: {avg_loss:.4f} | Global Model Accuracy: {accuracy*100:.2f}%")

    print("--- Federated Fraud Detection Training Complete ---")