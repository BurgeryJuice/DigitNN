import torch
import numpy as np
from pathlib import Path
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype("float32")
y = mnist.target.astype(np.int64)
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)
x_train_np = x_train_np / 255.0
x_test_np = x_test_np / 255.0
x_train = torch.tensor(x_train_np, dtype=torch.float32)
x_test = torch.tensor(x_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
y_test = torch.tensor(y_test_np, dtype=torch.long)
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
model = SimpleNN()
crit = nn.CrossEntropyLoss()
optimiz = optim.SGD(model.parameters(), lr=0.5)
epochs = 50
t1 = time.time()
best_acc = 0.0
for epoch in range(epochs):
    running_loss = 0.0
    for batch_x,batch_y in train_loader:
        output = model(batch_x)
        loss = crit(output, batch_y)
        optimiz.zero_grad()
        loss.backward()
        optimiz.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_out = model(x_train)
            train_preds = torch.argmax(train_out, dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            
            test_output = model(x_test)
            test_loss = crit(test_output, y_test)
            test_preds = torch.argmax(test_output, dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            
        best_acc = max(best_acc, test_acc)
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

total_time = time.time() - t1
print(f"Total training time: {total_time:.2f} seconds | Best Test Accuracy: {best_acc*100:.2f}%")
weights_path = Path(__file__).resolve().parent / "mnist_weights.npz"
exportdict = {}
for i,j in model.named_parameters():
    arr = j.detach().numpy()
    if "weight" in i and arr.ndim == 2:
        arr = arr.T
    clean_key = i.replace(".", "_")
    exportdict[clean_key] = arr

layer_weights = [v for k, v in exportdict.items() if "weight" in k]
input_size = layer_weights[0].shape[0]
output_size = layer_weights[-1].shape[1]
hidden_sizes = [w.shape[1] for w in layer_weights[:-1]]
exportdict["input_size"] = np.array([input_size])
exportdict["hidden_sizes"] = np.array(hidden_sizes)
exportdict["output_size"] = np.array([output_size])
exportdict["acc"] = np.array([best_acc * 100])
np.savez(weights_path, **exportdict)

print(f"Model successfully exported with {len(layer_weights)} layers to {weights_path}")