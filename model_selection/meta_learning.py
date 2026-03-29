import torch
import torch.nn as nn
class MetaModel(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=3):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
if __name__ == "__main__":
    import json
    import numpy as np
    import random
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        with open("meta_training_data.json", "r") as f:
            meta_learning_data = json.load(f)
    except FileNotFoundError:
        print("❌ Dataset not found. Make sure `meta_training_data.json` exists.")
        exit()
    class MetaLearningDataset(Dataset):
        def __init__(self, data):
            self.features = []
            self.labels = []
            correction_model_mapping = {
                "CycleGAN (Motion Correction)": 0,
                "CycleGAN (Denoising)": 1,
                "CycleGAN (Reconstruction)": 2
            }
            for sample in data:
                feature_vector = (
                    sample["true_corruption"] +
                    sample["ai1_classification"] +
                    sample["ai2_classification"] +
                    sample["radiologist_classification"] +
                    sample["pi_classification"]
                )
                self.features.append(np.array(feature_vector, dtype=np.float32))
                self.labels.append(correction_model_mapping[sample["final_model"]])
        def __len__(self):
            return len(self.features)
        def __getitem__(self, index):
            return self.features[index], self.labels[index]
    dataset = MetaLearningDataset(meta_learning_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in dataloader:
            features, labels = torch.tensor(features).to(device), torch.tensor(labels).to(device).long()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1}: Loss {running_loss/len(dataset):.4f}")
    torch.save(model.state_dict(), "meta_model_best.pth")
    print("✅ Model trained and saved!")
