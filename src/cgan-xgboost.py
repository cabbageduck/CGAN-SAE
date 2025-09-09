import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
from cgan import Discriminator, Generator 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

file_path = '../data/tall.csv'
df = pd.read_csv(file_path)

features = df.iloc[:, 1:]  
labels = df.iloc[:, 0] 
feature_names = features.columns.tolist()

print("Features:")
print(features.head())
print("Labels:")
print(labels)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

train_indices = [2, 3, 5, 4]
test_indices = [0, 1]

X_train = features_scaled[train_indices]
y_train = labels.iloc[train_indices]

X_test = features_scaled[test_indices]
y_test = labels.iloc[test_indices]

print("Training set:")
print(y_train)
print("Test set:")
print(y_test)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.values).long())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test.values).long())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

opt = {
    'latent_dim': 50, 
    'n_classes': 2, 
    'img_shape': (1, 30)  
}


generator = Generator(opt['latent_dim'], opt['n_classes'], opt['img_shape']).to(device)
discriminator = Discriminator(opt['n_classes'], opt['img_shape']).to(device)


adversarial_loss = torch.nn.BCEWithLogitsLoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))



num_epochs =100
for epoch in range(num_epochs):
    for i, (real_imgs, real_labels) in enumerate(train_loader):
        batch_size = real_imgs.size(0)

  
        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)

 
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)


        z = torch.randn(batch_size, opt['latent_dim']).to(device)
        gen_labels = torch.randint(0, opt['n_classes'], (batch_size,)).to(device)


        optimizer_G.zero_grad()
        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_validity = discriminator(real_imgs, real_labels)
        real_loss = adversarial_loss(real_validity, valid)
        fake_validity = discriminator(gen_imgs.detach(), gen_labels)
        fake_loss = adversarial_loss(fake_validity, fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
             (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item()))


def generate_samples(generator, n_samples, latent_dim, n_classes, device):
    z = torch.randn(n_samples, latent_dim).to(device)
    gen_labels = torch.randint(0, n_classes, (n_samples,)).to(device)
    with torch.no_grad():
        gen_imgs = generator(z, gen_labels)
    return gen_imgs.cpu().numpy(), gen_labels.cpu().numpy()


n_new_samples = 100 
new_samples, new_sample_labels = generate_samples(generator, n_new_samples, opt['latent_dim'], opt['n_classes'], device)

new_samples_reshaped = new_samples.reshape(new_samples.shape[0], -1)

n_train_samples = int(n_new_samples * 0.9)
n_test_samples = n_new_samples - n_train_samples

X_train_augmented = np.vstack([X_train, new_samples_reshaped[:n_train_samples]])
y_train_augmented = np.concatenate([y_train, new_sample_labels[:n_train_samples]])

X_test_augmented = np.vstack([X_test, new_samples_reshaped[n_train_samples:]])
y_test_augmented = np.concatenate([y_test, new_sample_labels[n_train_samples:]])

dtrain = xgb.DMatrix(X_train_augmented, label=y_train_augmented)
dtest = xgb.DMatrix(X_test_augmented, label=y_test_augmented)

param = {
    'max_depth': 4,  
    'eta': 0.1,      
    'objective': 'binary:logistic'
}

num_round = 50 
bst = xgb.train(param, dtrain, num_round)

predictions = bst.predict(dtest)

predicted_labels = [int(pred > 0.5) for pred in predictions]

print("Predicted labels:", predicted_labels)

accuracy = accuracy_score(y_test_augmented, predicted_labels)
f1 = f1_score(y_test_augmented, predicted_labels)
precision = precision_score(y_test_augmented, predicted_labels)
recall = recall_score(y_test_augmented, predicted_labels)
roc_auc = roc_auc_score(y_test_augmented, predictions)
conf_matrix = confusion_matrix(y_test_augmented, predicted_labels)

fpr, tpr, _ = roc_curve(y_test_augmented, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

