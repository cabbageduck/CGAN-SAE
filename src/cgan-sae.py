import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn

from sklearn.utils import shuffle
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve,precision_score

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn.functional import sigmoid
from cgan import Discriminator, Generator 
import torch.nn.functional as F
from axial import  *

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
type=0
file_path = '../data/tall.csv'
df = pd.read_csv(file_path)
if "tall.csv" not in file_path:
    type=1

features = df.iloc[:, 1:]
labels = df.iloc[:, 0]
feature_names = features.columns.tolist()
if type==1:
    features = df.iloc[:, 1:-2]
    labels = df.iloc[:, 0]
    feature_names = features.columns.tolist()


print("Features:")
print(features.head())
print("Labels:")
print(labels)


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


train_indices=[2, 3, 5, 4]
test_indices=[0, 1]

X_train = features_scaled[train_indices]
y_train = labels.iloc[train_indices]

X_test = features_scaled[test_indices]
y_test = labels.iloc[test_indices]

if "tall.csv" not in file_path:
    type=1
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)

print("Training set:")
print(y_train)
print("Test set:")
print(y_test)


train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.values).long())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test.values).long())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


opt = {
    'latent_dim': 50,
    'n_classes': 2,
    'img_shape': (1, 30)

}
if type == 1:
    opt['img_shape'] = (1, 15260)

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

        if epoch%20==0:
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

train_dataset_augmented = TensorDataset(torch.from_numpy(X_train_augmented).float(), torch.from_numpy(y_train_augmented).long())
train_loader_augmented = DataLoader(train_dataset_augmented, batch_size=30, shuffle=True)

test_dataset_augmented = TensorDataset(torch.from_numpy(X_test_augmented).float(), torch.from_numpy(y_test_augmented).long())
test_loader_augmented = DataLoader(test_dataset_augmented, batch_size=30, shuffle=False)
N=1

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_ratio):
        super(SparseAutoencoder, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, hidden_size),
            AxialAttention(in_planes=input_size, out_planes=hidden_size, groups=1)
        )
        self.drop=nn.Dropout(0.3)
        self.linear1=nn.Linear(input_size, 90)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(90, hidden_size)
        self.axial = AxialAttention(in_planes=input_size, out_planes=hidden_size, groups=N)
        self.encoder = AxialAttention(in_planes=input_size,out_planes=hidden_size,groups=N)
        self.decoder = AxialAttention(in_planes=input_size,out_planes=hidden_size,groups=N)
        self.sparsity_ratio = sparsity_ratio
    def forward(self, x):
        x=self.linear1(x)
        x = self.drop(x)
        x=self.relu(x)
        x = self.linear2(x)
        b, n = x.size()
        x = x.view(b, n, 1, 1)
        encoded = self.encoder(x)
        b, c, h, w = encoded.size()
        x = encoded.view(b, c)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.relu(x)
        decoded = self.decoder(encoded)
        b, n,w,h =decoded.size()
        decoded = decoded.view(b, n)
        return decoded, encoded

    def sparse_loss(self, encoded):
        epsilon = 1e-10
        sparsity = torch.mean(encoded, dim=0)

        sparsity = torch.clamp(sparsity, epsilon, 1 - epsilon)

        kl_div = self.sparsity_ratio * torch.log(self.sparsity_ratio / sparsity) + \
                 (1 - self.sparsity_ratio) * torch.log((1 - self.sparsity_ratio) / (1 - sparsity))
        return kl_div.sum()

input_size = 30
hidden_size = 30
if type==1:
    input_size = 15260
    hidden_size = 15260
sparsity_ratio = 0.01

sparse_autoencoder = SparseAutoencoder(input_size, hidden_size, sparsity_ratio).to(device)


reconstruction_criterion = nn.MSELoss()
optimizer_sparse = optim.Adam(sparse_autoencoder.parameters(), lr=0.01)

num_epochs_sparse =100
batch_size_sparse = 12
sparse_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_augmented).float()), batch_size=batch_size_sparse,
                           shuffle=True,drop_last=False)

for epoch in range(num_epochs_sparse):
    for data in sparse_loader:
        inputs = data[0].to(device)
        optimizer_sparse.zero_grad()


        outputs, encoded = sparse_autoencoder(inputs)


        reconstruction_loss = reconstruction_criterion(outputs, inputs)
        sparsity_penalty = sparse_autoencoder.sparse_loss(encoded)
        loss = reconstruction_loss + sparsity_penalty


        loss.backward()
        optimizer_sparse.step()
    if (epoch%20==0):
        print(f'Sparse Autoencoder Epoch [{epoch + 1}/{num_epochs_sparse}], Loss: {loss.item()}')


sparse_autoencoder.eval()


all_reconstruction_errors = []
all_true_labels = []

with torch.no_grad():
    for data in test_loader_augmented:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)


        reconstructed_data, encoded = sparse_autoencoder(inputs)


        reconstruction_error = torch.mean((inputs - reconstructed_data) ** 2, dim=1).cpu().numpy()
        all_reconstruction_errors.extend(reconstruction_error)
        all_true_labels.extend(labels.cpu().numpy())


all_reconstruction_errors = np.array(all_reconstruction_errors)
all_true_labels = np.array(all_true_labels)

original_auc = roc_auc_score(all_true_labels, all_reconstruction_errors)

from sklearn.utils import shuffle


def permutation_test(reconstruction_errors, true_labels, n_permutations=1000):
    permuted_aucs = []
    original_scores = reconstruction_errors

    for i in range(n_permutations):

        shuffled_labels = shuffle(true_labels, random_state=i)


        auc_permuted = roc_auc_score(shuffled_labels, original_scores)
        permuted_aucs.append(auc_permuted)

    p_value = (np.sum(np.array(permuted_aucs) >= original_auc) + 1.0) / (n_permutations + 1)
    return p_value, permuted_aucs



p_value, permuted_aucs = permutation_test(all_reconstruction_errors, all_true_labels, n_permutations=1000)

print(f"Original AUC: {original_auc}")
print(f"Permutation Test P-value: {p_value}")

fpr, tpr, thresholds = roc_curve(all_true_labels, all_reconstruction_errors,drop_intermediate=False)
roc_auc = roc_auc_score(all_true_labels, all_reconstruction_errors)


optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f'Optimal Threshold: {optimal_threshold}')


predicted_labels = (all_reconstruction_errors > optimal_threshold).astype(int)


accuracy = accuracy_score(all_true_labels, predicted_labels)
print(f'Accuracy: {accuracy}')


auc = roc_auc
print(f'AUC: {auc}')


recall = recall_score(all_true_labels, predicted_labels)
print(f'Recall: {recall}')


f1 = f1_score(all_true_labels, predicted_labels)
print(f'F1 Score: {f1}')

pre=precision_score(all_true_labels, predicted_labels)
print(f'precision Score: {pre}')


encoded_features = []

with torch.no_grad():
    for data in sparse_loader:
        inputs = data[0].to(device)
        _, encoded = sparse_autoencoder(inputs)
        b,n,h,w=encoded.size()
        encoded=encoded.view(b,n)
        encoded_features.append(encoded.cpu().numpy())

encoded_features = np.concatenate(encoded_features, axis=0)

pca = PCA(n_components=2)

pca_features = pca.fit_transform(encoded_features)

plt.figure(figsize=(10, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1])
plt.title('PCA of Encoded Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#plt.show()





