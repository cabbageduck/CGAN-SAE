import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import plot_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from cgan import Discriminator, Generator 
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

test_indices = [0, 1] 
train_indices = [2, 3, 5, 4]  

X_train = features_scaled[train_indices]
y_train = labels.iloc[train_indices]

X_test = features_scaled[test_indices]
y_test = labels.iloc[test_indices]

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.values).long())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = {
    'latent_dim': 50, 
    'n_classes': 2,  
    'img_shape': (1, 30)  #
}

generator = Generator(opt['latent_dim'], opt['n_classes'], opt['img_shape']).to(device)
discriminator = Discriminator(opt['n_classes'], opt['img_shape']).to(device)

adversarial_loss = torch.nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 100
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
merged_features = np.vstack([X_train, new_samples_reshaped])

merged_labels = np.concatenate([y_train, new_sample_labels])

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    merged_features, merged_labels, test_size=0.15, random_state=SEED, stratify=merged_labels
)


X_train_cnn_new = X_train_new.reshape(X_train_new.shape[0], X_train_new.shape[1], 1)
X_test_cnn_new = X_test_new.reshape(X_test_new.shape[0], X_test_new.shape[1], 1)

def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=12, kernel_size=4, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=1)(x)
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(1)(x)  
    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (X_train_new.shape[1], 1)
cnn = build_cnn(input_shape)
cnn.compile(loss='mean_squared_error', optimizer='adam')

history = cnn.fit(X_train_cnn_new, y_train_new, epochs=10, batch_size=32, validation_data=(X_test_cnn_new, y_test_new))


feature_extractor = Model(inputs=cnn.input, outputs=cnn.layers[-3].output) 
train_features = feature_extractor.predict(X_train_cnn_new)
test_features = feature_extractor.predict(X_test_cnn_new)


print("Train Features Shape:", train_features.shape)
print("Test Features Shape:", test_features.shape)

dense_layer_weights = cnn.layers[-2].get_weights()[0]
top_features_indices = np.argsort(np.abs(dense_layer_weights).sum(axis=0))[-10:][::-1]


feature_names = features.columns.tolist()
top_gene_names = [feature_names[i] for i in top_features_indices]


print("Top 10 Features by Score:")
for name in top_gene_names:
    print(name)

print("Top 10 Features Indexes:")
print(top_features_indices)


y_pred = cnn.predict(X_test_cnn_new)
mse = np.mean((y_test_new - y_pred.flatten()) ** 2)
print("Mean Squared Error on Test Set:", mse)

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, roc_curve

y_pred_classes = cnn.predict(X_test_cnn_new).flatten()

y_pred_binary = (y_pred_classes > 0.5).astype(int)


auc = roc_auc_score(y_test_new, y_pred_classes) 

accuracy = accuracy_score(y_test_new, y_pred_binary)


f1 = f1_score(y_test_new, y_pred_binary)

recall = recall_score(y_test_new, y_pred_binary)


precision = precision_score(y_test_new, y_pred_binary)


print("AUC:", auc)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Recall:", recall)
print("Precision:", precision)

