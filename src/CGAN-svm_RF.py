import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader
import torch
from cgan import Discriminator, Generator

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


SEED = 41
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

opt = {
    'latent_dim': 50, 
    'n_classes': 2, 
    'img_shape': (1, 30)  
}

generator = Generator(opt['latent_dim'], opt['n_classes'], opt['img_shape']).to(device)
discriminator = Discriminator(opt['n_classes'], opt['img_shape']).to(device)




def generate_samples(generator, n_samples, latent_dim, n_classes, device):
    z = torch.randn(n_samples, latent_dim).to(device)
    gen_labels = torch.randint(0, n_classes, (n_samples,)).to(device)
    with torch.no_grad():
        gen_imgs = generator(z, gen_labels)
    return gen_imgs.cpu().numpy(), gen_labels.cpu().numpy()

n_new_samples =280
new_samples, new_sample_labels = generate_samples(generator, n_new_samples, opt['latent_dim'], opt['n_classes'], device)


new_samples_reshaped = new_samples.reshape(new_samples.shape[0], -1)


n_train_samples = int(n_new_samples * 0.9)
n_test_samples = n_new_samples - n_train_samples


X_train_augmented = np.vstack([X_train, new_samples_reshaped[:n_train_samples]])
y_train_augmented = np.concatenate([y_train, new_sample_labels[:n_train_samples]])


X_test_augmented = np.vstack([X_test, new_samples_reshaped[n_train_samples:]])
y_test_augmented = np.concatenate([y_test, new_sample_labels[n_train_samples:]])


svc_linear = SVC(kernel='linear', random_state=43)
n_features_to_select = 6
rfe = RFE(estimator=svc_linear, n_features_to_select=n_features_to_select)


rfe.fit(X_train_augmented, y_train_augmented)


X_train_rfe = rfe.transform(X_train_augmented)
X_test_rfe = rfe.transform(X_test_augmented)


svc_poly = SVC(kernel='poly', degree=12, probability=True, random_state=43)
svc_poly.fit(X_train_rfe, y_train_augmented)

predictions = svc_poly.predict(X_test_rfe)
probabilities = svc_poly.predict_proba(X_test_rfe)[:, 1]


accuracy = accuracy_score(y_test_augmented, predictions)
precision = precision_score(y_test_augmented, predictions)
recall = recall_score(y_test_augmented, predictions)
f1 = f1_score(y_test_augmented, predictions)
auc = roc_auc_score(y_test_augmented, probabilities)


fpr, tpr, _ = roc_curve(y_test_augmented, probabilities)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

