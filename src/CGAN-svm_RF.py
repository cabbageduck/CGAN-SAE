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
from cgan import Discriminator, Generator  # 假设cgan.py包含了Generator和Discriminator类定义

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


n_new_samples = 100
new_samples, new_sample_labels = generate_samples(generator, n_new_samples, opt['latent_dim'], opt['n_classes'], device)

new_samples_reshaped = new_samples.reshape(new_samples.shape[0], -1)

n_train_samples = int(n_new_samples * 0.9)
n_test_samples = n_new_samples - n_train_samples

X_train_augmented = np.vstack([X_train, new_samples_reshaped[:n_train_samples]])
y_train_augmented = np.concatenate([y_train, new_sample_labels[:n_train_samples]])

X_test_augmented = np.vstack([X_test, new_samples_reshaped[n_train_samples:]])
y_test_augmented = np.concatenate([y_test, new_sample_labels[n_train_samples:]])

svc_linear = SVC(kernel='linear', random_state=43)
n_features_to_select = 10  # 保留前10个特征
rfe = RFE(estimator=svc_linear, n_features_to_select=n_features_to_select)


rfe.fit(X_train_augmented, y_train_augmented)


X_train_rfe = rfe.transform(X_train_augmented)
X_test_rfe = rfe.transform(X_test_augmented)


svc_poly = SVC(kernel='poly', degree=6, probability=True, random_state=43)
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


from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import interp1d
from scipy.special import expit, logit
unique_fpr, unique_indices = np.unique(fpr, return_index=True)
average_tpr = np.zeros(len(unique_fpr))
for i, idx in enumerate(unique_indices):
    if i < len(unique_indices) - 1:
        mask = (fpr >= unique_fpr[i]) & (fpr < unique_fpr[i+1])
    else:
        mask = (fpr == unique_fpr[i])
    average_tpr[i] = tpr[mask].mean()

epsilon = 1e-6
average_tpr_clipped = np.clip(average_tpr, epsilon, 1 - epsilon)

fpr_smooth = np.linspace(0, 1, 100)
interp_func = Akima1DInterpolator(unique_fpr, average_tpr_clipped)
tpr_smooth = interp_func(fpr_smooth)

tpr_smooth = np.round(tpr_smooth, decimals=6)

plt.figure(figsize=(10, 6))
plt.plot(fpr_smooth, tpr_smooth, label=f'Smoothed ROC curve (area = {auc:.6f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
#plt.show()
# with open('cgan-svm.txt', 'w') as f:
#     f.write(f"Accuracy: {accuracy:.4f}\n")
#     f.write(f"Precision: {precision:.4f}\n")
#     f.write(f"Recall: {recall:.4f}\n")
#     f.write(f"F1 Score: {f1:.4f}\n")
#     f.write(f"AUC: {auc:.4f}\n")
#     f.write("\nFPR and TPR:\n")
#     for fp, tp in zip(fpr_smooth, tpr_smooth):
#         f.write(f"{fp:.4f} {tp:.4f}\n")
#
# print("Results saved to cgan-svm.txt")
from sklearn.utils import shuffle
def permutation_test(model, X, y, n_permutations=1000):
    original_probabilities = model.predict_proba(X)[:, 1]
    original_auc = roc_auc_score(y, original_probabilities)

    permuted_aucs = []
    for i in range(n_permutations):
        y_shuffled = shuffle(y, random_state=i)

        auc_permuted = roc_auc_score(y_shuffled, original_probabilities)
        permuted_aucs.append(auc_permuted)

    p_value = (np.sum(np.array(permuted_aucs) >= original_auc) + 1.0) / (n_permutations + 1)
    return p_value, permuted_aucs, original_auc


p_value, permuted_aucs, original_auc = permutation_test(svc_poly, X_test_rfe, y_test_augmented, n_permutations=1000)

print(f"Original AUC: {original_auc}")
print(f"Permutation Test P-value: {p_value}")

plt.figure(figsize=(10, 6))
plt.hist(permuted_aucs, bins=30, alpha=0.75, label='Permuted AUCs')
plt.axvline(x=original_auc, color='r', linestyle='--', label=f'Original AUC ({original_auc:.3f})')
plt.xlabel('AUC')
plt.ylabel('Frequency')
plt.title('Permutation Test Results')
plt.legend()
#plt.show()

alpha = 0.05
if p_value < alpha:
    print('The classifier\'s performance is significantly different from the baseline.')
else:
    print('The classifier\'s performance is not significantly different from the baseline.')
