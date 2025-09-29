import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score,roc_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.nn.functional import sigmoid
from cgan import Discriminator, Generator  #
def evaluate_model(discriminator, test_loader, device):
    discriminator.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for real_imgs, real_labels in test_loader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            outputs = discriminator(real_imgs, real_labels).view(-1)

            probabilities = sigmoid(outputs).cpu().numpy()

            all_preds.extend(probabilities)
            all_labels.extend(real_labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate_roc_auc(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for real_imgs, real_labels in data_loader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)


            outputs = model(real_imgs, real_labels).view(-1)


            probabilities = sigmoid(outputs).cpu().numpy()

            all_preds.extend(probabilities)
            all_labels.extend(real_labels.cpu().numpy())


    roc_auc = roc_auc_score(all_labels, all_preds)
    return roc_auc


SEED = 42
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


train_indices=[2, 3, 5, 4]
test_indices=[0, 1]


X_train = features_scaled[train_indices]
y_train = labels.iloc[train_indices]

X_test = features_scaled[test_indices]
y_test = labels.iloc[test_indices]
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
    merged_features, merged_labels, test_size=0.1, random_state=SEED, stratify=merged_labels
)

model = RandomForestClassifier(
    n_estimators=500,
    criterion="entropy",
    max_depth=42,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1
)

model.fit(X_train_new, y_train_new)


feature_importances = model.feature_importances_
important_features_indices = np.argsort(feature_importances)[::-1]


top_features = important_features_indices[:10]
feature_names = features.columns.tolist()
top_gene_names = [feature_names[i] for i in top_features]



plt.figure(figsize=(10, 6))
plt.bar(range(len(top_gene_names)), feature_importances[top_features], tick_label=top_gene_names)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances from Random Forest')
#plt.show()


y_pred = model.predict(X_test_new)

acc = accuracy_score(y_test_new, y_pred)
print(" (Accuracy):", acc)

f1 = f1_score(y_test_new, y_pred, average='weighted')
print("F1:", f1)

y_prob = model.predict_proba(X_test_new)[:, 1]  # 取正类的概率
auc = roc_auc_score(y_test_new, y_prob)
print("AUC:", auc)

recall = recall_score(y_test_new, y_pred, average='weighted')
print("(Recall):", recall)


precision = precision_score(y_test_new, y_pred, average='weighted')
print("(Precision):", precision)


fpr, tpr, _ = roc_curve(y_test_new, y_prob)


df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
print(df)
from scipy.interpolate import interp1d

unique_fpr, unique_indices = np.unique(fpr, return_index=True)
average_tpr = np.zeros(len(unique_fpr))
for i, idx in enumerate(unique_indices):
    if i < len(unique_indices) - 1:
        mask = (fpr >= unique_fpr[i]) & (fpr < unique_fpr[i+1])
    else:
        mask = (fpr == unique_fpr[i])
    average_tpr[i] = tpr[mask].mean()


fpr_smooth = np.linspace(0, 1, 100)
tpr_smooth = interp1d(unique_fpr, average_tpr, kind='cubic')(fpr_smooth)


plt.figure(figsize=(10, 6))
plt.plot(fpr_smooth, tpr_smooth, label=f'Smoothed ROC curve (area = {auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
#plt.show()


from sklearn.utils import shuffle



def permutation_test(model, X, y, n_permutations=1000):
    original_scores = model.predict_proba(X)[:, 1]  # 使用概率预测值
    original_auc = roc_auc_score(y, original_scores)

    permuted_aucs = []
    for i in range(n_permutations):

        y_shuffled = shuffle(y, random_state=i)

        auc_permuted = roc_auc_score(y_shuffled, original_scores)
        permuted_aucs.append(auc_permuted)

    p_value = (np.sum(np.array(permuted_aucs) >= original_auc) + 1.0) / (n_permutations + 1)
    return p_value, permuted_aucs, original_auc



p_value, permuted_aucs, original_auc = permutation_test(model, X_test_new, y_test_new, n_permutations=1000)

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


