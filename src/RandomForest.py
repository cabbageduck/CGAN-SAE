import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

file_path = '../data/tall.csv'
df = pd.read_csv(file_path)

print(df.head())


features = df.iloc[:, 1:]
labels = df.iloc[:, 0]

print("Features:")
print(features.head())
print("Labels:")
print(labels)


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

features_scaled_df = pd.DataFrame(
    features_scaled, index=features.index, columns=features.columns)


selector = SelectKBest(score_func=f_classif, k=10)

selected_features = selector.fit_transform(features_scaled_df, labels)

train_indices = [0, 1]
test_indices = [2, 3, 5, 4]

X_train = features_scaled[train_indices]
y_train = labels.iloc[train_indices]

X_test = features_scaled[test_indices]
y_test = labels.iloc[test_indices]

model = RandomForestClassifier(
    n_estimators=500,
    criterion="entropy",
    max_depth=42,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    class_weight=None
)
model.fit(X_train, y_train)
if model.oob_score:
    print("Out-of-bag score estimate:", model.oob_score_)

feature_importances = model.feature_importances_
print("Feature importances:", feature_importances)

important_features_indices = np.argsort(feature_importances)[::-1]
print("Indices of top important features:", important_features_indices[:10])

top_features = important_features_indices[:10]
print("Top 10 important features:", top_features)

feature_names = features.columns.tolist()
top_gene_names = [feature_names[i] for i in top_features]
print("Names of top 10 important features:", top_gene_names)


y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("(Accuracy):", acc)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1:", f1)


y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)
