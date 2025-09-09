from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import plot_model

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


def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


input_shape = (X_train.shape[1], 1)
cnn = build_cnn(input_shape)
cnn.compile(loss='mean_squared_error', optimizer='adam')


history = cnn.fit(X_train_cnn, y_train, epochs=10,
                  batch_size=32, validation_data=(X_test_cnn, y_test))

feature_extractor = Model(inputs=cnn.input, outputs=cnn.layers[-3].output)
train_features = feature_extractor.predict(X_train_cnn)
test_features = feature_extractor.predict(X_test_cnn)

dense_layer_weights = cnn.layers[-2].get_weights()[0]
top_features_indices = np.argsort(
    np.abs(dense_layer_weights).sum(axis=0))[-10:][::-1]

feature_names = features.columns.tolist()
top_gene_names = [feature_names[i] for i in top_features_indices]


print("Top 10 Features by Score:")
for name in top_gene_names:
    print(name)

print("Top 10 Features Indexes:")
print(top_features_indices)


y_pred = cnn.predict(X_test_cnn)
mse = np.mean((y_test - y_pred.flatten()) ** 2)
print("Mean Squared Error on Test Set:", mse)


y_pred_classes = cnn.predict(X_test_cnn).flatten()

y_pred_binary = (y_pred_classes > 0.5).astype(int)


auc = roc_auc_score(y_test, y_pred_binary)

accuracy = accuracy_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)


print("AUC:", auc)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
