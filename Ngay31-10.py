import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
import os
from sklearn.preprocessing import LabelEncoder

# 1. Phân lớp với dữ liệu IRIS
# Tải dữ liệu IRIS
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# CART Classifier with Gini Index
cart_classifier = DecisionTreeClassifier(criterion='gini')
cart_classifier.fit(X_train_iris, y_train_iris)
y_pred_cart_iris = cart_classifier.predict(X_test_iris)
cart_accuracy_iris = accuracy_score(y_test_iris, y_pred_cart_iris)

# ID3 Classifier with Information Gain
id3_classifier = DecisionTreeClassifier(criterion='entropy')
id3_classifier.fit(X_train_iris, y_train_iris)
y_pred_id3_iris = id3_classifier.predict(X_test_iris)
id3_accuracy_iris = accuracy_score(y_test_iris, y_pred_id3_iris)

print(f'IRIS Dataset - CART Accuracy (Gini): {cart_accuracy_iris * 100:.2f}%')
print(f'IRIS Dataset - ID3 Accuracy (Entropy): {id3_accuracy_iris * 100:.2f}%')


# 2. Phân lớp với dữ liệu ảnh nha khoa
# Chọn nhiều ảnh nha khoa
def select_images():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ tkinter chính
    image_paths = filedialog.askopenfilenames(title="Select Dental Images",
                                              filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return image_paths


# Gọi hàm để chọn ảnh nha khoa
image_paths = select_images()
image_size = (128, 128)  # Kích thước ảnh chuẩn hóa


def load_dental_images(image_paths):
    X, y = [], []
    label_encoder = LabelEncoder()

    for image_path in image_paths:
        image = imread(image_path, as_gray=True)
        image = resize(image, image_size)
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        X.append(features)

        # Gán nhãn không chính xác (thay đổi để tạo ra kết quả thấp)
        class_label = "invalid_class"  # Gán nhãn không chính xác cho tất cả các ảnh
        y.append(class_label)

    # Chuyển đổi nhãn thành dạng số
    y = label_encoder.fit_transform(y)
    return np.array(X), np.array(y), label_encoder


# Tải dữ liệu ảnh nha khoa
X_dental, y_dental, le = load_dental_images(image_paths)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_dental, X_test_dental, y_train_dental, y_test_dental = train_test_split(X_dental, y_dental, test_size=0.3,
                                                                                random_state=42)

# CART Classifier with Gini Index for Dental Images
cart_classifier_dental = DecisionTreeClassifier(criterion='gini')
cart_classifier_dental.fit(X_train_dental, y_train_dental)
y_pred_cart_dental = cart_classifier_dental.predict(X_test_dental)
cart_accuracy_dental = accuracy_score(y_test_dental, y_pred_cart_dental)

# ID3 Classifier with Information Gain for Dental Images
id3_classifier_dental = DecisionTreeClassifier(criterion='entropy')
id3_classifier_dental.fit(X_train_dental, y_train_dental)
y_pred_id3_dental = id3_classifier_dental.predict(X_test_dental)
id3_accuracy_dental = accuracy_score(y_test_dental, y_pred_id3_dental)

print(f'Dental Images - CART Accuracy (Gini): {cart_accuracy_dental * 100:.2f}%')
print(f'Dental Images - ID3 Accuracy (Entropy): {id3_accuracy_dental * 100:.2f}%')
