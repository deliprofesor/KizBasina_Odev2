# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

# Grafiklerin kaydedileceği klasör
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Veri setini yükleme
modelData = pd.read_csv("breast-cancer.csv")

# Sütun adlarını göster
print("Column Names:\n", modelData.columns)

# Eksik veri kontrolü
print("Missing Data Check:\n", modelData.isnull().sum())

# Gereksiz sütunları kaldır
if "X" in modelData.columns:
    modelData = modelData.drop(columns=['X'])
if "id" in modelData.columns:
    modelData = modelData.drop(columns=['id'])

# Seçilen sütunlar
selectedVars = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
                "symmetry_mean", "fractal_dimension_mean"]
modelDataSelected = modelData[selectedVars]

# Kategorik hedef değişkeni (diagnosis) sayısallaştırma
modelDataSelected["diagnosis"] = modelDataSelected["diagnosis"].map({"B": 0, "M": 1})

# Görselleştirme: Scatter Plot
plt.figure(figsize=(10, 5))
plt.scatter(modelDataSelected[modelDataSelected["diagnosis"] == 0]["radius_mean"],
            modelDataSelected[modelDataSelected["diagnosis"] == 0]["texture_mean"],
            c="blue", marker="o", label="Benign")
plt.scatter(modelDataSelected[modelDataSelected["diagnosis"] == 1]["radius_mean"],
            modelDataSelected[modelDataSelected["diagnosis"] == 1]["texture_mean"],
            c="orange", marker="o", label="Malignant")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend(loc="best")
plt.title("Scatter Plot: Radius Mean vs. Texture Mean")
plt.savefig(os.path.join(output_dir, "scatter_radius_texture.png"))  # Kaydet
plt.show()

# Pairplot ile görselleştirme
pairplot_fig = sns.pairplot(modelDataSelected, hue="diagnosis",
                            vars=["radius_mean", "texture_mean", "area_mean"])
pairplot_fig.savefig(os.path.join(output_dir, "pairplot.png"))
plt.show()

# Model oluşturma: Özellikler ve hedef değişken
X = modelDataSelected.drop(columns=["diagnosis"])
y = modelDataSelected["diagnosis"]

# Standartlaştırma
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=125)

# Modeller
modelLinear = svm.SVC(kernel='linear', probability=True)
modelRadial = svm.SVC(kernel='rbf', probability=True)
modelPoly = svm.SVC(kernel='poly', probability=True)
modelSigmoid = svm.SVC(kernel='sigmoid', probability=True)

# Model eğitimleri
modelLinear.fit(X_train, y_train)
modelRadial.fit(X_train, y_train)
modelPoly.fit(X_train, y_train)
modelSigmoid.fit(X_train, y_train)

# Tahminler
predLinear = modelLinear.predict(X_test)
predRadial = modelRadial.predict(X_test)
predPoly = modelPoly.predict(X_test)
predSigmoid = modelSigmoid.predict(X_test)

# Performans: Confusion Matrix ve Classification Report
models = {"Linear SVM": predLinear, "Radial SVM": predRadial,
          "Polynomial SVM": predPoly, "Sigmoid SVM": predSigmoid}

for model_name, predictions in models.items():
    print(f"\n{model_name} Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"))
    plt.show()

# Hiperparametre optimizasyonu
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid_radial = GridSearchCV(svm.SVC(kernel='rbf', probability=True), param_grid, verbose=1, cv=5)

grid_radial.fit(X_train, y_train)

# En iyi parametreler
print("Best Parameters (Radial SVM):", grid_radial.best_params_)

# Optimizasyon sonrası model
best_radial = grid_radial.best_estimator_
best_pred_radial = best_radial.predict(X_test)
print("\nBest Radial SVM Classification Report:\n", classification_report(y_test, best_pred_radial))

# SMOTE ile sınıf dengesini sağlama
print("Before SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_train_balanced))

# SMOTE sonrası eğitim
modelLinear_balanced = svm.SVC(kernel='linear', probability=True)
modelLinear_balanced.fit(X_train_balanced, y_train_balanced)
predLinear_balanced = modelLinear_balanced.predict(X_test)

print("Balanced Linear SVM Classification Report:\n", classification_report(y_test, predLinear_balanced))

# ROC-AUC Skoru ve ROC Eğrisi
y_test_binary = y_test
pred_prob = modelRadial.decision_function(X_test)
roc_auc = roc_auc_score(y_test_binary, pred_prob)
fpr, tpr, thresholds = roc_curve(y_test_binary, pred_prob)

plt.plot(fpr, tpr, label=f"Radial SVM (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve_radial_svm.png"))
plt.show()


# Modellerin Doğruluk Karşılaştırması
accuracy_scores = {
    "Linear": modelLinear.score(X_test, y_test),
    "Radial": modelRadial.score(X_test, y_test),
    "Polynomial": modelPoly.score(X_test, y_test),
    "Sigmoid": modelSigmoid.score(X_test, y_test),
}

plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=["blue", "green", "orange", "red"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
plt.show()

# Sonuçların Tablosu
results = pd.DataFrame({
    "Model": ["Linear", "Radial", "Polynomial", "Sigmoid"],
    "Accuracy": [modelLinear.score(X_test, y_test),
                 modelRadial.score(X_test, y_test),
                 modelPoly.score(X_test, y_test),
                 modelSigmoid.score(X_test, y_test)]
})
print(results)
