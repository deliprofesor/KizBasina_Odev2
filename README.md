# Breast Cancer Classification Project

<img width="1092" height="618" alt="image" src="https://github.com/user-attachments/assets/a2d1e7b8-e93d-45f6-be94-d7dde3c83282" />


## Proje Hakkında

Bu proje, Wisconsin Breast Cancer Dataset (`breast-cancer.csv`) kullanılarak meme kanserinin **iyi huylu (benign)** veya **kötü huylu (malignant)** olarak sınıflandırılması üzerine geliştirilmiştir. Projede amaç, biyometrik özellikler kullanılarak kanser türünü doğru tahmin eden modeller geliştirmek ve performanslarını karşılaştırmaktır.

---

## Veri Seti

- Toplam 569 gözlem ve 33 sütun içerir.
- Hedef değişken: `diagnosis` (B: Benign - iyi huylu, M: Malignant - kötü huylu)
- Diğer sütunlar kanser hücre özelliklerini sayısal olarak temsil eder.
- `id` ve `Unnamed: 32` sütunları çıkarıldı.

---

## Veri Ön İşleme

- `diagnosis` değişkeni sayısallaştırıldı: B → 0, M → 1
- Özellikler `StandardScaler` ile ölçeklendirildi.
- Veri %80 eğitim, %20 test olarak bölündü (stratify ile sınıf dengesine dikkat edildi).
- Eğitim seti üzerinde SMOTE ile sınıf dengesi sağlandı.

---

## Aykırı Değer Analizi

- Bazı özelliklerde aykırı değerler gözlemlendi.
- Aykırı değerler, model başarısını düşürmediği için veri setinden çıkarılmadı.
- Gelecekte aykırı değer temizliği veya winsorizing uygulanabilir.

---

## Modeller ve Performansları

| Model                  | Kernel          | Test Doğruluk (%) |
|------------------------|-----------------|-------------------|
| Linear SVM             | Linear          | 93.8              |
| Radial Basis Function  | RBF             | 92.1              |
| Polynomial SVM         | Polynomial      | 90.3              |
| Sigmoid SVM            | Sigmoid         | 89.4              |
| Balanced Linear SVM*   | Linear + SMOTE  | 94.0              |

\* SMOTE ile dengelenmiş eğitim verisi kullanılmıştır.

---

## Model Performans Analizi

<img width="640" height="480" alt="confusion_matrix_linear_svm" src="https://github.com/user-attachments/assets/b91002d4-1a9d-42b5-8823-2a9abe2d47b2" />

- **Linear SVM** yüksek doğruluk, precision ve recall değerleri gösterdi.
- Hiperparametre optimizasyonuyla **Radial SVM** modeli iyileştirildi (C=10, gamma=0.1).
  
<img width="640" height="480" alt="roc_curve_radial_svm" src="https://github.com/user-attachments/assets/4d6ecdaf-fab2-45c6-8fbc-a4253cabacbc" />

  
- ROC-AUC skorları özellikle Radial SVM için ~0.95 olarak yüksek çıktı.
- SMOTE kullanımı dengesiz veri sorununu gidererek model performansını artırdı.

---

## Görselleştirmeler

- **Scatter Plot:** İki önemli özellik (`radius_mean`, `texture_mean`) ile sınıflar ayrıldı.

<img width="1000" height="500" alt="scatter_radius_texture" src="https://github.com/user-attachments/assets/0589df73-069d-4bbf-a33f-f38b3cccdecf" />


- **Pairplot:** Çoklu özellikler arasındaki ilişkiler ve sınıf dağılımları incelendi.

<img width="824" height="741" alt="pairplot" src="https://github.com/user-attachments/assets/6c16e093-6195-4a17-a6e0-00bcda71a476" />

- **Confusion Matrix:** Modellerin doğru/yanlış sınıflandırma sayıları gösterildi.

# Radial SVM   
<img width="640" height="480" alt="confusion_matrix_radial_svm" src="https://github.com/user-attachments/assets/8216a06d-b51f-4f17-9291-055ee23dd5c6" />

# Polynomial SVM 
<img width="640" height="480" alt="confusion_matrix_polynomial_svm" src="https://github.com/user-attachments/assets/566312ec-8056-4f23-993e-9b4629804ecd" />

# Linear SVM    
<img width="640" height="480" alt="confusion_matrix_linear_svm" src="https://github.com/user-attachments/assets/7376e884-7f2f-459b-a18a-f42e4e15dfc4" />

# Sigmoid SVM     
<img width="640" height="480" alt="confusion_matrix_sigmoid_svm" src="https://github.com/user-attachments/assets/fd68c84d-f753-4689-8455-0dc590aa94b8" />


  
- **ROC Eğrisi:** Model ayrım gücü grafikle ifade edildi.
  
<img width="640" height="480" alt="roc_curve_radial_svm" src="https://github.com/user-attachments/assets/7a6509e4-2a48-477d-b580-ce34db5d3401" />

  
- **Doğruluk Karşılaştırması:** Modellerin test doğrulukları bar grafikte sunuldu.
  
<img width="640" height="480" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/8f5dc94d-9f1f-40dd-977a-bce74847724c" />

---

## Sonuç ve Öneriler

- SMOTE ile dengelenmiş Linear SVM ve optimize edilmiş Radial SVM en başarılı modeller oldu.
- Veri ön işleme ve model optimizasyonları sınıflandırma başarısını artırdı.
- İleri çalışmalarda:
  - Boyut indirgeme (PCA vb.),
  - Model açıklanabilirliği (SHAP, LIME),
  - Farklı algoritmaların denenmesi önerilir.

---

## Kullanım

1. Python 3.x ortamı kurulu olmalı.
2. Gerekli kütüphaneler: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imblearn`
3. Veri dosyasını proje klasörüne koyun: `breast-cancer.csv`
4. Jupyter Notebook veya Python script dosyasını çalıştırın.
5. Kodlar eğitim, değerlendirme ve grafik çıktılarını otomatik üretir.

---

## Referanslar

- [Wisconsin Breast Cancer Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [imblearn (SMOTE) Documentation](https://imbalanced-learn.org/stable/)

---



