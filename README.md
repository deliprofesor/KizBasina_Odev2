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
- Eksik veri bulunmamaktadır.

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

- **Linear SVM** yüksek doğruluk, precision ve recall değerleri gösterdi.
- Hiperparametre optimizasyonuyla **Radial SVM** modeli iyileştirildi (C=10, gamma=0.1).
- ROC-AUC skorları özellikle Radial SVM için ~0.95 olarak yüksek çıktı.
- SMOTE kullanımı dengesiz veri sorununu gidererek model performansını artırdı.

---

## Görselleştirmeler

- **Scatter Plot:** İki önemli özellik (`radius_mean`, `texture_mean`) ile sınıflar ayrıldı.
- **Pairplot:** Çoklu özellikler arasındaki ilişkiler ve sınıf dağılımları incelendi.
- **Confusion Matrix:** Modellerin doğru/yanlış sınıflandırma sayıları gösterildi.
- **ROC Eğrisi:** Model ayrım gücü grafikle ifade edildi.
- **Doğruluk Karşılaştırması:** Modellerin test doğrulukları bar grafikte sunuldu.

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

## İletişim

Projeyle ilgili soru, öneri ve geri bildirim için:  
**ilknur@example.com**

---

*Teşekkürler!*

