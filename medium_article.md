# Çevresel Sensörlerle Bir Odada Kaç Kişi Olduğu Tahmin Edilebilir mi? Uçtan Uca Bir Makine Öğrenmesi Yaklaşımı

![Oda Doluluk Tahmini](outputs/figures/05_temporal_trend.png)

---

## 1. Giriş (Introduction)

Günümüzde akıllı binalar ve nesnelerin interneti (IoT) sistemleri, enerji tasarrufu sağlamak ve iç mekân konforunu artırmak için giderek daha fazla veri odaklı hale gelmektedir. Isıtma, havalandırma ve klima (HVAC) sistemleri, aydınlatma otomasyonları ve acil durum tahliye planları, bir mekânda anlık olarak **kaç kişinin bulunduğuna** doğrudan bağımlıdır.

Ancak oda doluluğunu tespit etmek için kamera veya mikrofon gibi invaziv (mahremiyeti ihlal eden) sistemlerin kullanılması ciddi gizlilik ve etik endişeleri beraberinde getirmektedir. Bu noktada akla şu soru gelmektedir:

> **"Kameralara ihtiyaç duymadan; yalnızca sıcaklık, ışık, ses, $\text{CO}_2$ ve hareket gibi pasif çevresel sensör verilerini kullanarak bir odadaki kişi sayısını ($0, 1, 2, 3$ kişi) ne kadar doğru tahmin edebiliriz?"**

Bu çalışmada, UCI Machine Learning Repository'den alınan gerçek zamanlı sensör verileriyle uçtan uca, metodolojik olarak sağlam ve veri sızıntısından arındırılmış bir makine öğrenmesi akışı inşa ettik.

---

## 2. Veri Seti (Dataset)

Çalışmamızda, California Üniversitesi Irvine (UCI) makine öğrenmesi havuzunda yer alan [Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) veri setini kullandık.

* **Toplam Gözlem Sayısı:** 10,129 satır
* **Sensör Kanalları:** 16 nümerik sensör kanalı
* **Zaman Aralığı:** 22 Aralık 2017 – 11 Ocak 2018
* **Örnekleme Sıklığı:** Yaklaşık 31 saniyede bir ölçüm
* **Hedef Değişken (`Room_Occupancy_Count`):** 
  * `0`: Boş oda (8,228 kayıt - %81.23)
  * `1`: 1 kişi (459 kayıt - %4.53)
  * `2`: 2 kişi (748 kayıt - %7.38)
  * `3`: 3 kişi (694 kayıt - %6.85)

Problem, belirgin sınıf dengesizliği (class imbalance) içeren bir **çok sınıflı sınıflandırma (multiclass classification)** problemidir.

---

## 3. Sensörleri ve Değişkenleri Tanıma

Veri setinde 4 farklı konuma yerleştirilmiş sensör dizilimleri bulunmaktadır:

* **Sıcaklık ($S_1\text{--}S_4\text{ Temp}$):** Odanın farklı noktalarındaki ortam sıcaklığı (°C).
* **Işık ($S_1\text{--}S_4\text{ Light}$):** Farklı köşelerdeki ışık şiddeti (Lux).
* **Ses ($S_1\text{--}S_4\text{ Sound}$):** Ortam gürültü ve ses voltaj sinyalleri.
* **Karbondioksit ($S_5\text{ CO}_2$ ve $S_5\text{ CO}_2\text{ Slope}$):** $\text{CO}_2$ konsantrasyonu (ppm) ve $\text{CO}_2$'nin zamana göre türevi/değişim eğimi.
* **PIR Hareket Sensörleri ($S_6\text{ PIR}, S_7\text{ PIR}$):** Pasif kızılötesi hareket algılama (0: Hareket yok, 1: Hareket var).

Tarih (`Date`) ve saat (`Time`) bilgileri doğrudan modele verilmemeli, zamansal sıralama ve döngüsel öznitelik çıkarımı için kullanılmalıdır.

---

## 4. Keşifçi Veri Analizi (Exploratory Data Analysis - EDA)

Veri temizleme aşamasında veride **0 eksik değer (NaN)**, **0 sonsuz değer (Inf)** ve **0 yinelenen satır** tespit edildi. Ardından veriyi anlamak için beş temel görselleştirme gerçekleştirdik.

### 4.1 Sınıf Dağılımı ve Dengesizlik
![Hedef Sınıf Dağılımı](outputs/figures/01_class_distribution.png)
Verinin %81.2'si boş oda gözlemlerinden oluşmaktadır. Bu durum, model performansını değerlendirirken yalnızca **Accuracy** (Doğruluk) metriğine güvenmenin yanıltıcı olacağını; **Macro Precision**, **Macro Recall** ve **Macro F1-Score** metriklerinin temel alınması gerektiğini net bir şekilde ortaya koymaktadır.

### 4.2 Sensör Dağılımları
![Sensör Dağılımları](outputs/figures/02_sensor_distributions.png)
Işık ve ses sensörlerinde sağa çarpık (right-skewed) dağılımlar görülürken, $\text{CO}_2$ konsantrasyonu insan varlığına bağlı olarak keskin tepe noktaları oluşturmaktadır.

### 4.3 Kişi Sayısına Göre Sensör Tepkileri
![Kutu Grafikleri](outputs/figures/03_sensor_vs_occupancy_boxplots.png)
Kutu grafiklerinde (Boxplot) görüldüğü üzere, odadaki insan sayısı 0'dan 3'e yükseldikçe $\text{CO}_2$ ve Işık değerleri belirgin bir artış trendi sergilemektedir. Sıcaklık ve ses sensörleri ise tek başlarına daha yüksek varyans göstermektedir.

### 4.4 Korelasyon Isı Haritası
![Korelasyon Matrisi](outputs/figures/04_correlation_heatmap.png)
Korelasyon analizinde en yüksek pozitif ilişkiyi **Işık sensörleri ($r \approx 0.63 - 0.70$)**, **$\text{CO}_2$ ($r \approx 0.63$)** ve **PIR hareket sensörleri ($r \approx 0.55$)** vermektedir.

### 4.5 Zaman Serisi Dinamikleri
![Zaman Serisi Trendi](outputs/figures/05_temporal_trend.png)
Zaman ekseninde yapılan incelemede, odaya insanlar girdiğinde ışıkların açıldığı ve $\text{CO}_2$ seviyesinin hızla tırmandığı; oda boşaldığında ise $\text{CO}_2$'nin kademeli olarak azaldığı görülmektedir.

---

## 5. Öznitelik Mühendisliği (Feature Engineering)

Fiziksel sensör gruplarını anlamlı özetlere dönüştürmek amacıyla iki temel yaklaşım uyguladık:

```python
# Mekânsal Ortalama ve Aralık (Spatial Aggregation)
df["Avg_Temp"] = df[["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]].mean(axis=1)
df["Temp_Range"] = df[["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]].max(axis=1) - df[["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]].min(axis=1)

df["Avg_Light"] = df[["S1_Light", "S2_Light", "S3_Light", "S4_Light"]].mean(axis=1)
df["Light_Range"] = df[["S1_Light", "S2_Light", "S3_Light", "S4_Light"]].max(axis=1) - df[["S1_Light", "S2_Light", "S3_Light", "S4_Light"]].min(axis=1)

df["Avg_Sound"] = df[["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]].mean(axis=1)
df["Sound_Range"] = df[["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]].max(axis=1) - df[["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]].min(axis=1)
```

Bu sayede hem genel ortam seviyesini (`Avg`) hem de mekân içerisindeki homojensizlik ve bölgesel hareketliliği (`Range`) yakalamış olduk.

---

## 6. Deney Tasarımı ve Veri Sızıntısını (Data Leakage) Önleme

### Zaman Serisi Ayrımı (Temporal Split)
Veriler ardışık 31 saniyelik zaman adımlarıyla toplandığından, **rastgele train/test ayrımı yapmak ciddi bir metodolojik hatadır (Temporal Data Leakage)**. Rastgele split yapıldığında, birbirinin neredeyse kopyası olan 30 saniye aralıklı gözlemler hem eğitim hem test setine dağılır ve model ezberleyerek yapay bir başarı sergiler.

Bu nedenle ana deneylerimizde kesin bir **zamansal ayrım (Chronological Split)** uyguladık:
* **Eğitim Seti (Train):** İlk %80 kronolojik veri ($8,103$ gözlem - Geçmiş)
* **Test Seti (Test):** Son %20 kronolojik veri ($2,026$ gözlem - Gelecek)

### Preprocessing Pipeline Mimarisi
Ölçekleme sızıntısını önlemek için Scikit-learn `Pipeline` nesneleri kullanıldı:

```python
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_dt = Pipeline([
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])
```

---

## 7. Makine Öğrenmesi Modelleri

Araştırmamızda üç temel ve açıklanabilir algoritmayı değerlendirdik:
1. **Lojistik Regresyon (Logistic Regression):** Doğrusal karar sınırları için sağlam bir baseline.
2. **K-En Yakın Komşu (KNN):** $k \in [3, 5, 7]$ ile mesafe tabanlı yerel örüntüleri yakalama.
3. **Karar Ağaçları (Decision Tree):** $\text{max\_depth} \in [3, 5, 7, 10]$ ile doğrusal olmayan eşik değerlerini ve sensör etkileşimlerini modelleme.

---

## 8. Model Performans Sonuçları

Tüm modeller Feature Set A (16 sensör) üzerinde eğitilmiş ve kronolojik test setinde değerlendirilmiştir:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree ($\text{depth}=5$)** | **95.06%** | **88.42%** | **73.19%** | **76.85%** | **94.66%** |
| **Decision Tree ($\text{depth}=7$)** | 95.06% | 88.42% | 73.19% | 76.85% | 94.66% |
| **Logistic Regression** | 95.26% | 88.59% | 66.81% | 61.58% | 93.63% |
| **Decision Tree ($\text{depth}=3$)** | 92.05% | 83.24% | 61.04% | 67.85% | 90.94% |
| **KNN ($k=3$)** | 93.48% | 55.90% | 46.34% | 46.85% | 93.08% |
| **KNN ($k=5$)** | 92.65% | 51.49% | 43.61% | 44.18% | 92.44% |
| **KNN ($k=7$)** | 91.41% | 44.77% | 40.43% | 41.83% | 91.52% |

![Karışıklık Matrisleri](outputs/figures/06_confusion_matrices.png)

### Hata Analizi:
* Logistic Regression yüksek accuracy (%95.26) almasına rağmen, azınlık sınıfları yakalamada zorlanmış ve Macro F1'de %61.58'de kalmıştır.
* **Decision Tree ($\text{depth}=5$)**, hem boş odayı (0 kişi) hem de dolu odayı (2 ve 3 kişi) yüksek hassasiyetle ayırt ederek **%76.85 Macro F1** ile en dengeli performansı sergilemiştir.

---

## 9. Feature Set Ablation Deneyi (Sensörlerin Katkısı)

Sensör sayısını azaltmanın ve özet öznitelik kullanmanın etkisini ölçmek için 5 farklı veri seti kombinasyonunu Decision Tree ile test ettik:

![Feature Set Ablation](outputs/figures/08_feature_set_ablation.png)

| Feature Set | Değişken Sayısı | Accuracy | Macro F1 | Weighted F1 | Temel Çıkarım |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Set A (All Sensors)** | 16 | 95.06% | **76.85%** | 94.66% | Baseline performans |
| **Set B (No PIR)** | 14 | 92.05% | **67.85%** | 90.94% | PIR çıkarıldığında Macro F1'de **%9.0 net düşüş** |
| **Set C (Environmental Only)** | 14 | 92.05% | **67.85%** | 90.94% | Yalnızca pasif çevresel ölçümler |
| **Set D (Engineered Summary)** | 10 | 91.61% | **52.34%** | 89.91% | 16 sensör $\rightarrow$ 10 değişkene indiğinde genel doğruluk korunuyor |
| **Set E (Sensors + Time)** | 19 | **95.95%** | **78.98%** | **95.62%** | Zaman değişkenleri en yüksek skoru sağlıyor |

---

## 10. Öznitelik Önem Düzeyleri ve Tartışma

![Öznitelik Önem Düzeyleri](outputs/figures/07_feature_importance.png)

Decision Tree modelinden elde edilen öznitelik önem analizine göre:
1. **Işık Sensörleri ($S_1\text{_Light}$):** Odaya girildiğinde ilk tetiklenen ve en yüksek bilgi kazancını (information gain) sağlayan değişken olmuştur.
2. **$\text{CO}_2$ Değişim Eğimi ($S_5\text{_CO2_Slope}$):** $\text{CO}_2$'nin mutlak seviyesinden ziyade **artış hızı**, odada insan olup olmadığını belirlemede kritik rol oynamıştır.
3. **PIR Hareket Sensörleri ($S_6\text{_PIR}$):** Hızlı hareket anlarını yakalayarak pasif sensörlerin gecikmeli tepkisini telafi etmiştir.

### Veri Sızıntısının Kanıtı: Temporal vs. Random Split
![Zaman Serisi Split Farkı](outputs/figures/09_temporal_vs_random_split.png)

Rastgele split yapıldığında model **%99.56 Accuracy ve %98.37 Macro F1** almaktadır. Ancak bu sonuç gerçekçi değildir; veri sızıntısının bir illüzyonudur. Kronolojik split uygulandığında gerçek dünya genelleme başarısının **%95.06** olduğu doğrulanmıştır.

---

## 11. Limitasyonlar (Limitations)

1. **Tekil Ortam Kısıtı:** Veri seti tek bir kontrollü oda ortamında toplanmıştır. Farklı oda hacimleri, pencere konumları ve yalıtım tiplerinde sensör tepkileri farklılık gösterecektir.
2. **Kişi Sayısı Sınırı:** Veri seti maksimum 3 kişi ile sınırlıdır; kalabalık konferans salonları veya ofisler için model yeniden eğitilmelidir.
3. **Test Periyodu Özelliği:** Zaman serisinin son %20'lik bölümünde 1 kişilik kullanım gözlenmemiştir; bu durum zaman serisi veri setlerinin doğal dinamiklerinden kaynaklanmaktadır.

---

## 12. Sonuç (Conclusion)

Bu çalışma, invaziv kameralar ve mikrofonlar olmadan, yalnızca çevresel sensörler ve basit makine öğrenmesi modelleriyle **%95'in üzerinde oda doluluk tahmini** yapılabileceğini kanıtlamaktadır. 

* Işık ve $\text{CO}_2$ değişim hızı en belirleyici çevresel faktörlerdir.
* PIR hareket sensörleri, pasif sensörlerin gecikmesini önleyerek F1 skorunu yaklaşık %9 artırmaktadır.
* Zaman serisi verilerinde temporal split uygulamak, gerçekçi ve üretime hazır modeller geliştirmek için vazgeçilmezdir.

---

## 13. GitHub Proje Deposu

Projenin tüm kaynak kodlarına, veri yükleme modüllerine, Scikit-learn pipeline'larına, analiz notebook'una ve yüksek çözünürlüklü grafiklerine aşağıdaki bağlantıdan ulaşabilirsiniz:

🔗 **GitHub Deposu:** [https://github.com/gdnz1/room_occupancy_estimation](https://github.com/gdnz1/room_occupancy_estimation)
