# ResNet34 için yüzde 25 train manifesti

Proje kökünden:

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.subset_manifest
```

Kaynak `data/processed/patches_full/patches.csv`; çıktı
`data/processed/patches_subset25/patches.csv`. Seed (`random_state`) 42, train hedefi
33.507 satırdır. Train satırları 134.028 olduğunda bu tam %25'tir. Kaynak validation
satırlarının tamamı, aynı sırada ve hücre değerleri değiştirilmeden eklenir. Test
satırları eklenmez; test image_id bilgileri yalnızca split çakışması kontrolündedir.
Görseller okunmaz, değiştirilmez veya kopyalanmaz. Patch yolları orijinal
`patches_full/` dosyalarını göstermeyi sürdürür.

Örnekleme birimi bina patch'idir. Önce train satırları `(disaster, damage)` ile
katmanlanır. Her mevcut katmana en az bir örnek ayrılır; oransal ideal kontenjanların
tabanı alınır ve en büyük eksik/fazla kontenjanlar düzeltilerek hedef tam tutturulur.
Eşitlikler sıralı katman anahtarıyla çözülür. Hedef katman sayısından azsa veya train
popülasyonundan büyükse açık hata verilir; katman sessizce düşürülmez.

Katmanlar ve bina anahtarları sıralanır, `numpy.random.RandomState(42)` ile her
katmandan replacement olmadan seçilir. Kaynak satır sırası çıktı içinde korunur.
Bu yöntem `sklearn.train_test_split` ile aynı seçimleri üretmeyi hedeflemez; tek
örnekli katmanları da koruyan açık, sürümlenmiş bir kontenjan yöntemi kullanır.
Kaynak hash'i, NumPy/Pandas sürümleri ve bütün kontenjanlar raporda kaydedilir.

Çıktı dizinindeki dosyalar:

- `patches.csv`: Seçilmiş train + eksiksiz validation, orijinal sütunlar.
- `summary.json`: Kaynak/yeni SHA256, seed, split/sınıf/afet/katman dağılımları,
  sınıf oranındaki yüzde puan değişimleri, görüntü sayıları, overlap kontrolü,
  loader uyumluluğu ve karşılaştırma uyarısı.
- `image_distribution.csv`: Her orijinal train image_id için afet, kaynak patch
  sayısı, seçilen patch sayısı ve seçilme oranı; hiç seçilmeyen görüntüler de bulunur.

Aynı ayarlar ve kaynakla tekrar çalıştırma aynı dosyaları üretir; mevcut içerik
aynıysa dosyalar yeniden yazılmaz. Farklı mevcut içerik varsa üzerine yazılmaz.
Çok dosyalı yayın işlemi tek bir atomik işlem değildir; kesintide kısmi çıktılar
kalabilir. Farklı/eksik çıktı durumunda dosyaları inceleyin; araç otomatik silmez.

`DamagePatchDataset` ve `build_loaders` gerçek çıktı üzerinde oluşturularak
train/val uzunlukları ve dört sınıfın ağırlık hesaplaması doğrulanır. Bu kontrol
model oluşturmaz veya eğitim başlatmaz. Küçük sentetik testlerde iki loader'dan
batch de okunur. Kaynak ve yeni manifestte train/validation image_id çakışması
sıfır olmalıdır.

Bir train görüntüsünün binalarının bir kısmı seçilebilir; bu, train/validation
split'ini yeniden yapmak değildir. Aynı görüntüden gelen binalar bağımsız görüntü
örnekleri sayılmamalıdır. Kaynak train görüntülerinin tamamının seçileceği veya
her görüntüden %25 alınacağı garanti edilmez; gerçek dağılım raporlanır.

Bu alt kümeyle eğitilen ResNet34 ve tam train verisiyle eğitilen Simple CNN'in veri
bütçeleri eşit değildir. Aralarındaki fark yalnızca mimariye atfedilemez; eşit veri
bütçeli kontrollü mimari karşılaştırması olarak sunulmamalıdır. Validation seti
aynı kalır; test split'i örnek seçimi veya hiperparametre kararlarında kullanılmaz.

Veri klasörü `.gitignore` kapsamındadır; üretilen manifest ve raporlar yerel
artefaktlardır. Kaynak sonuçlar, görüntüler ve Faz 3 çıktıları yeniden üretilmez.
