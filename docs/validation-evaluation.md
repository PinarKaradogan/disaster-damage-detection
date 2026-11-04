# Kesilmiş deneyin validation değerlendirmesi

Bu komut yalnızca değerlendirme yapar. Eğitimi sürdürmez; optimizer oluşturmaz.
Eski, tam resume durumu içermeyen `best.pth` dosyaları kullanılabilir.

Proje kökünden, gerçek değerlendirmeyi başlatmak istediğinizde çalıştırın:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_validation --experiment-dir results/phase4_cnn_weighted_sampler_seed42 --output results/validation_reports/phase4_cnn_weighted_sampler_seed42_best_validation.json
```

Bu komut hazırlık sırasında gerçek veride çalıştırılmamıştır.

Araç `config.json` içindeki model, sınıf sayısı, manifest yolu/SHA256, giriş boyutu,
batch size, workers, normalizasyon ve device/AMP politikasını doğrular/kullanır.
Checkpoint `weights_only=True`, CPU map location ve strict model state yüklemesiyle
açılır; checkpoint metadata'sı config ile aynı olmalıdır. Pretrained indirme yapılmaz.
Desteklenmeyen preprocessing, smoke deneyler ve tamamlanmış deneyler reddedilir.
CUDA ile kaydedilmiş politika CUDA yoksa sessizce CPU'ya geçirilmez.

Yalnızca `split="val"` dataset/loader oluşturulur. Train sampler kurulmaz; sampler
metadata'sındaki `weighted` bilgisi validation örnekleme sırasını değiştirmez.
CE loss ağırlıksızdır; config'teki sampler sınıf ağırlıkları CE'ye uygulanmaz.
Validation sıralı ve augmentationsızdır. Mevcut `run_epoch` modeli `eval()` moduna
alır ve gradient hesaplamasını kapatır.

Rapor kaynak deney dizininin dışında, yeni bir `.json` dosyası olmalıdır; var olan
dosyalar ve `metrics.json` adı reddedilir. Sonuç yazılırken exclusive creation da
üzerine yazmayı engeller. Kaynak `best.pth`, history, config ve status değişmez.
`compare.py` bu raporu tamamlanmış eğitim sonucu olarak tüketmez.

Rapor şunları içerir:

- `report_type="interrupted_experiment_validation"`, `interrupted_experiment=true`,
  orijinal `source_status`, `training_performed=false`, `test_used=false`.
- Checkpoint epoch'u, SHA256 ve history'nin son satırındaki epoch. History bilgisi
  yalnızca kaynak kaydıdır; yeni bir training durumu veya resume garantisi değildir.
- `validation`: loss, Macro F1, accuracy, `class_precision`, `class_recall`,
  `class_f1`, `confusion_matrix`. Sınıf sırası label ID 0..3; matrix satırları gerçek,
  sütunları tahmindir. Eksik sınıfların precision/recall/F1 değerleri sıfırdır.
- Checkpoint'teki F1, yeniden hesaplanan F1'in kayıtlı değerden işaretli/mutlak farkı
  ve `1e-6` toleransla eşleşme sonucu. Fark toleransı aşsa da rapor kaydedilir ve
  fark CLI JSON çıktısında da görülür. Politika aynı tutulsa bile donanım/sürüm ve
  CUDA sayısal farklılıkları sonucu etkileyebilir; rapor orijinal ve güncel sürümleri taşır.

Kaynak deney batch size 32, input size 128, 2 worker ve CUDA AMP kullanır. Config'te
25.615 validation örneği vardır; tam değerlendirme 801 batch işler. Bir batch'in iki
float32 RGB giriş tensor'u toplam yaklaşık 12 MiB'dir; bu toplam RAM/VRAM tahmini
değildir. Aktivasyonlar, CUDA çalışma alanları ve image decoding ek bellek kullanır.
İki worker varsayılan prefetch ile birden fazla batch'i RAM'de tutar; Windows worker
süreçlerinin Python/PyTorch bellek yükü de vardır. Manifest pandas ile okunur;
dataset bütün görselleri belleğe almaz. Etiket/tahmin listeleri validation örneği
sayısıyla büyür. Eski checkpoint'in optimizer tensor'ları yükleme sırasında CPU
RAM kullanır, model yüklenince serbest bırakılır; GPU'ya optimizer taşınmaz.
Eğitime göre backward/optimizer GPU belleği gerekmese de burada gerçek validation
VRAM, RAM veya süre ölçümü yapılmamıştır. Yoğun disk erişimi ve diğer GPU süreçleri
kaynak baskısı yaratabilir.

Raporun son yazımı sırasında süreç kesilirse yalnızca yeni rapor dosyası eksik
kalabilir. Kaynak deney korunur; tekrar denemede farklı bir çıktı adı seçin.
