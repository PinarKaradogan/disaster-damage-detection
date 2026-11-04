# Epoch sınırında eğitime devam etme

Yeni bir deneyi toplam 30 epoch hedefiyle başlatıp ilk oturumu 2 epoch ile sınırlayın:

```powershell
python -m src.train --model simple_cnn --loss weighted_ce --experiment-name cnn_resume_new --epochs 30 --session-epochs 2 --num-workers 0
```

Aynı ayarlarla sonraki 2 epoch'u eğitin (epoch 3 ve 4):

```powershell
python -m src.train --model simple_cnn --loss weighted_ce --experiment-name cnn_resume_new --epochs 30 --session-epochs 2 --num-workers 0 --resume
```

`--session-epochs` verilmezse toplam hedefe veya early stopping sınırına kadar devam edilir.
`--epochs` her oturumda aynı kalmalıdır; cosine scheduler'ın `T_max` değeri toplam
hedefe bağlıdır. Smoke modunun mevcut en fazla 2 epoch toplam sınırı korunur.

`best.pth` değerlendirme için en yüksek validation Macro F1 modelini taşır.
`last.pth` scheduler adımından sonraki son tamamlanmış epoch'u, model/AdamW/scheduler/
AMP GradScaler durumlarını, early stopping best ve bad_epochs değerlerini, RNG
durumlarını, konfigürasyonu, history ve en iyi model kaydını taşır. Test split'i
eğitim veya model seçiminde kullanılmaz.

`running` devam eden oturumu, `paused` planlı oturum sınırını, `completed` toplam
hedef veya early stopping bitişini, `failed` oturum hatasını belirtir. Resume yalnızca
açık `--resume` ile ve tam durum içeren yeni `last.pth` üzerinden yapılır. `paused`,
`failed` veya kesinti nedeniyle `running` kalmış deneyler devam edebilir. Tamamlanan
deneyler, eski/eksik checkpoint'ler ve farklı ayarlar reddedilir. Normal başlatma,
dolu sonuç veya TensorBoard klasörlerinin üzerine yazmaz.

Model, manifest SHA256, seed, loss/sampler, batch size, toplam hedef, optimizer,
scheduler, patience, augmentation, normalization, workers, AMP/device ve sürümler
dahil metadata eşitliği doğrulanır. ResNet resume sırasında pretrained ağırlıklar
tekrar indirilmez; model durumu checkpoint'ten yüklenir. Bu nedenle ilk oturumun
pretrained tercihinin aynı verilmesi gerekir. Mevcut deneyleri yeni formata
dönüştürme veya toplam hedefi sonradan değiştirme desteklenmez.

Her dosya önce aynı dizinde geçici dosyaya yazılır, flush/fsync ardından `os.replace`
ile değiştirilir. Önce `last.pth` epoch'u commit eder, sonra türetilen `best.pth` ve
`history.csv` yayınlanır. Aralarında kesinti olursa doğrulanmış `last.pth` yetkili
kayıttır: resume uyarı vererek history ve best dosyalarını bu kayıttan tekrar yazar.
History'nin commit edilmiş bütün satırları korunur; commit edilmemiş ileri satırlar
atılır, eksik satırlar onarılır. Checkpoint içindeki history 1..epoch aralığında
kesintisiz olmalıdır. TensorBoard `purge_step=son_epoch+1` ile yeniden açılır;
kesintide loglanmış ama commit edilmemiş epoch adımları görünümden temizlenir.
Son epoch commit edilmiş fakat status yazımı kesilmişse `finished` bilgisi yeni
eğitim başlatılmasını engeller; böyle bir durumda durum dosyasını otomatik onarmaz.

Python random (paired augmentation), NumPy, PyTorch CPU/CUDA ve train/val loader ile
sampler generator durumları saklanır. `num_workers=0` için aynı CPU yazılım/donanım
ortamında kesintisiz ve bölünmüş eğitim eşdeğerliği sentetik testlerle doğrulanır.
Çok işçili yüklemede worker içi RNG/prefetch durumları saklanmaz; birebir deterministik
eşdeğerlik garanti edilmez. CUDA çekirdeklerinin deterministik çalışması da garanti
edilmez. CUDA RNG geri yüklemede cihaz sayısı aynı olmalıdır. Epoch ortasındaki
kesintide o epoch yeniden eğitilir.

Checkpoint'ler yalnızca `weights_only=True` ile yüklenir; NumPy RNG dizisi güvenli
yerleşik tiplere dönüştürülür. Güvensiz pickle yüklemeye geri dönüş yapılmaz. Dosya
bozulması veya eksik `last.pth` durumunda best dosyasından optimizer/RNG tahmin
edilmez. Aynı deney için eşzamanlı iki eğitim süreci çalıştırmak desteklenmez.
Atomic replace tam disk/işletim sistemi arızasında mutlak kalıcılık garantisi
vermez. Son durumun yanında en iyi durum da saklandığından `last.pth` daha büyüktür.
