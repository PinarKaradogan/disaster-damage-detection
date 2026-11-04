18 Ekim 2026 — Gerçek RTX 4050 Simple CNN eğitim throughput benchmark raporu

Bu makinenin ölçüm sırasındaki durumu için öneri: **num_workers=2, persistent_workers=False**.
Bu kombinasyon iki döngüde 547,25 / 552,19 çift/sn verdi ve minimum kullanılabilir sistem
RAM'i 1,83 GiB oldu. `2/True` açılış maliyetini düşürdü fakat RAM payını 1,15 GiB'a indirdi.
`4/True` ikinci döngüde daha hızlıydı; minimum kullanılabilir RAM'in **26,14 MiB** olması
nedeniyle bu sistem durumunda uzun eğitim için önerilmiyor. `4/False` da belirgin değişkenlik
ve 0,81 GiB minimum RAM payı gösterdi. Hiçbir kalıcı training ayarı değiştirilmedi.

Ölçüm kaynağı: [ham JSON](phase4_training_benchmark_20261018.json), 21.272 byte.
Araç: [benchmark_phase4_training.py](../scripts/benchmark_phase4_training.py).
Sonuçlar model kalitesi veya model seçimi için kullanılmadı.

Python 3.13.7, PyTorch 2.14.1+cu132, NVIDIA GeForce RTX 4050 Laptop GPU,
6140 MiB VRAM, yaklaşık 15,71 GiB sistem RAM'i, 10 ana süreç CPU thread'i.
cuDNN benchmark kapalı; mevcut TF32 ayarı korundu. Gerçek xBD train: 134028 çift;
validation: 25615 çift. Test patch'leri açılmadı.

Batch 32, input 128x128, seed 42, gerçek SimpleCNN, AdamW lr=0.0003,
weight_decay=0.0001, tam train etiketlerinden Weighted CE, CUDA AMP float16 + GradScaler.
Sınıf ağırlıkları: [0.1473379433, 1.2164909840, 1.2160875797, 1.4200832844].
Phase 4 `build_loaders` dataset/augmentation/collation/sampler davranışı kullanıldı;
pin_memory=False ve multiprocessing prefetch_factor=2 korundu. Benchmark'a özel bounded
batch sampler iki kısa iterator'ı gerçekten tüketir; tam dataset üzerinden eğitim yapmaz.
Train örnekleri shuffle edilir; validation sıralı ön ek kullanır.

Her kombinasyonda aynı başlangıç ağırlıkları ve temiz optimizer/scaler oluşturuldu.
Başlangıç state SHA256: `08ecd5e9c0a352f4f110b3ead4cfd0aeb839a9fa79f5f3d7af63698b8307106d`.
İki kısa epoch boyunca model/optimizer durumu devam eder; ikinci train shuffle seed'i 43'tür.
Her split/epoch'un warm-up ve ölçüm örnek sırası hash ile kombinasyonlar arasında doğrulandı.
Worker RNG akışları nedeniyle augmentation parametreleri birebir aynı olmak zorunda değildir;
augmentation dağılımı ve ayarları gerçek training ile aynıdır.

Her döngü: train 10 warm-up + 100 ölçüm batch'i (3200 ölçülen çift),
validation 5 warm-up + 20 inference batch'i (640 ölçülen çift).
Forward, Weighted CE, backward ve scaler.step/update gerçekten çalıştırıldı.
**Bütün 10 train ölçüm penceresinde 100 scaler.step çağrısına karşılık 99 gerçek AdamW
güncellemesi oldu.** GradScaler bir gradyan overflow güncellemesini atladı; ilk pencerede
scale 65536 -> 32768, ikinci pencerede 32768 -> 16384. Bu, gerçek engine'in AMP davranışıdır;
throughput başarılı güncelleme sayısını değil işlenen görüntü çiftlerini sayar.

| Workers | Persistent | Train çift/sn 1 / 2 | Val çift/sn 2 | Train dk 2* | Val dk 2* | Toplam dk 1 / 2* | Başlangıç sn 1 / 2** |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | False | 206,75 / 111,92 | 250,78 | 19,96 | 1,70 | 12,56 / 21,67 | 0,25 / 0,21 |
| 2 | False | 547,25 / 552,19 | 767,59 | 4,24 | 0,75 | 5,06 / 4,99 | 21,49 / 21,83 |
| 2 | True | 531,47 / 535,13 | 664,00 | 4,18 | 0,64 | 5,14 / 4,82 | 21,66 / 0,23 |
| 4 | False | 1007,11 / 377,18 | 444,70 | 6,48 | 1,68 | 3,31 / 8,16 | 47,49 / 73,88 |
| 4 | True | 438,26 / 727,49 | 1223,51 | 3,08 | 0,35 | 7,34 / 3,43 | 76,83 / 0,40 |

* Süreler tam epoch ölçümü değildir; kısa pencere üzerinden train+validation tahminidir.
İki döngü farklı örnekler/model durumu ve değişen makine koşulları içerir; aralarındaki fark
saf persistent etkisi olarak yorumlanamaz. Örneğin 4/False throughput farkının nedeni bu
deneyle ayrıştırılmadı; tek hızlı sonucu genel hızlanma iddiası olarak kullanmak uygun değildir.

** Başlangıç, her split için iter(loader) oluşturulmasından ilk CPU batch teslimine kadar
geçen süredir; tabloda train+validation toplamı verilir. Saf süreç açılışını ilk batch işinden
tam ayırmaz. JSON, iterator ve ilk next bekleme sürelerini ayrıca içerir.
Persistent açıkken ikinci döngüde train ve validation worker PID'lerinin korunduğu doğrulandı.
Kapalı durumda iterator tükenirken süreç kapatma maliyeti de ayrıca ölçülüp tahmine eklendi.

Train ikinci döngü başına ortalama süreler:

| Workers | Persistent | Veri bekleme ms | Model duvar saati ms | H2D CUDA event ms | Hesaplama CUDA event ms |
|---|---|---:|---:|---:|---:|
| 0 | False | 222,097 | 62,924 | 3,104 | 59,265 |
| 2 | False | 32,788 | 23,010 | 5,407 | 17,153 |
| 2 | True | 31,235 | 26,390 | 5,246 | 20,689 |
| 4 | False | 34,995 | 44,126 | 15,538 | 27,340 |
| 4 | True | 16,100 | 25,202 | 6,519 | 18,105 |

Uçtan uca süre ölçüm penceresinin başında ve sonunda CUDA synchronize uygulanır.
CUDA event'leri H2D ve forward/loss/backward/optimizer aralıklarını ayırır; aralıklar kernel
gönderim boşluklarını da içerebilir. Ek faz senkronizasyonu eklenmez. Model duvar saati H2D,
host overhead ve engine ile aynı finite-loss kontrolü/CPU metrik aktarımını içerir.
E2E ayrıca next(iterator), kimlik hash kontrolü ve benchmark muhasebesini içerir.
CUDA event süreleri model duvar saatiyle örtüşür; birbirlerine veya E2E'ye eklenmezler.

| Workers | Persistent | GPU peak allocated / reserved MiB | Host private commit peak GiB | Minimum sistem kullanılabilir GiB |
|---|---|---:|---:|---:|
| 0 | False | 229,5 / 336,0 | 3,02 | 2,73 |
| 2 | False | 229,5 / 352,0 | 6,20 | 1,83 |
| 2 | True | 229,5 / 352,0 | 9,21 | 1,15 |
| 4 | False | 229,5 / 352,0 | 9,31 | 0,81 |
| 4 | True | 229,5 / 352,0 | 15,23 | 0,03 |

VRAM açısından bu sınırlı deneyde 4 worker/persistent için OOM görülmedi ve PyTorch belleği
6 GB sınırının oldukça altında kaldı. Ancak 4/True'da 4 train + 4 validation worker'ı aynı anda
bellekte kaldı; host RAM payı kritik düzeye indi. Kısa testin tamamlanması uzun eğitimin güvenli
olduğunu kanıtlamaz. Windows private commit fiziksel RAM tüketimiyle aynı değildir; paylaşılan
sayfalar RSS toplamında tekrar sayılabilir. Process-tree peak RSS de JSON'dadır. Host bellek
250 ms aralıkla örneklendiğinden daha kısa sıçramalar kaçabilir. GPU peak'leri ölçüm penceresine
aittir, driver/context ve başka uygulamaların VRAM'ini kapsamaz. Isı/güç sınırı veya paging
bu deneyde doğrudan ölçülmedi; throughput değişiminin kesin nedeni olarak ileri sürülmüyor.

Her split için tahmin formülü:

`ceil(split_pairs / 32) * (ölçülen_E2E / batch_sayısı) + max(0, başlangıç - ortalama_veri_bekleme) + iterator_kapanışı`

Train 4189, validation 801 batch kabul edilir. Başlangıç ilk veri beklemesini içerdiği için
ortalama veri bekleme bir kez çıkarılır. Tahmin; tam epoch sonunda sklearn metrikleri,
TensorBoard/CSV yazımı, checkpoint I/O, scheduler ve early stopping maliyetini içermez.
Disk cache temizlenmedi; sıralama 0/False, 2/False, 2/True, 4/False, 4/True idi.
Kısa pencere, sıralı validation ön eki, son kısmi batch, cache, sistem yükü, güç/ısı koşulları
ve AMP değişimleri belirsizlik yaratır. Tablodaki iki döngü güven aralığı değildir.

Mevcut RAM durumunda önerilen kombinasyonu tekrar ölçmek için proje kökünde:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_phase4_training --workers 2 --persistent false
```

Tam matris CLI'si aşağıdadır. Bu makinede 4/True için ölçülen 26 MiB RAM payı nedeniyle
tam matrisi tekrar çalıştırmadan önce sistem RAM'inde daha fazla yer açılması gerekir:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_phase4_training --batch-size 32 --seed 42 --warmup-batches 10 --batches 100 --val-warmup-batches 5 --val-batches 20 --workers 0 2 4 --persistent false true
```

İsteğe bağlı `--json-output YENI_DOSYA.json` küçük ölçüm çıktısı yazar; mevcut dosya veya
korunan Weighted CE klasörüne yazmayı reddeder. Checkpoint, tam training ve test inference
çalıştırılmaz. CPU doğrulaması ayrı test dosyalarında bulunur; model/engine dosyaları değişmedi.

Son CPU doğrulaması: **51 PASS (15,19 sn)**. Yeni training benchmark testleri, önceki veri
benchmark testleri, training data/foundation ve dataset testleri birlikte çalıştırıldı.
Windows worker oluşturma, bounded iterator tüketimi, persistent PID koruması, örnek sırası,
tahmin hesabı, başlangıç state restorasyonu, CUDA yokken erken hata ve çıktı korumaları doğrulandı.
Pytest geçici klasör erişimi için izinli çalıştırma kullanıldı. `git diff --check` temiz;
mevcut tracked dosyalarda değişiklik, stage/commit/push yok.
