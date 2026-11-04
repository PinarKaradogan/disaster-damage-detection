# Kısa ResNet34 head-only GPU benchmark'ı

Bu görevde gerçek GPU ölçümü çalıştırılmadı. Sonradan proje kökünde Windows CMD ile:

```bat
.venv\Scripts\python.exe -m scripts.benchmark_phase4_training --model resnet34_early_fusion --training-mode head_only --loss focal --sampler none --manifest data/processed/patches_subset25/patches.csv --batch-size 16 --workers 0 --persistent false --warmup-batches 1 --batches 2 --val-warmup-batches 1 --val-batches 2
```

Varsayılanlar korunur: Simple CNN, full training mode, weighted CE, sampler none,
batch size 32, workers 0/2/4 ve persistent false/true (worker 0 için true atlanır).
Benchmark artık model factory'yi kullanır. ResNet34 ilk kez `pretrained=True` ile
CPU üzerinde oluşturulur; ağırlık indirme/cache erişimi başarısızsa açık hata ile
durur. Rastgele ağırlıklara fallback yoktur. Sonraki konfigürasyonlar, indirme
yapmadan aynı başlangıç ağırlıkları/buffer snapshot'ından strict load ile kurulur.
Mod, loss, sampler, pretrained tercihi ve toplam/eğitilebilir parametre sayıları
konsola ve isteğe bağlı yeni JSON dosyasına kaydedilir.

Head-only modeli her train geçişinde frozen backbone'u eval modunda tutar;
BatchNorm istatistikleri değişmez. AdamW yalnızca eğitilebilir head parametrelerini
alır. Focal loss gamma=2, alpha yok; `sampler none` shuffled train loader'ı korur.
Bu benchmark'ta weighted sampler desteklenmez. Loss metriğinin paydası gerçek
`run_epoch()` gibi weighted CE'de hedef sınıf ağırlıklarının toplamı, standart CE
ve Focal'da örnek sayısıdır. Warmup/ölçüm batch'lerinde train optimizer adımları
gerçektir; bu sadece bellekte kalan sınırlı performans penceresidir, deney eğitimi
veya checkpoint üretimi değildir.

Mevcut iki bounded epoch düzeni korunur. Yukarıdaki komut her train ve validation
geçişinde 1 warmup + 2 ölçüm batch'i işler; iki train ve iki validation geçişiyle
toplam 12 batch vardır. `--batches 2` tam veri üzerinde iki epoch anlamına gelmez.
Her train geçişi için güncelleme sayıları ve AMP scale değerleri raporlanır.
Görüntüler önbelleğe alınmadan sınırlı iterator sonlandırılır. Train ve val tüm
dataset'i taramaz; test görselleri yüklenmez, hiçbir checkpoint/deney yazılmaz.

İsteğe bağlı `--json-output benchmark_resnet34_head_only_focal.json` yalnızca yeni
bir dosyaya yazar; mevcut dosyanın üzerine yazmaz. Bu seçenek olmadan çıktı konsola
gider. Mevcut sonuç/checkpoint dizinlerini çıktı yolu olarak kullanmayın.

CUDA AMP, GPU allocated/reserved zirveleri, host process-tree RAM örneklemesi,
H2D/compute event süreleri, loader startup/wait ve batch/E2E wall süreleri korunur.
CPU testleri CUDA timing/transfer çağrılarını test double ile değiştirip gerçek
CPU loss/backward akışını engine ile karşılaştırır; GPU ölçümlerini doğruladıkları
iddia edilmez. Gerçek VRAM veya epoch süresi bu görevde ölçülmedi.

İki ölçüm batch'i kısa bir denemedir: disk cache, laptop güç/ısı durumu, AMP scale
gelişimi ve ilk kullanım maliyetlerinden etkilenir. Yazdırılan full-split süreleri
manifestteki örnek sayılarına dayanan ekstrapolasyonlardır; ölçülmüş epoch süreleri
değildir. VRAM sayıları driver/context veya diğer uygulamaları kapsamaz; host RAM
250 ms örnekleme aralığı kısa zirveleri kaçırabilir. Head-only forward hesabını ve
veri I/O'sunu ortadan kaldırmaz. İlk pretrained yükleme snapshot/RAM ve ağ erişimi
gerektirir. %25 manifest ile tam verideki CNN'in veri bütçeleri eşit değildir.
