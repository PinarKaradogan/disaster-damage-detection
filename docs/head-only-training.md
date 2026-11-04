# ResNet34 head-only transfer learning

`--training-mode full` varsayılandır; bütün parametreler eğitilebilir ve BatchNorm
mevcut eğitim davranışını sürdürür. Simple CNN yalnızca `full` destekler.

`--training-mode head_only` ResNet34 backbone parametrelerinin `requires_grad`
değerini kapatır; yalnızca `get_classifier()` (`fc`, 512 → 4) eğitilir. timm'in
pretrained RGB ağırlıklarını altı kanala tekrar edip 0.5 ile ölçekleme davranışı
değişmez. Freezing ağırlıkları yeniden başlatmaz. Her `train()` çağrısında backbone
eval modunda, classifier istenen train/eval modunda kalır. Böylece dondurulmuş BN
running mean/variance ve num_batches_tracked değerleri sabit kalır.

Gerçek altı kanallı, dört sınıflı ResNet34 parametre sayıları:

| Mod | Toplam | Eğitilebilir |
| --- | ---: | ---: |
| full | 21.296.132 | 21.296.132 |
| head_only | 21.296.132 | 2.052 |

AdamW yalnızca `requires_grad=True` parametreleri alır. Toplam sayı mevcut
`parameter_count`, eğitilebilir sayı `trainable_parameter_count`, mod
`training_mode` alanlarında config/checkpoint/metrics metadata'sına kaydedilir;
oturum başında da yazdırılır. State dict anahtarları ve checkpoint şeması değişmez.

Windows CMD, proje kökünde (eğitimi kullanıcı başlatır):

```bat
.venv\Scripts\python.exe -m src.train ^
  --model resnet34_early_fusion ^
  --training-mode head_only ^
  --manifest data/processed/patches_subset25/patches.csv ^
  --loss weighted_ce --sampler none ^
  --experiment-name phase4_resnet34_head_only_subset25_seed42 ^
  --epochs 30 --session-epochs 2 --batch-size 32 --num-workers 0 --seed 42
```

İkinci oturum (aynı toplam hedef ve aynı ayarlar):

```bat
.venv\Scripts\python.exe -m src.train ^
  --model resnet34_early_fusion ^
  --training-mode head_only ^
  --manifest data/processed/patches_subset25/patches.csv ^
  --loss weighted_ce --sampler none ^
  --experiment-name phase4_resnet34_head_only_subset25_seed42 ^
  --epochs 30 --session-epochs 2 --batch-size 32 --num-workers 0 --seed 42 --resume
```

Bu örnek iki yeni epoch eğitir; 30 toplam scheduler hedefidir. `weighted_ce` yalnızca
bir kullanım örneğidir, bu görevde strateji seçimi/eğitim yapılmamıştır. Eğitim
CLI'si yeni ResNet deneyinde pretrained ağırlıkları kullanır; ağ/cache erişimi
gerekebilir. İndirme hatasında rastgele ağırlıklara sessiz dönüş yapılmaz. Testler
`pretrained=False` veya küçük sentetik backbone kullanır; indirme yapmaz.

Resume modu, manifest hash'i, eğitim politikası, sayılar ve mevcut diğer kritik
metadata eşitliğiyle doğrulanır. Freezing modeli oluştururken uygulanır; optimizer
yalnızca aynı head parametreleriyle kurulduktan sonra durum yüklenir. Resume sırasında
pretrained dosyalar yeniden indirilmez. Head-only → full veya full → head-only
aynı deneyin resume işlemi olarak kabul edilmez; ayrı deney gerektirir. Kademeli
fine-tuning uygulanmaz.

Eski metadata'da `training_mode` yoksa yalnızca `full` kabul edilir; eğitilebilir
parametre sayısı yoksa full modunda eski toplam sayı kullanılır. Bu uyumluluk
yalnızca zaten tam resume durumu bulunan checkpoint'ler içindir. Eski eksik
`best.pth` dosyalarına tam resume desteği kazandırmaz. Eski config dosyaları
resume sırasında yeniden yazılmaz; tamamlanmış deneyler reddedilir. Validation
aracı da eksik modu full kabul eder. Model state dict yapısı değişmediğinden eski
checkpoint'ler değerlendirme için yüklenebilir.

Gerçek ResNet34 CPU testleri yalnızca head ağırlıklarının güncellenmesini, bütün
backbone parametre/buffer'larının sabitliğini ve full modda eski davranışı kontrol
eder. Tekrarlı resume testlerinde disk/RAM kullanımını küçük tutmak için timm
backbone'u aynı classifier/BN sözleşmesine sahip küçük bir CPU modeliyle değiştirilir;
gerçek wrapper, optimizer/scheduler, loader, RNG ve checkpoint akışı kullanılır.

Freezing backbone forward hesabını veya görsel yüklemeyi ortadan kaldırmaz. Head
ile sınırlı optimizasyon hedef veriye uyumu kısıtlayabilir; BN istatistikleri bilinçli
olarak pretrained değerlerinde tutulur. Tam model ağırlıkları ve best snapshot hâlâ
checkpoint'e yazılır; head-only checkpoint dosyasını yalnızca classifier içerir hale
getirmez. CPU snapshot RAM/disk maliyeti devam eder. GPU bellek/hız ölçümü yapılmadı.
RNG eşdeğerliği için örnekler `num_workers=0` kullanır; çok işçili/CUDA deterministik
eşdeğerlik garantisi mevcut sınırlamalara tabidir. %25 ResNet train bütçesi ile tam
verideki Simple CNN bütçesi eşit değildir; mimari farkı tek başına ölçülmüş sayılmaz.
