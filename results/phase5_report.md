# Faz 5 — Gelişmiş Modeller ve Bina Lokalizasyonu: Sonuç Raporu

Tarih: 5 Kasım 2026 · Branch: `feature/phase5-advanced-models` · Kapsam: yalnızca train/validation.
**Test verisi hiçbir aşamada açılmadı.** Model seçimi, threshold ve post-processing seçimi validation ile yapıldı;
bu yüzden aşağıdaki validation skorları iyimser olabilir. Tarafsız tahmin Faz 6'da test setinde tek seferlik alınacak.

## 1. Özet

| Alt adım | Çıktı | Ana sonuç (validation) |
|---|---|---|
| 5.1 Siamese ResNet34 | `phase5_siamese_resnet34_focal_seed42_w4` | Macro-F1 0,7773 |
| 5.2 Siamese ViT-Small (DeiT-S) | `phase5_siamese_vit_small_focal_seed42_w4` | Macro-F1 0,7701 |
| 5.3 U-Net (ResNet34 encoder) | `phase5_unet_resnet34_seed42_w4` | Piksel Dice 0,861 / IoU 0,756 |
| 5.4 U-Net + sınıflandırıcı | `src/evaluation/two_stage.py` | Tespit edilen binalarda Macro-F1 0,780; katı uçtan uca 0,700 |

Bütün eğitimler tam train (134.028 çift / 2.380 görüntü) ve tam validation (25.615 çift / 419 görüntü) ile yapıldı.
Bütün koşular denetimden geçti (`results/phase5_audit/audit_*.json`): dosyalar, manifest ve split hash'leri,
history/early-stopping tutarlılığı, `best.pth`/`last.pth` bütünlüğü ve `best.pth`'nin validation'da yeniden
değerlendirilmesi (kayıtlı skorla fark 0,0).

## 2. Hasar sınıflandırma karşılaştırması (Faz 4 + Faz 5)

Kaynak: `results/phase5_classification_validation.{csv,json}` (`python -m src.training.compare --kind models ...`).

| Model | Macro-F1 | No | Minor | Major | Destroyed | Parametre | Best / toplam epoch | Süre |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Simple CNN (early fusion), Faz 4 | 0,7411 | 0,924 | 0,559 | 0,632 | 0,849 | 0,09 M | 14 / 19 | 3 sa 4 dk* |
| **ResNet34 early fusion, Faz 4** | **0,7789** | 0,942 | **0,629** | 0,670 | **0,874** | 21,30 M | 4 / 9 | 50 dk |
| Siamese ResNet34, Faz 5 | 0,7773 | **0,944** | 0,620 | **0,673** | 0,872 | 21,68 M | 4 / 9 | 62 dk |
| Siamese ViT-Small (DeiT-S), Faz 5 | 0,7701 | 0,938 | 0,616 | 0,666 | 0,860 | 21,91 M | 9 / 14 | 100 dk |

\* Farklı bilgisayar ve `num_workers=2` ile ölçüldü; süreler hız karşılaştırması olarak kullanılmamalıdır.

**Ortak politika:** Focal Loss (γ=2), AdamW 3e-4 / wd 1e-4, Cosine (T_max 30, eta_min 1e-6), en çok 30 epoch,
patience 5, batch 32, AMP, 128×128 giriş, ImageNet normalizasyonu, aynı train augmentasyonu, seed 42. Tek kabul
edilen fark `num_workers` (2 / 4); `compare.py` bunu kayda geçiriyor, diğer bütün politika alanları birebir aynı.

**Değerlendirme:**
- Üç ön eğitimli model 0,770–0,779 bandındadır. Ardışık epoch'lar arasındaki validation F1 dalgalanması 0,02–0,03
  olduğundan aradaki farklar (≤ 0,009) **anlamlı kabul edilmemelidir**; tek seed ile ölçülmüştür. Kurala göre seçilen
  sınıflandırıcı ResNet34 early fusion'dur ve 5.4 entegrasyonunda bu kullanıldı.
- Siamese yapı, early fusion'a göre ek bir kazanç sağlamadı; ViT, ortak politika altında daha yavaş öğrendi
  (best epoch 9) ve daha güçlü overfitting gösterdi (son epoch train F1 0,853 / val F1 0,722).
- Bütün ön eğitimli modeller en iyi değere erken ulaştı ve sonrasında val loss yükseldi; 3e-4 öğrenme oranı ön
  eğitimli ağlar için yüksek olabilir (test edilmemiş hipotez).
- Tüm modellerde en zor ayrım **minor-damage ↔ no-damage**'dir (ResNet34'te minor binaların %29'u hasarsız tahmin ediliyor).

**Model tasarımları:**
- *Siamese ResNet34:* tek `timm` ResNet34 encoder (`resnet34.a1_in1k`) PRE ve POST'a uygulanır (ağırlık paylaşımı
  yapısal; testle doğrulandı), özellikler `[f_pre, f_post, |f_pre − f_post|]` (1536) → MLP (256, ReLU, dropout 0,2) → 4 logit.
- *Siamese ViT-Small:* aynı yapı, encoder ViT-Small/16 mimarisi + **DeiT-Small** ImageNet-1k ağırlıkları
  (`deit_small_patch16_224.fb_in1k`). Seçim gerekçesi: veri hattımızla aynı ImageNet normalizasyonu (ViT augreg
  ağırlıkları 0,5/0,5 kullanır), ResNet34 ile aynı ön eğitim verisi, 128 girişte Swin-Tiny'den ~2 kat hızlı.
  128×128 girişte konum gömmesi 14×14'ten 8×8 token'a yeniden örneklendi (64 token; düşük mekânsal çözünürlük).

## 3. U-Net bina lokalizasyonu

**Veri:** Ham xBD PRE görüntüleri (1024×1024) ve resmî `*_pre_disaster_target.png` maskeleri (0/1). Sınıflandırıcılarla
**aynı görüntü düzeyindeki split** (`splits/standard`) kullanıldı; train/val/test ayrık, sızıntı yok. 2.799 train+val
maskesinin hepsi doğrulandı (1024×1024, ikili); görüntülerin %18'inde bina yok, piksellerin ~%6'sı bina.
Maske–poligon hizası görsel ve sayısal olarak doğrulandı (`phase5_audit/unet_mask_alignment.png`, IoU 0,96).

**Model ve eğitim:** `smp.Unet('resnet34')`, ImageNet encoder (`smp-hub/resnet34.imagenet` = torchvision ağırlıkları),
24,4 M parametre. Train: görüntü başına 4 rastgele 512×512 kesit + görüntü/maskeye aynı rot90 ve yatay flip.
Val: sabit 2×2 kesit ızgarası. Kayıp: BCE + soft Dice. AdamW 3e-4, Cosine, 30 epoch, patience 5, batch 8, AMP.
Seçim metriği: validation **global piksel Dice**.

| Ölçüm | Val kesitleri (seçim) | Val tam 1024 görüntü |
|---|---:|---:|
| Dice (piksel F1) | **0,861** | 0,864 |
| IoU | 0,756 | 0,760 |
| Precision / Recall | 0,875 / 0,847 | 0,876 / 0,852 |
| Görüntü bazlı Dice (binalı 343 görüntü) | 0,798 | 0,802 |
| Boş görüntüde yanlış alarm oranı (76 görüntü) | %10,5 | %7,9 |

Boş maske kuralı: global Dice/IoU tüm pikseller üzerinden; binasız görüntüler yalnızca FP olarak katkı yapar.
Görüntü bazlı Dice yalnızca binalı görüntülerde ortalanır; binasız görüntüler ayrı yanlış alarm oranıyla raporlanır.
U-Net metrikleri dört sınıflı Macro-F1 ile **aynı anlamda karşılaştırılamaz**.

En iyi değer son epoch'ta (30) geldi: model 30 epoch sınırında hâlâ yavaşça iyileşiyordu (son 10 epoch 0,846 → 0,861).
Eğitim süresi 3,4 saat.

**Afet bazında piksel Dice (tam görüntü):** hurricane-matthew 0,730 · midwest-flooding 0,818 · guatemala-volcano 0,842
(3 görüntü) · socal-fire 0,846 · hurricane-michael 0,849 · mexico-earthquake 0,857 · santa-rosa-wildfire 0,877 ·
hurricane-harvey 0,889 · hurricane-florence 0,891 · palu-tsunami 0,892.

**Hatalı örnekler** (`phase5_audit/unet_val_predictions.png`): büyük ve belirgin yapılar Dice 0,97–0,98; başarısızlıklar
(1) ağaç örtüsü altındaki küçük evler (socal-fire), (2) görüntü kenarındaki tek bina (mexico-earthquake_96, Dice 0,03),
(3) binasız görüntülerde çadır/branda/enkaz yığınlarının bina sanılması.

## 4. İki aşamalı sistem (U-Net + hasar sınıflandırıcı)

**Akış:** PRE (1024) → U-Net olasılık haritası → threshold → bağlı bileşenler (4-komşuluk, < 10 piksel atılır, isteğe
bağlı erosion çekirdekli bina ayırma) → her bileşenin kutusu → **Faz 3 kırpma kuralı** (`patch_geometry`: kare kenar
max(64, 2×kutu), siyah dolgu) PRE ve POST'a aynı koordinatla → 128×128 → ResNet34 early fusion → 4 hasar sınıfı →
binalar POST görüntüsünde hasara göre renklendirilir. Tam görüntü çıkarımı yapıldığından kesit → orijinal koordinat
dönüşümü gerekmez.

**Değerlendirme ayrımı:**
- *Oracle:* gerçek bina kutularıyla aynı kırpma yolu. **Faz 4 sonucunu birebir üretir** (Macro-F1 0,7789, accuracy
  0,8805) → entegrasyon kodunun sınıflandırıcıya eğitimdekiyle aynı girdiyi verdiği doğrulandı.
- *Uçtan uca:* U-Net bileşenleri gerçek binalarla eşleştirilir. Lokalizasyon için birebir eşleşme (IoU ≥ 0,5);
  hasar için bir binanın ≥ %50'sini kapsayan bileşen o binaya atanır (birleşik bileşenler birden çok binaya hizmet edebilir).

| Ölçüm | t=0,5, ayırma yok | **t=0,9, erosion 4 (seçilen)** |
|---|---:|---:|
| Lokalizasyon F1 (IoU ≥ 0,5) | 0,513 | **0,550** |
| Precision / Recall | 0,678 / 0,413 | 0,652 / 0,476 |
| Birden çok binayı içeren bileşen (içindeki bina) | 2.426 (12.217) | 1.962 (8.760) |
| Uygun binaların kapsanma oranı | **%87,2** | %80,9 |
| Tespit edilen binalarda hasar Macro-F1 | 0,763 | **0,780** |
| Aynı binalarda oracle Macro-F1 | 0,791 | 0,790 |
| Katı uçtan uca Macro-F1 (kaçırılan = no-damage) | **0,710** | 0,700 |

Threshold ve erosion, validation'da önceden belirlenen ölçütle (lokalizasyon F1) seçildi: önce 5 threshold × 4 erosion,
seçim ızgara kenarında çıktığı için genişletilmiş ızgara (threshold 0,7–0,9 × erosion 4–8); erosion için en iyi değer
ızgaranın içinde (4), threshold için kazanç 0,9'a kadar azalarak sürdü (`phase5_audit/two_stage_sweep_val*.json`).

**Yorum:** Bina ayırma birleşmeleri %28 azalttı ve tespit edilen binalarda hasar tahmininin oracle'a farkını 0,028'den
0,009'a indirdi (minor F1 0,572 → 0,611). Bedeli, küçük/soluk binaların silinmesiyle kapsamanın %87'den %81'e düşmesidir;
kaçırılanlar hasarsız sayıldığında toplam skor 0,700'dür. Lokalizasyon kaynaklı kayıp, sınıflandırma hatasından daha
büyük bir sınırlamadır. U-Net PRE görüntüsünü kullandığı için yıkılmış binalar da büyük ölçüde bulunur (destroyed
kapsama %77–83). Örnek: `phase5_audit/two_stage_example_palu-tsunami_00000025*.png`.

## 5. Sınırlılıklar

1. Tüm seçimler validation üzerinden yapıldı; validation skorları iyimserdir. Test Faz 6'da tek seferlik değerlendirilecek.
2. Sınıflandırıcılar tek seed ile eğitildi; ≤ 0,01'lik farklar istatistiksel olarak ayırt edilemez.
3. Ortak eğitim politikası (lr 3e-4, 128 giriş) Transformer için uygun olmayabilir; ViT sonucu "bu politika altında" geçerlidir.
4. U-Net yalnızca bina varlığını öğrenir; bitişik binaları ayırmak için kenar/instance bilgisi yoktur. İşlem sonrası
   ayırma kısmi çözüm sağladı; kenar sınıflı (iç/kenar/arka plan) U-Net veya instance segmentasyon gelecek çalışmadır.
5. U-Net 30 epoch sınırında hâlâ iyileşiyordu; daha uzun eğitim isteğe bağlı iyileştirmedir.
6. Afet bazında hasar Macro-F1, bazı afetlerde bazı sınıflar hiç bulunmadığından (F1 = 0 sayılır) bu raporda verilmedi;
   uygun metriklerle Faz 6'da yapılacak.

## 6. Yeniden üretim ve Faz 6'ya devir

| Ne | Komut / dosya |
|---|---|
| Siamese ResNet34 | `python -m src.train --model siamese_resnet34 --loss focal --experiment-name phase5_siamese_resnet34_focal_seed42_w4 --num-workers 4` |
| Siamese ViT-Small | `python -m src.train --model siamese_vit_small --loss focal --experiment-name phase5_siamese_vit_small_focal_seed42_w4 --num-workers 4` |
| U-Net | `python -m src.train_segmentation --experiment-name phase5_unet_resnet34_seed42_w4` |
| Sınıflandırıcı denetimi | `python -m src.evaluation.audit_run --experiment <ad> --output results/phase5_audit/audit_<ad>.json` |
| U-Net denetimi | `python -m src.evaluation.audit_segmentation --experiment <ad> --output results/phase5_audit/audit_<ad>.json` |
| İki aşamalı örnek | `python -m src.evaluation.two_stage predict --image-id <val id> --threshold 0.9 --split-erosion 4 --output <png>` |
| İki aşamalı val değerlendirmesi | `python -m src.evaluation.two_stage evaluate-val --threshold 0.9 --split-erosion 4 --output <json>` |
| Sınıflandırma karşılaştırması | `python -m src.training.compare --kind models --runs <4 koşu> --output results/phase5_classification_validation.csv` |

Checkpoint'ler (`best.pth`, `last.pth`) ve koşu klasörleri GitHub'a gönderilmez; yedek: `C:\Users\bhrto\faz5_yedek`
(Faz 4 için `C:\Users\bhrto\faz4_yedek_20261104`). Faz 6'da test değerlendirmesi için gereken modeller:
ResNet34 early fusion (seçilen sınıflandırıcı), Siamese ResNet34, Siamese ViT-Small, U-Net ve seçilen iki aşamalı
ayar (threshold 0,9, erosion 4). Pınar'ın Faz 4 Simple CNN checkpoint'leri Pınar'ın bilgisayarındadır.
