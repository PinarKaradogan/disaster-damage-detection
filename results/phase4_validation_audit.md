# Faz 4 — Validation Denetimi ve Sonuç Değerlendirmesi

Tarih: 4 Kasım 2026 · Hazırlayan: Bahar (Claude ile) · Kapsam: yalnızca train/validation. **Test verisi açılmadı.**
Bu denetimde yeni eğitim yapılmadı, hiperparametre değiştirilmedi, mevcut deney dosyaları değiştirilmedi.

## 1. Bahar'ın bilgisayarındaki iki deneyin denetimi

Kontrol aracı: `python -m src.evaluation.audit_run --experiment <ad> --output results/phase4_audit/audit_<ad>.json` (salt okuma; çıktılar `results/phase4_audit/audit_*.json`).

| Kontrol | Weighted Sampler (`phase4_cnn_sampler_ce_seed42_w4`) | ResNet34 (`phase4_final_resnet34_focal_seed42_w4`) |
|---|---|---|
| `best.pth`, `last.pth`, `config.json`, `metrics.json`, `history.csv`, `status.json` | ✅ hepsi var | ✅ hepsi var |
| TensorBoard event dosyası (`runs/`) | ✅ 1 | ✅ 1 |
| `status = completed`, `smoke = false`, `evaluation_split = val` | ✅ | ✅ |
| Manifest SHA-256 dosyayla eşleşiyor (`383da7b8…1ec`) | ✅ | ✅ |
| Sınıf sırası `LABEL_MAP` ile aynı (0 no, 1 minor, 2 major, 3 destroyed) | ✅ | ✅ |
| `history.csv` satır sayısı = tamamlanan epoch | ✅ 17 | ✅ 9 |
| History'deki en iyi epoch/F1 = `metrics.json` | ✅ epoch 12 | ✅ epoch 4 |
| Early stopping tutarlı (en iyiden sonra 5 epoch iyileşme yok) | ✅ | ✅ |
| `best.pth` epoch ve config'i `config.json` ile birebir aynı | ✅ | ✅ |
| `last.pth` son epoch, `finished`, optimizer + RNG + early-stopping durumu içeriyor | ✅ | ✅ |
| `last.pth` içindeki en iyi model ağırlıkları = `best.pth` | ✅ | ✅ |
| Validation örnek sayısı 25.615, yalnızca `split == val` | ✅ | ✅ |
| `best.pth` validation'da yeniden değerlendirildi; Macro-F1 kayıtla aynı | ✅ fark 0,0 | ✅ fark 0,0 |

**Eksik dosya yok.** Yedek: iki deney klasörü, TensorBoard kayıtları ve bu denetim dosyaları GitHub dışında
`C:\Users\bhrto\faz4_yedek_20261104` klasörüne kopyalandı (736 MB, 16 sonuç dosyası; SHA-256 listesi
`yedek_sha256.txt`, kaynakla birebir eşleşiyor). Kayıtlı sürümler: Python 3.13.16, PyTorch 2.14.1+cu132, timm 1.0.30, RTX 4050 Laptop.

## 2. Karşılaştırılabilirlik

İki koşuda da aynı olanlar: manifest SHA-256, train/val sayıları (134.028 / 25.615), train sınıf sayıları, seed 42,
128×128 giriş, ImageNet normalizasyonu, train augmentasyonu (flip 0,5/0,5, ±15°), validation'da augmentasyon yok,
AdamW 3e-4 / 1e-4, Cosine (T_max 30, eta_min 1e-6), patience 5, AMP, batch 32. Metrik her koşuda aynı fonksiyonla
(`src/training/metrics.py`, sklearn, `labels=[0,1,2,3]`, `zero_division=0`, macro = 4 sınıf F1 ortalaması) hesaplanıyor.

**Bilinen farklar (kayda geçirilmeli):**

1. **`num_workers`: Bahar'ın koşuları 4, Pınar'ın koşuları 2.** Bu yalnızca hız ayarıdır; model, kayıp, veri ve
   metrik aynıdır. Etkisi augmentasyondaki rastgele sayı akışının worker'lar arasında farklı dağılmasıdır.
   **Ancak `src/training/compare.py` `num_workers` alanını `POLICY_KEYS` içinde tuttuğu için mevcut haliyle bu
   koşuları "Unfair comparison" diyerek reddedecektir.** Karar Pınar'ın: (a) `num_workers`'ı politika alanlarından
   çıkarıp raporda farkı belirtmek veya (b) karşılaştırmayı bu fark açıklanarak elle yapmak.
2. Farklı bilgisayar ve muhtemelen farklı Python yaması (Pınar 3.13.7 / Bahar 3.13.16). Seed aynı olsa da GPU
   sonuçları bit düzeyinde aynı değildir; bu, Faz 4 planında zaten kabul edilmiş durumdur.
3. Pınar'ın Focal ve Weighted CE koşuları resume altyapısından önce yapıldıysa `config.json` şeması farklı olabilir
   (ör. `training_mode`, `scheduler_t_max` alanları). Dosyalar gelince kontrol edilecek.

## 3. Pınar'ın bilgisayarından beklenen dosyalar

Her iki koşu için (`phase4_cnn_focal_seed42`, `phase4_cnn_weighted_ce_seed42` veya gerçek klasör adları):
`config.json`, `metrics.json`, `history.csv`, `status.json`. Mümkünse `best.pth` da (yeniden değerlendirme ve
confusion matrix için). Bu dosyalar gelmeden aşağıdaki tabloda **sınıf bazlı F1 değerleri boş bırakılmıştır;
tahmin edilmemiştir.**

## 4. Faz 4 sonuç tablosu (validation, best epoch)

| Model | Dengesizlik | Macro-F1 | No | Minor | Major | Destroyed | Acc. | Best ep. / toplam | Süre | Kaynak |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| ResNet34 full (ImageNet) | Focal (γ=2) | **0,7789** | 0,942 | 0,629 | 0,670 | 0,874 | 0,881 | 4 / 9 | 50 dk | Bahar, denetlendi |
| Simple CNN | Focal (γ=2) | 0,7411 | *bekleniyor* | *bekleniyor* | *bekleniyor* | *bekleniyor* | ≈0,850 | 14 / 19 | ≈3 sa 4 dk | Pınar, dosya bekleniyor |
| Simple CNN | Weighted Sampler + CE | 0,7179 | 0,888 | 0,562 | 0,623 | 0,799 | 0,805 | 12 / 17 | 74 dk | Bahar, denetlendi |
| Simple CNN | Weighted CE | 0,6977 | *bekleniyor* | *bekleniyor* | *bekleniyor* | *bekleniyor* | ≈0,793 | 10 / 15 | ≈1 sa 58 dk | Pınar, dosya bekleniyor |

Pınar'ın satırlarındaki doğruluk ve süre değerleri Pınar'ın raporundan alınmıştır; dosyalar gelince doğrulanacaktır.
Süreler farklı worker sayısı ve bilgisayarlarda ölçüldüğü için hız karşılaştırması olarak kullanılmamalıdır.

### Precision / recall (yalnızca dosyası denetlenen koşular)

| Model | Minor P / R | Major P / R | Destroyed P / R |
|---|---|---|---|
| ResNet34 + Focal | 0,695 / 0,575 | 0,686 / 0,654 | 0,873 / 0,875 |
| CNN + Weighted Sampler | 0,442 / 0,769 | 0,637 / 0,610 | 0,727 / 0,887 |

Confusion matrix'ler `results/phase4_audit/` altındaki JSON dosyalarındadır (satır = gerçek, sütun = tahmin).
Dikkat çekici gözlem: Weighted Sampler minor sınıfını sık tahmin ediyor (recall yüksek, precision düşük; 2.099
hasarsız bina minor olarak tahmin edilmiş). ResNet34 ise tersine minor binaların %29'unu (843 / 2.916) hasarsız
olarak tahmin ediyor. En zor ayrım her iki modelde de **no-damage ↔ minor-damage** arasında.

## 5. ResNet34 epoch analizi — overfitting değerlendirmesi

![ResNet34 öğrenme eğrileri](phase4_audit/resnet34_learning_curves.png)

| Epoch | Train loss | Val loss | Train F1 | Val F1 |
|---:|---:|---:|---:|---:|
| 1 | 0,2500 | 0,2031 | 0,575 | 0,667 |
| 2 | 0,1756 | 0,1799 | 0,689 | 0,713 |
| 3 | 0,1502 | 0,1496 | 0,725 | 0,762 |
| **4** | 0,1363 | **0,1426** | 0,746 | **0,779** |
| 5 | 0,1264 | 0,1443 | 0,760 | 0,749 |
| 6 | 0,1176 | 0,1475 | 0,772 | 0,768 |
| 7 | 0,1111 | 0,1472 | 0,780 | 0,754 |
| 8 | 0,1035 | 0,1435 | 0,787 | 0,778 |
| 9 | 0,0974 | 0,1625 | 0,800 | 0,766 |

**Kanıtlar (overfitting yönünde):**
- Train loss epoch 4'ten 9'a sürekli düşüyor (0,136 → 0,097; yaklaşık %29).
- Validation loss en düşük değerine epoch 4'te ulaşıyor (0,1426), sonra bir daha bu seviyeye inmiyor; epoch 9'da en yüksek değere çıkıyor (0,1625; yaklaşık +%14).
- Train Macro-F1 artmaya devam ederken (0,746 → 0,800) validation Macro-F1 0,749–0,778 bandında yatay kalıyor.

**Sınırlamalar (yorumu yumuşatan):**
- Validation Macro-F1 epoch 8'de 0,7783 ile neredeyse en iyi değere geri dönüyor; yani belirgin bir çöküş değil, **plato + dalgalanma** var.
- Ardışık epoch'lar arasındaki ortalama validation F1 değişimi 0,027; tek bir epoch'un en iyi seçilmesi bu dalgalanmadan kısmen etkilenir.
- Train metrikleri augmentasyonlu veride ve eğitim sırasında hesaplanıyor; train/val açığı yalnızca **eğilim** olarak yorumlanmalı (ilk epoch'larda train F1'in val'den düşük olması bundandır).

**Sonuç:** Epoch 4–5'ten sonra **hafif overfitting** başladığına dair tutarlı kanıt vardır. Early stopping ve
`best.pth` seçimi bunu doğru biçimde yönetmiştir; raporlanan model epoch 4 modelidir. Tüm katmanların hazır
ağırlıklarla 3e-4 öğrenme oranında eğitilmesi olası bir katkı faktörüdür, ancak bu bir **hipotezdir**, test
edilmemiştir. Bu denetimde hiperparametre değiştirilmemiştir.

## 6. Simple CNN Focal koşusu final CNN olarak kullanılabilir mi?

**Değerlendirme: Evet, aşağıdaki koşullar sağlanırsa ayrı bir final CNN eğitimi bilimsel olarak gerekli değildir.**

Gerekçe:
1. Planlanan "final CNN" koşusu, seçilen yöntemle (Simple CNN + Focal, sampler yok, seed 42, aynı veri ve ayarlar)
   **aynı yapılandırmanın** tekrarıdır. Aynı seed ve ayarlarla yeniden eğitmek yeni bilgi üretmez; yalnızca GPU
   kaynaklı küçük sayısal farklar oluşturur.
2. Seçim yalnızca validation Macro-F1 ile yapıldı ve test kullanılmadı. Bu yüzden mevcut koşuyu final olarak
   kullanmak test sızıntısı yaratmaz.
3. Yeniden eğitim, aynı validation setinde ikinci bir "şans" denemesi anlamına gelir; bu, seçim yanlılığını azaltmaz.

Koşullar (Pınar'ın dosyaları gelince doğrulanacak):
- `status = completed`, `smoke = false`, manifest SHA-256 `383da7b8…1ec`, val sayısı 25.615.
- Config'in deney planındaki politika ile aynı olduğu (ResNet34 ile aynı loss/sampler, aynı optimizer/scheduler).
- `best.pth` validation'da yeniden değerlendirildiğinde kayıtlı 0,7411 değerini vermesi.
- Raporda plandan sapmanın açıkça yazılması: *"Final CNN olarak `phase4_cnn_focal_seed42` koşusu yeniden kullanıldı; ayrı bir koşu yapılmadı."*

Raporlanması gereken sınırlama: Üç dengesizlik yöntemi arasındaki farklar (0,698 / 0,718 / 0,741) tek seed ile
ölçülmüştür. Ardışık epoch'lar arasındaki validation F1 dalgalanması CNN'de ortalama 0,016'dır. Bu nedenle Focal'ın
üstünlüğü **"bu deney koşullarında gözlenen"** olarak ifade edilmelidir; istatistiksel kesinlik için çoklu seed
gerekir (Faz 4 kapsamı dışında, isteğe bağlı).

## 7. Kısa akademik değerlendirme

Validation setinde (25.615 bina, 4 sınıf) ImageNet ön eğitimli ResNet34 early-fusion modeli, Focal Loss ile
**0,779 Macro-F1** elde ederek en iyi sonucu vermiştir. Bu değer aynı kayıp fonksiyonunu kullanan Simple CNN'e göre
yaklaşık 0,038 daha yüksektir. Ön eğitimli, daha yüksek kapasiteli bir modelin yaklaşık 225 kat daha fazla
parametreyle (21,3 M'ye karşı 94,9 K) daha iyi sonuç vermesi beklenen bir bulgudur. Model en iyi sonucuna 4.
epoch'ta ulaşmış, sonrasında hafif overfitting gözlenmiş ve early stopping bunu sınırlamıştır.

Sınıf dengesizliği için Simple CNN üzerinde karşılaştırılan üç yöntem arasında Focal Loss (0,741) en yüksek
Macro-F1'i vermiş, onu Weighted Sampler (0,718) ve Weighted CE (0,698) izlemiştir. Tüm modellerde en zor sınıflar
minor-damage ve major-damage'dir. Hatalar ağırlıklı olarak komşu hasar seviyeleri arasında, özellikle no-damage ile
minor-damage arasında yoğunlaşmaktadır. Destroyed sınıfı görsel olarak daha belirgin olduğundan yüksek F1 ile
ayrılabilmektedir.

Tüm skorlar model seçiminde de kullanılan validation setine aittir ve bu nedenle iyimser olabilir. Tarafsız
performans tahmini Faz 6'da, test seti üzerinde **tek seferlik** değerlendirmeyle elde edilecektir.
