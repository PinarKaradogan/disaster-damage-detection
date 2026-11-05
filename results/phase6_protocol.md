# Faz 6 — Test Değerlendirme Protokolü (ön kayıt)

Durum: **taslak, onay bekliyor** · Hazırlayan: Bahar (Claude ile) · Onay: Pınar
Bu dosya test seti açılmadan **önce** commit edilir. Commit tarihi, aşağıdaki kararların test sonuçlarından önce
verildiğinin kaydıdır. Test sonuçları görüldükten sonra bu dosyadaki hiçbir karar değiştirilmez; bir sapma
gerekirse ayrı bir "sapma" bölümüne gerekçesiyle eklenir ve raporda belirtilir.

## 1. Kurallar

1. Test seti yalnızca bu protokoldeki değerlendirmeler için, **her değerlendirme bir kez** çalıştırılır.
   Değerlendirme aracı (`src/evaluation/evaluate_test.py`) var olan bir sonuç dosyasının üzerine yazmayı reddeder.
2. Test sonuçlarına bakarak model, eşik, post-processing, metrik veya örnek seçimi değiştirilmez.
3. Aşağıdaki model dosyaları donmuştur; SHA-256 değerleri değerlendirmeden önce doğrulanır.
4. Hiçbir model test verisiyle eğitilmez veya ayarlanmaz. Tek yeni eğitim (bölüm 6) test verisini kullanmaz.

## 2. Donmuş modeller

| Rol | Koşu | `best.pth` SHA-256 | Val sonucu (kayıtlı) |
|---|---|---|---|
| Sınıflandırıcı (seçilen) | `phase4_final_resnet34_focal_seed42_w4` | `16f10bd6d85048ccc950001248a95486388ba5d0fc4f512abb8e384443ceeb11` | Macro-F1 0,7789 |
| Sınıflandırıcı | `phase5_siamese_resnet34_focal_seed42_w4` | `52307a9356b798aa2213e54e66e3b65d6afd283231d5f8e480c48cb490785e85` | Macro-F1 0,7773 |
| Sınıflandırıcı | `phase5_siamese_vit_small_focal_seed42_w4` | `1ea45bf6eae0e4313d9ea08b3dc16ecad3bca8498da442ecfa07babba1553d4f` | Macro-F1 0,7701 |
| Sınıflandırıcı | `phase4_cnn_focal_seed42_retry_20261018` (Pınar) | dosya geldiğinde kaydedilir* | Macro-F1 0,7411 |
| Genelleme (matthew görmemiş) | `phase6_unseen_matthew_resnet34_focal_seed42_w4` | eğitim + val denetimi sonrası kaydedilir | — |
| Lokalizasyon | `phase5_unet_resnet34_seed42_w4` | `9d80a6aabe4a9d9b1eb712d1d8567c3bbfa60229e7838074ad1fe27ddade8604` | Piksel Dice 0,8609 |

\* Simple CNN `best.pth` Pınar'ın bilgisayarında. Geldiğinde, test öncesinde şu doğrulama yapılır ve sonucu bu
dosyaya eklenir: (a) checkpoint config'i kayıtlı `config.json` ile aynı (config.json SHA-256
`fdfb5272b3648c89d16dd78fc4d50509478f199a2835e899db2940114528ce3c`), (b) validation'da yeniden değerlendirme
kayıtlı 0,7411'i 1e-4 tolerans içinde verir. Doğrulama geçmezse CNN test edilmez ve bu raporlanır.

İki aşamalı sistem ayarı (validation'da lokalizasyon F1 ile seçildi): **threshold 0,9, split erosion 4,
min area 10 px**, sınıflandırıcı ResNet34 early fusion. İkincil (önceden bildirilen) karşılaştırma: threshold 0,5,
erosion 0.

## 3. Test verisi

- Sınıflandırma: `patches.csv` içinde `split == test` olan **53.806** uygun bina çifti (739 görüntü);
  manifest SHA-256 `383da7b841c0bd8b11b9cf807873b029291ad23b3c3f48135d9b7dde1ffe11ec`.
- Lokalizasyon ve iki aşamalı sistem: `splits/standard/test_images.txt` içindeki **933** görüntünün tamamı
  (`data/test/images`, `data/test/targets`, `data/test/labels`).

## 4. Metrikler

**Sınıflandırma (her sınıflandırıcı):** 4 sınıflı Macro-F1 (ana metrik), sınıf bazlı precision/recall/F1,
confusion matrix (satır = gerçek), accuracy. Hesap Faz 4/5 ile aynı fonksiyon (`src/training/metrics.py`).
Verimlilik: parametre sayısı, kayıtlı eğitim süresi, test çıkarım hızı (çift/sn, aynı GPU, batch 32, AMP).

**Lokalizasyon (U-Net):** tam 1024×1024 test görüntülerinde global piksel Dice (F1) ve IoU, threshold 0,5
(eğitimdeki seçim eşiği); görüntü bazlı Dice (binalı görüntüler), binasız görüntülerde yanlış alarm oranı;
çıkarım süresi (sn/görüntü). Boş maske kuralı Faz 5 ile aynıdır.

**İki aşamalı sistem:** aşağıdaki değerler **ayrı adlarla** raporlanır, birbirine karıştırılmaz ve tek bir
"sistem skoru" olarak birleştirilmez. Hepsi 53.806 uygun test binası üzerinden, 4 sınıf için hesaplanır.

| Ad (raporda bu adla) | Hangi binalar | Kaçırılan bina nasıl sayılır | Özet |
|---|---|---|---|
| **M1 — Oracle Macro-F1** | tümü, gerçek kutularla kırpılır | kaçırılan yok | sınıflandırıcının tek başına başarısı |
| **M2 — Tespit edilenlerde Macro-F1** | yalnızca U-Net'in bulduğu binalar | hesaba girmez | + kapsama oranı ayrıca verilir |
| **M3 — Uçtan uca Macro-F1 (kaçırılan = no-damage)** | tümü | tahmini **no-damage** kabul edilir: gerçek sınıfı no-damage ise doğru, değilse kendi sınıfı için yanlış negatif **ve** no-damage için yanlış pozitif | Faz 5'teki 0,700'ün tanımı |
| **M4 — xView2-benzeri hasar F1 (kaçırılan = yanlış negatif)** | tümü | yalnızca kendi sınıfı için **yanlış negatif**; hiçbir sınıfa yanlış pozitif yazılmaz | dört sınıf F1'inin **harmonik** ortalaması |

M3 ile M4'ün tek farkı kaçırılan binanın sayılış biçimidir; M3 Macro-F1 (aritmetik ortalama), M4 harmonik
ortalamadır. Ayrıca lokalizasyon F1 (IoU ≥ 0,5) ve birleşik bileşen sayıları raporlanır.

**xView2-benzeri skor (literatürle kaba kıyas için):**
S = 0,3 × F1_lok + 0,7 × F1_hasar, F1_lok = U-Net test global piksel F1 (threshold 0,5). İki sürüm:
- *S_uçtan uca:* F1_hasar = **M4**;
- *S_oracle:* F1_hasar = M1'deki dört sınıf F1'inin harmonik ortalaması (üst sınır).
Bu skor resmî xView2 skoru **değildir**: resmî hasar F1 piksel düzeyindedir, bizimki bina düzeyindedir.

## 5. Afet bazlı analiz kuralları

- Her afet ve afet türü için (flooding: florence, harvey, midwest · wind: matthew, michael · fire: santa-rosa,
  socal · earthquake: mexico · tsunami: palu · volcano: guatemala) örnek sayıları her tabloda verilir.
- Bir afette **en az 30 test binası** olan sınıflar için sınıf F1 raporlanır; daha az olan hücreler "n < 30"
  diye işaretlenir. Afet Macro-F1'i yalnızca bu sınıfların F1 ortalamasıdır ("mevcut sınıf Macro-F1") ve kaç
  sınıfın dahil olduğu yazılır. Toplam 30'dan az binası olan afetler (guatemala-volcano: 30) yalnızca accuracy ile verilir.
- Afet bazlı analiz seçilen sınıflandırıcı (ResNet34 early fusion) ve U-Net (piksel Dice) için yapılır.

## 6. Genelleme testi (görülmemiş afet: hurricane-matthew)

- Model: ResNet34 early fusion, Faz 4 ile **aynı politika** (Focal, AdamW 3e-4, batch 32, 30 epoch, patience 5,
  seed 42, 128 giriş, 4 worker), `splits/unseen_disaster/hurricane-matthew/{train,val}_images.txt` ile eğitilir.
  Model seçimi bu val bölmesinde (matthew içermez) yapılır. Yeni hiperparametre denemesi yapılmaz.
- Manifest: `data/processed/patches_unseen_hurricane-matthew/patches.csv`
  (`python -m src.preprocessing.unseen_manifest --holdout-disaster hurricane-matthew`), SHA-256
  `270c5cdf19a27b7b9adca8974d2641ddb5602993640231dc1f67d29368018df6`. Doğrulandı: train (123.167 bina, 1.712
  görüntü) ve val (23.324 bina, 308 görüntü) bölmelerinde **hiç hurricane-matthew satırı yok**; holdout 17.120
  matthew binası (13.152 train-source + 3.968 Challenge test; ikincisi standart test setindeki matthew sayısıyla aynı).
  Not: matthew binalarının %52'si minor-damage'dır; matthew çıkarılınca eğitimdeki minor örnek sayısı 12.060'tan
  7.039'a düşer.
- Eğitim: `python -m src.train --model resnet34_early_fusion --loss focal --manifest
  data/processed/patches_unseen_hurricane-matthew/patches.csv --experiment-name
  phase6_unseen_matthew_resnet34_focal_seed42_w4 --num-workers 4`. Bu modelin `best.pth` SHA-256'sı, **test
  sonuçlarına bakılmadan önce**, eğitim bitip validation denetimi geçince bölüm 2'ye eklenir.
- Karşılaştırma: matthew **Challenge test** görüntülerinde (`holdout_challenge_test_images.txt`, 73 görüntü)
  (a) standart model (matthew'i eğitimde görmüş) ve (b) matthew'i hiç görmemiş model. Ek olarak (b) matthew'in
  train-source görüntülerinde (`holdout_train_source_images.txt`) raporlanır. Metrikler bölüm 4–5 ile aynı.

## 7. Grad-CAM ve hata galerisi

- Seçilen sınıflandırıcı; son konvolüsyon bloğu (`layer4`) üzerinde Grad-CAM, PRE ve POST girişleri için.
- Örnek seçimi sonuçtan bağımsız ve sabit: test binalarından her gerçek sınıf için seed 42 ile 4 doğru ve
  4 yanlış tahmin (toplam 32). Hata nedenleri (bulut, gölge, hizalama, komşu sınıf belirsizliği) görsel
  incelemeyle etiketlenir; nicel iddia olarak sunulmaz.

## 8. Çıktılar

`results/phase6_test/`: her değerlendirme için JSON (metrikler, model SHA-256, protokol commit'i, süre) ve
bina bazlı tahmin CSV'si (CSV'ler GitHub'a eklenmez); `results/phase6_report.md`: sonuçlar ve yorum.
