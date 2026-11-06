# Faz 6 — Test Değerlendirmesi: Sonuç Raporu

Tarih: 6 Kasım 2026 · Branch: `feature/phase6-evaluation` · Hazırlayan: Bahar (Claude ile)

Test seti, önceden kayıtlı ve Pınar tarafından onaylanan protokole (`results/phase6_protocol.md`, onay commit'i
`5396622`) göre **tek seferlik** açıldı. Bütün değerlendirmeler aynı oturumda, protokoldeki sırayla çalıştırıldı
(15:22–15:53). Her sonuç dosyası protokol commit'ini ve model `best.pth` SHA-256 değerini içerir
(`results/phase6_test/*.json`). Test sonuçları görüldükten sonra hiçbir model, eşik, metrik veya örnek seçimi
değiştirilmedi; **protokolden sapma yoktur.**

## 1. Özet

- **En yüksek test Macro-F1'ini Siamese ResNet34 verdi (0,760).** Validation'a göre seçilen ResNet34 early fusion
  testte üçüncü oldu (0,736). Protokol gereği seçim değiştirilmedi; iki aşamalı sistem early fusion ile değerlendirildi.
- **U-Net bina lokalizasyonu testte de aynı başarıyı korudu:** piksel Dice 0,8609 / IoU 0,7557 (val 0,8609).
- **İki aşamalı sistem:** tespit edilen binalarda Macro-F1 0,733 (kapsama %80,8); kaçırılanlar no-damage kabul
  edilince 0,660; xView2-benzeri skor 0,684.
- **Görülmemiş afet (hurricane-matthew):** aynı 3.968 test binasında Macro-F1, matthew'i gören modelde 0,535 iken
  görmeyen modelde **0,272**'ye düştü. Görmeyen model matthew'deki minor binaların %86'sını no-damage tahmin etti.

## 2. Hasar sınıflandırma (test: 53.806 bina, 739 görüntü)

| Model | Test Macro-F1 | Val Macro-F1 | no | minor | major | destroyed | Acc. | Parametre | Eğitim süresi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Siamese ResNet34** | **0,7598** | 0,7773 | **0,955** | 0,589 | 0,669 | **0,827** | **0,896** | 21,68 M | 62 dk |
| Siamese ViT-Small (DeiT-S) | 0,7564 | 0,7701 | 0,951 | **0,595** | **0,682** | 0,797 | 0,891 | 21,91 M | 100 dk |
| ResNet34 early fusion *(val ile seçilen)* | 0,7363 | 0,7789 | 0,951 | 0,560 | 0,649 | 0,786 | 0,887 | 21,30 M | 50 dk |
| Simple CNN | 0,7011 | 0,7411 | 0,934 | 0,518 | 0,600 | 0,752 | 0,861 | 0,09 M | 3 sa 4 dk* |

\* Farklı bilgisayar ve `num_workers=2`. Test çıkarım hızı tüm modellerde 361–386 çift/sn'dir (veri okuma
dahil, aynı GPU); bu hızlar veri okumayla sınırlı olduğundan model hızı karşılaştırması değildir.

**Değerlendirme:**
- Val → test düşüşü Siamese modellerde 0,014–0,018, early fusion'da 0,043, Simple CNN'de 0,040'tır. Önceden
  belirlenen kurala göre (en yüksek val Macro-F1) seçilen model testte en iyi değildir; validation'daki ≤ 0,009'luk
  farklar test sıralamasını öngörmedi. Tek seed ile çalışıldığı için **istatistiksel anlamlılık test edilmedi**;
  bulgu, Siamese yapının (PRE ve POST için ayrı özellik + fark) bu veride daha iyi genelleştiği yönünde bir
  **gözlem** olarak raporlanır.
- Tüm modellerde en zor sınıf **minor-damage**'dir (F1 0,52–0,60). Early fusion'da minor binaların %31,4'ü
  (1.505 / 4.793) no-damage tahmin edildi; Siamese ResNet34'te bu oran %30,5 (1.464). Siamese'de asıl fark minor
  recall'un 0,464'ten 0,515'e ve destroyed precision'ın 0,750'den 0,825'e yükselmesidir.
- xView2-benzeri hasar F1 (dört sınıf F1'inin harmonik ortalaması, gerçek kutularla): Siamese ResNet34 0,734,
  Siamese ViT 0,734, early fusion 0,708, Simple CNN 0,667.

## 3. U-Net bina lokalizasyonu (test: 933 görüntü)

| Ölçüm | Test | Val (Faz 5, tam görüntü) |
|---|---:|---:|
| Piksel Dice (F1) | **0,8609** | 0,8637 |
| IoU | 0,7557 | 0,7601 |
| Precision / Recall | 0,867 / 0,854 | 0,876 / 0,852 |
| Görüntü bazlı Dice (binalı 752 görüntü) | 0,786 | 0,802 |
| Binasız görüntüde yanlış alarm (181 görüntü) | %6,1 | %7,9 |
| Çıkarım süresi | 0,10 sn / görüntü | – |

Test Dice'ı (0,86087) eğitimdeki seçim değeriyle (val kesitleri 0,86094) neredeyse aynıdır; bu bir rastlantıdır,
iki değer farklı veri ve farklı hesaplama düzeniyle (tam görüntü / 512 kesit) ölçülmüştür.

**Afet türüne göre Dice:** tsunami 0,890 · flooding 0,881 · fire 0,852 · wind 0,840 · earthquake 0,837 ·
volcano 0,406 (yalnızca 5 görüntü). Afet bazında en düşük değerler hurricane-matthew (0,762) ve socal-fire (0,827).

## 4. İki aşamalı sistem (test: 933 görüntü, 53.806 uygun bina)

Sınıflandırıcı: ResNet34 early fusion (protokolde donmuş). Metrikler protokoldeki adlarıyla, ayrı raporlanır.

| Metrik | **t=0,9, erosion 4 (seçilen)** | t=0,5, erosion 0 (ikincil) |
|---|---:|---:|
| Lokalizasyon F1 (IoU ≥ 0,5) · P / R | 0,536 · 0,632 / 0,465 | 0,499 · 0,674 / 0,396 |
| Birden çok binayı içeren bileşen (içindeki bina) | 4.955 (19.236) | 5.376 (26.880) |
| Uygun binaların kapsanma oranı | %80,8 | %87,6 |
| **M1** — Oracle Macro-F1 (gerçek kutular) | 0,7363 | 0,7363 |
| **M2** — Yalnızca tespit edilenlerde Macro-F1 | **0,7329** (aynı binalarda oracle 0,7468) | 0,6976 (oracle 0,7457) |
| **M3** — Uçtan uca Macro-F1, kaçırılan = no-damage | **0,6598** | 0,6531 |
| **M4** — xView2-benzeri hasar F1, kaçırılan = yanlış negatif (harmonik) | **0,6085** | 0,6070 |
| xView2-benzeri skor S (uçtan uca / oracle) | **0,684** / 0,754 | 0,683 / 0,754 |
| Çıkarım süresi (oracle geçişi dahil) | 0,62 sn / görüntü | 0,43 sn / görüntü |

S = 0,3 × piksel F1 (0,8609) + 0,7 × hasar F1. Bu skor resmî xView2 skoru değildir (resmî hasar F1 piksel
düzeyinde, bizimki bina düzeyinde).

**Değerlendirme:** Validation'da seçilen bina ayırma ayarı testte de ikincil ayardan iyi çıktı (M2 +0,035, M3 +0,007).
Bina ayırma, tespit edilen binalarda sınıflandırmanın oracle'a farkını 0,048'den 0,014'e indirdi; bedeli kapsamanın
%87,6'dan %80,8'e düşmesidir. Kapsama sınıfa göre: no-damage %82,0, minor %75,5, major %82,7, destroyed %72,7.
Uçtan uca skoru en çok sınırlayan etken lokalizasyondur (bitişik binaların birleşmesi ve küçük binaların kaçırılması).

## 5. Afet bazlı analiz (ResNet34 early fusion, test)

Kural (protokol bölüm 5): sınıf F1 yalnızca o afette en az 30 test binası olan sınıflar için; "mevcut sınıf
Macro-F1" yalnızca bu sınıfların ortalaması (parantezde sınıf sayısı); 30'dan az binalı afette yalnızca accuracy.

| Afet | Tür | Bina | no | minor | major | destroyed | Mevcut sınıf Macro-F1 | Acc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santa-rosa-wildfire | fire | 4.213 | 0,983 | n < 30 | n < 30 | 0,949 | 0,966 (2) | 0,971 |
| palu-tsunami | tsunami | 12.401 | 0,976 | n < 30 | 0,072 | 0,840 | 0,629 (3) | 0,949 |
| hurricane-florence | flooding | 2.116 | 0,966 | 0,000 | 0,859 | n < 30 | 0,608 (3) | 0,925 |
| socal-fire | fire | 4.098 | 0,969 | n < 30 | 0,051 | 0,798 | 0,606 (3) | 0,938 |
| hurricane-matthew | wind | 3.968 | 0,468 | 0,743 | 0,236 | 0,691 | 0,535 (4) | 0,647 |
| hurricane-michael | wind | 5.597 | 0,846 | 0,452 | 0,305 | 0,430 | 0,508 (4) | 0,722 |
| hurricane-harvey | flooding | 7.599 | 0,869 | 0,145 | 0,762 | 0,087 | 0,466 (4) | 0,770 |
| midwest-flooding | flooding | 2.388 | 0,985 | 0,000 | n < 30 | 0,373 | 0,453 (3) | 0,958 |
| mexico-earthquake | earthquake | 11.396 | 0,995 | 0,000 | 0,000 | n < 30 | 0,332 (3) | 0,990 |
| guatemala-volcano | volcano | 30 | – | – | – | – | – | 0,867 |

Afet türüne göre mevcut sınıf Macro-F1: tsunami 0,629 (3) · wind 0,591 (4) · flooding 0,493 (4) · fire 0,479 (4) ·
earthquake 0,332 (3); volcano yalnızca accuracy 0,867 (30 bina).

**Gözlemler:**
- Yangın afetlerinde destroyed sınıfı çok iyi ayrılıyor (0,80–0,95); yanmış yapılar POST görüntüde belirgindir.
- Hasar dağılımı az olan sınıflarda model çoğunlukla başarısız: mexico-earthquake'te minor (84 bina) ve major
  (31 bina) hiç doğru tahmin edilmedi; palu-tsunami'de major 0,072, socal-fire'da major 0,051.
- Accuracy, baskın no-damage sınıfı nedeniyle yanıltıcıdır (mexico-earthquake: accuracy 0,990, mevcut sınıf
  Macro-F1 0,332).

## 6. Genelleme testi: görülmemiş afet (hurricane-matthew)

İki model aynı politikayla eğitildi; fark yalnızca eğitimde matthew'in bulunup bulunmamasıdır (protokol bölüm 6).

| Model | Değerlendirilen binalar | Macro-F1 | no | minor | major | destroyed | Acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Standart (matthew'i eğitimde görmüş) | matthew Challenge test, 3.968 | **0,535** | 0,468 | **0,743** | 0,236 | 0,691 | 0,647 |
| Görülmemiş afet modeli | matthew Challenge test, aynı 3.968 | **0,272** | 0,321 | **0,060** | 0,112 | 0,594 | 0,280 |
| Görülmemiş afet modeli | matthew train-source, 13.152 | 0,266 | 0,362 | 0,082 | 0,079 | 0,540 | 0,292 |

Matthew'in gerçek dağılımı: no 616 · minor 2.337 · major 336 · destroyed 679 (Challenge test).

**Değerlendirme:** Görülmemiş bir afette Macro-F1 yarıdan fazla düştü (0,535 → 0,272). Görmeyen model gerçek minor
binaların 2.003'ünü (%86) no-damage tahmin etti; matthew'i gören model bu binaların 1.644'ünü (%70) doğru buldu.
Matthew'in minor hasarı (kasırgaya bağlı çatı hasarı, etrafa saçılmış döküntü) diğer afetlerdeki minor hasardan
görsel olarak farklıdır ve matthew çıkarıldığında eğitimdeki minor örnek sayısı 12.060'tan 7.039'a düşer.
Destroyed sınıfı görece korunuyor (0,691 → 0,594): yıkılmış bina afet türünden bağımsız olarak daha belirgindir.
Sonuç, modellerin **afete özgü hasar görünümünü öğrendiğini** ve yeni bir afet türünde, özellikle hafif hasar
için, doğrudan kullanılmadan önce o afetten etiketli örnek gerektirdiğini gösterir. Tek bir holdout afetiyle
yapılan bu test, diğer afetlere genellenmeden yorumlanmalıdır.

## 7. Grad-CAM ve hata galerisi

Seçilen sınıflandırıcı (ResNet34 early fusion), `backbone.layer4`. Model PRE ve POST'u tek 6 kanallı girdi olarak
işlediği için her bina için **tek bir harita** vardır ve PRE/POST'ta aynı konumu gösterir. Örnekler sonuçtan bağımsız
sabit kuralla seçildi: her gerçek sınıf için seed 42 ile 4 doğru + 4 yanlış (toplam 32).
Görseller: `phase6_test/gradcam_test_{no-damage,minor-damage,major-damage,destroyed}.png`.

Gözlemler (görsel inceleme; nicel iddia değildir):
- **Doğru destroyed tahminlerinde** dikkat binanın ortasındaki enkaz bölgesinde yoğunlaşıyor.
- **Minor sınıfında doğru ve yanlış tahminlerin ikisinde de** dikkat binanın merkezinde; yani hatalar çoğunlukla
  "yanlış yere bakmaktan" değil, hafif hasarın 128×128 kırpıntıda görsel olarak çok ince kalmasından kaynaklanıyor.
- Görülen hata türleri: PRE ve POST arasında ışık/çekim açısı farkı (aynı çatının farklı parlaklıkta görünmesi),
  ağaç örtüsü ve gölge, kırpıntının siyah dolgulu kenarları, POST'ta tamamen kaybolmuş binanın (yalnızca zemin
  kalmış) no-damage sanılması (validation incelemesinde görüldü).

## 8. Sınırlılıklar

1. Tüm sınıflandırıcılar tek seed ile eğitildi; model farkları için istatistiksel anlamlılık test edilmedi.
2. Model seçimi tek bir validation bölmesine dayandı ve test sıralamasını öngörmedi (bölüm 2).
3. İki aşamalı sistem protokol gereği early fusion sınıflandırıcısıyla değerlendirildi; testte daha iyi olan Siamese
   ResNet34 ile bir uçtan uca değerlendirme yapılmadı (yapılırsa yeni, ön kayıtsız bir analiz olur).
4. Afet bazlı analizde birçok afet/sınıf hücresinde örnek sayısı 30'un altında; bu hücreler raporlanmadı.
5. Genelleme testi tek bir holdout afetiyle (hurricane-matthew) yapıldı.
6. xView2-benzeri skor bina düzeyindedir; resmî xView2 sonuçlarıyla doğrudan karşılaştırılamaz.

## 9. Dosyalar ve yeniden üretim

| Ne | Dosya / komut |
|---|---|
| Protokol (onaylı) | `results/phase6_protocol.md` (commit `5396622`) |
| Test sonuçları (JSON) | `results/phase6_test/{classifier,segmentation,two_stage}_*.json` |
| Afet analizi | `results/phase6_test/analysis_disaster_resnet34_early_fusion.json`, `analysis_disaster_unet.json`, `analysis_unseen_matthew.json` |
| Grad-CAM | `results/phase6_test/gradcam_test_*.png` |
| Model doğrulamaları | `results/phase6_audit/` (matthew modeli audit, Simple CNN checkpoint doğrulaması) |
| Değerlendirme aracı | `python -m src.evaluation.evaluate_test {segmentation,classifier,two-stage} ...` (tek seferlik; var olan sonucu yeniden yazmaz) |
| Analiz araçları | `python -m src.evaluation.disaster_analysis ...`, `python -m src.evaluation.gradcam ...` |

Bina/görüntü bazlı tahmin CSV'leri (`results/phase6_test/*.csv`) ve checkpoint'ler GitHub'a eklenmez; yedek:
`C:\Users\bhrto\faz6_yedek` (+ `faz4_yedek_20261104`, `faz5_yedek`).
