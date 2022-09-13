# miuuleducation

RFM Analizi ile Müşteri Segmentasyonu
İş Problemi
Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor. Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak

Veri Seti Hikayesi

Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

Görev 1: Veriyi Anlama ve Hazırlama

Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
Adım 2: Veri setinde
a. İlk 10 gözlem,
b. Değişken isimleri,
c. Betimsel istatistik,
d. Boş değer,
e. Değişken tipleri, incelemesi yapınız.
Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.

Görev 2: RFM Metriklerinin Hesaplanması
Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.

Görev 3: RF Skorunun Hesaplanması
Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz. 
Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

Görev 4: RF Skorunun Segment Olarak Tanımlanması
Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

