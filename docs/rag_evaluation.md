# Adaptive RAG değerlendirme kaydı

Tarih: 22 Eylül 2026. Ortam: Windows, Python 3.13.1, Chroma 1.3.5. Ücretli model çağrısı yapılmadı. Üretim Chroma dizini ve BookStack verisi kullanılmadı; ölçümler geçici dizinlerde çalıştı.

## Ne ölçüldü

`python -m pytest -q` sonucu: 35 test geçti. Aynı sonuç, çalışan servisi durdurmadan açılan tek kullanımlık konteynerde de alındı. Bu konteyner güncel kodu `/app` altına ve widget dosyasını `/widget` altına bağladı. Çalışan `bookstack_rag_service` yeniden oluşturulmadı; `/ready` hâlâ eski imaja aittir.

Güvenlik ve tutarlılık fixture'larında bu koşuda ihlal görülmedi:

- Eksik, sahte ve süresi dolmuş token; AI izni kapalı kullanıcı; servis sırrı olmadan yönetim ucu.
- Widget betiğinde servis sırrı ve `X-RAG-Token` yok.
- Yasak sayfa kimliği ve `SALARY-991` / `MERGER-228` terimleri, izinli olmayan kapsamın yanıtına girmedi.
- Eski arama yolu selamlama ve aramada tüm katalog metadata taraması yapmıyor. İstemci `top_k` değeri sunucu limitinin (`LEGACY_RESULT_LIMIT`, varsayılan 8) yerine geçmiyor.
- Başlık kelimesi yanıtta geçti diye aktif sayfa kaynak olmuyor.
- Lexical yazımı başarısız olunca eski revizyon kalıyor. Yayın öncesi kesilen indeks kurtarılabiliyor. Tombstone daha eski iş ile geri gelmiyor. Değişmeyen içerikte embedding çağrısı artmıyor.
- Eksik tarama veya okuma hatası toplu silme üretmiyor.
- Dış host ve yönlendirme hedefi BookStack kimlik başlığını almıyor. Aynı URL'de değişen bayt ve prompt sürümü yeniden analiz ediliyor.

Etiketli set: 23 ayrı sayfa, 60 soru (`rag_service/eval/questions.json`). Kalibrasyon ve holdout ayrımı dosyadadır. Holdout içinde beklenen sayfası olan 16 sorunun kanıt sayfa recall'u bu fixture'da 1.00 oldu. `simple` işaretli soruların tamamı ek retrieval turu açmadı. Bu sayı, hash bag-of-words gömme ve FTS5 ile, soruların sayfalardaki ayırt edici ifadeleri taşıdığı bir fixture sonucudur.

## Ne ölçülmedi

Aşağıdakilere başarı yazılmamıştır. Bu kapılar kapanmadan `ADAPTIVE_RAG` varsayılanı `off` kalır ve canlı kesim yapılmaz.

- Üretim korpusunda `all-MiniLM-L6-v2` kalitesi. İzole 23 sayfalık fixture'da aynı modelle holdout kanıt recall'u 16/16 oldu. Bu sayfalar sorudaki ayırt edici ifadeyi taşıdığı için semantik kalite onayı değildir. Ayrıca iki İngilizce yoklama yapıldı: kodu yazmayan “no code is listed” cümlesi kanıt sayılmadı ve yanıt Türkçe `ATL-19` sayfasına gitti; “Project Atlas owner: Selin Karaca” ile “Deniz Aksoy” çelişki olarak işaretlendi ve çelişki cümlesi İngilizce kuruldu.
- Ücretli yanıt modeli ile cevap doğruluğu, gerçek tokenizer usage veya baseline token karşılaştırması.
- 1.000 kitap / 10.000 sayfa / yaklaşık 100.000 chunk kapasitesi. 20 sayfalık izole smoke: indeks 1.196 saniye, üç aramanın ortanca süresi 0.0154 saniye. Bu p99 veya 10 eşzamanlı warm SLO kanıtı değildir.
- BookStack'in çalışan sürümü bu oturumda konteynerden okunmadı. İzin modeli resmi API dokümantasyonu ve kurulu Chroma kaynak kodu ile sınırlandı.

## Baseline

İzole koleksiyonda 30 sayfa, hash gömme ile yazıldı ve gerçek `RAGEngine._get_indexed_catalog` üç sayfalık izin filtresiyle çağrıldı. Katalog metni 444 karakterdi; karakter/4 tahmini 111 token. Bu, üretim kataloğunun token maliyeti değildir. Aynı çağrı `metadata_scans` sayacını 1 artırdı. Normal arama testinde bu sayaç 0 kaldı.

Eski karakter parçalayıcı, boş satırı olmayan uzun bir paragrafta 2123 karakterlik tek parça üretti. Yeni parçalayıcı child token sınırında böler ve 256 token kesmesinden küçük kalmayı hedefler.

## Eşikler

Güvenlik fixture'ında bu koşu için ihlal sayısı 0. Fixture recall eşiği 0.90 bu holdout kümesinde hash gömme ve MiniLM ile sayı olarak geçti. Küme küçük ve ifadeler ayırt edici olduğu için genel kalite onayı sayılmaz. Küçük testteki sıfır ihlal genel güvenlik ispatı değildir.
