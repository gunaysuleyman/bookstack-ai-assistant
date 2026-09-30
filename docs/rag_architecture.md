# Adaptive RAG mimarisi

Bu belge 22 Eylül 2026 tarihinde bu depoda doğrulanan modeli kaydeder. Üretim kesimi `ADAPTIVE_RAG=on` yapılmadan önce `docs/rag_evaluation.md` içindeki ölçülmemiş kapılar kapanmalıdır.

## Aktif yol

`ADAPTIVE_RAG=off` iken yanıt hâlâ `rag_engine.py` içindeki mevcut koleksiyon olan `bookstack_articles` üzerinden gelir. `on` iken yanıt `adaptive` paketinden ve ayrı koleksiyon `bookstack_articles_pc_v1` üzerinden gelir. `shadow` kullanıcıya eski yanıtı verir; yeni motorda yalnız retrieval çalışır ve `shadow_log` tablosuna sayfa kimlikleri ile süreyi yazar. Shadow ikinci bir yanıt modeli çağrısı yapmaz.

Hangi koleksiyonun aktif olduğu `GET /ready` yanıtındaki `active_collection` alanındadır.

## Kimlik ve ACL

BookStack API token'ı, token sahibinin görebildiği sayfaları döner. Başka bir son kullanıcının izinlerini soran bir endpoint yoktur. Bu nedenle izin listesi, oturum açmış kullanıcının tema şablonunda çalışan `Page::visible()` kapsamından gelir. Bunu role indirgemek BookStack'in sayfa, bölüm, kitap ve raf izin kalıtımını kaçırır.

İlk geçiş iki kanalla çalışır:

1. BookStack kabı, `RAG_INTERNAL_URL/api/scope` adresine servis sırrı ile sunucu tarafında POST atar. Tarayıcı yalnız `scope_ref` taşır.
2. Bu çağrı başarısız olursa tarayıcı imzalı `payload` ve `sig` taşır. HMAC anahtarı JavaScript'e yazılmaz.

`user_token` yoksa istek admin sayılmaz ve 401 döner. `can_use_ai=false`, bozuk imza, eksik tür veya süresi dolmuş `ts` içeriğe erişmez. Yönetim uçları (`/api/sync`, `/api/jobs/status`, `/api/scope`) ayrı `X-RAG-Token` servis sırrı ister.

İzin değişikliği için güvenilir bir anlık bildirim yoktur. `bookshelf_*` olayı raf nesnesini taşır; servis rafın güncel kitap listesini API'den okur ve o kitaplarla indekste hâlâ bu raf adıyla duran kitaplar için `book_refresh` kuyruğa alır. Kabul edilen eskime süresi `TOKEN_TTL_SECONDS` (varsayılan 900 saniye) kadardır. Süre dolunca arama yapılmaz; kullanıcı sayfayı yenileyince yeni kapsam üretilir. Anında iptal garantisi yoktur.

## Chroma süreç modeli

Kurulu paket: Chroma 1.3.5. `PersistentClient` aynı süreçte iş parçacıkları arasında kullanılabilir; aynı dizini yazan ikinci bir süreç güvenli değildir. Bu servis Uvicorn'u `--workers 1` ile çalıştırır. İndeks işçisi aynı süreçteki bir iş parçacığıdır. Ayrı bir worker prosesi veya birden fazla Uvicorn worker'ı bu dizini paylaşmamalıdır. Çok süreç gerekirse Chroma HTTP sunucusu ayrıca kurulmalıdır; Docker volume paylaşımı bunu kendiliğinden güvenli yapmaz.

Yeni vektörler eski `bookstack_articles` koleksiyonuna yazılmaz. Şema sürümü `pc-v1`, indeks sürümü `adaptive-1`.

## Tokenizer

Chroma'nın varsayılan gömme fonksiyonu `ONNXMiniLM_L6_V2` / `all-MiniLM-L6-v2` kullanır ve diziyi 256 WordPiece tokenında keser. Bu sınır, kurulu `chromadb/utils/embedding_functions/onnx_mini_lm_l6_v2.py` dosyasında `tokenizer.enable_truncation(max_length=256)` satırı ile doğrulanmıştır. `DefaultEmbeddingFunction.max_tokens()` 256 döner.

Yeni child parçalar gömme metnini varsayılan 180 tahmini tokenın altında tutar. Hiyerarşi başlığı bu bütçenin en fazla beşte birini alabilir; sığmazsa gömülmez. Tahmin `max(kelime * 1.4, karakter / 3)` formülüdür ve WordPiece sayımı değildir. Sağlayıcı `usage` döndürürse gerçek prompt/çıktı tokenları tahminle birlikte `usage_log` tablosuna yazılır.

## İndeks durumu

SQLite WAL dosyası `rag_state.sqlite` iş kuyruğu, sayfa durumu, revizyon, lexical FTS5 ve katalogu tutar. Chroma ile ortak bir ACID işlemi yoktur. Sıra `prepared → indexed → published` şeklindedir. Yayın işaretçisi dönmeden süreç kapanırsa sorgu eski aktif revizyonu kullanır. `prepared` kayıt kurtarmada silinir. Yazmaları bitmiş `indexed` kayıt kurtarmada yayınlanır. Eski vektör kimlikleri `vector_gc` üzerinden silinir. Sorgu, adayın revizyonunu aktif revizyonla karşılaştırır.

Aynı içerik ve metadata karması embedding çağırmaz. Sayfa adı, kitap adı ve bölüm adı gömme metninin parçasıdır (`Kitap › Bölüm › Sayfa` öneki); bunlardan biri değişirse sayfa yeniden parçalanır ve gömülür. Yalnız raf, etiket veya URL değişirse metadata ve FTS `location` sütunu yerinde güncellenir. Bir kitabın bütün rafları `shelf_names` listesinde durur; ilk raf tek gerçek raf sayılmaz.

Hiyerarşi değişiklikleri: `book_update`/`book_delete` → `book_refresh` (yalnız raf değiştiyse yerinde etiketleme; ad değiştiyse veya kitap silindiyse kitabın sayfaları yeniden kuyruğa alınır). `chapter_*` → `chapter_refresh` (bölümde olan ve olmuş sayfalar yeniden kuyruğa alınır; önbellekteki bölüm/kitap adı düşürülür). `page_move` bir sayfa güncellemesi gibi işlenir. Kitap, bölüm ve raf işleri kuyrukta negatif ve ayrı aralıklı anahtar kullanır; böylece aynı sayısal kimlikli bir sayfa işini geçersiz kılmaz.

FTS tablosu `body`, `title` ve `location` (raflar, kitap, bölüm) sütunlarını tutar; bm25 ağırlıkları 1.0 / 1.5 / 1.0'dır. FTS satırları `chunk_records.fts_rowid` üzerinden silinir; `page_id` sütunu indekslenmediği için `WHERE page_id` silmesi bütün tabloyu tarar ve indekslemeyi karesel yavaşlatırdı. Eski şemalı bir veritabanı açılışta FTS'i `chunk_records` üzerinden yeniden kurar.

Silinen sayfa tombstone olur. Daha eski bir iş onu geri getiremez. Tam uzlaştırma, tarama hatasız bitmeden eksik sayfaları silmez.

## Retrieval

Selamlama, katalog, aktif sayfa özeti ve arama önce deterministik ayrılır. Planlayıcı model varsayılan kapalıdır. Açılırsa yalnız soru, kısa geçmiş ve şema doğrulaması görür; katalog görmez. Hatalı plan tek soruya döner. Kitap ipuçları ACL filtresi yapılmaz.

Vektör ve FTS5 ayrı kanaldır. Her kanal `limit`'in iki katı aday getirir; birleştirme reciprocal rank fusion ile yapılır; cosine mesafesi ile BM25 toplanmaz. Her kanalın ilk iki adayı sonuçta tutulur, böylece bir kod veya ad gibi kesin FTS eşleşmesi iki kanalda da orta sıradaki adaylar tarafından dışarı itilmez. Reranker varsayılan kapalıdır ve metrik `fallback_rrf` yazar. İzin listesi FTS'e ve katalog sorgularına tek JSON parametresi (`json_each`) olarak gider. Chroma'da izinli sayfa sayısı `ACL_FILTER_BATCH` (200) altındaysa tek `$in` filtresi kullanılır; daha büyük kümelerde filtresiz daha geniş bir aday listesi çekilip izne göre süzülür, yetmezse tek bir `$in` sorgusuna düşülür. Arama sorguları iş parçacığı başına salt okunur bir SQLite bağlantısı kullanır (WAL okuyucuları yazarı beklemez). Parent metni token bütçesini aşıyorsa tamamı yüklenmez. Kaynak listesi seçilen kanıtın sayfa kimliğinden kurulur.

Pasajlar `book_name`, `chapter_name` ve `shelf_names` taşır; kanıt hakemi her pasajın konumunu (`raf › kitap › bölüm › sayfa`) görür ve soruda adı geçen konumu tercih eder. `document_search` isteğe bağlı `shelf`, `book`, `chapter` alır: ad kullanıcının erişebildiği kaplarla eşleşirse arama o kapların sayfalarıyla sınırlanır, eşleşmezse sınırlama yapılmaz ve `container.status` bunu bildirir.

Katalog: `catalog_counts` raf/kitap/bölüm/sayfa sayılarını, `catalog_browse` raf, kitap, bölüm veya sayfa listesini (sayfalı, `has_more` ile) döner; ikisi de raf/kitap/bölüm filtresi alır. Ad eşleştirme büyük/küçük harf ve Türkçe karakter farkını yok sayar; sorgunun her kelimesi addaki bir kelimeyle (önek) eşleşmelidir. Kısmi eşleşme öneri olarak döner, sessizce başka bir kaba çevrilmez. Adlar yalnız kullanıcının erişebildiği sayfaların kaplarından çözülür; gizli bir raf, kitap veya bölüm adı öneride de görünmez.

Durma: kanıt tamam, yeni tur kanıt getirmez, iki ek tur biter veya süre/çağrı bütçesi dolar. Bütçe bitmesi kanıtın yeterli olduğu anlamına gelmez. Büyük yanıt modeli tur başına tekrar çağrılmaz. İlk sürümde yanıt önbelleği yoktur.

## Görseller

İndirme boyut, süre ve host ile sınırlıdır. BookStack `Authorization` başlığı yalnız BookStack hostuna gider; yönlendirme hedefi dış host ise başlık taşınmaz. Önbellek anahtarı içerik karması, vision modeli ve prompt sürümüdür. Aynı URL yeni bayt veya yeni prompt ile yeniden analiz edilir.
