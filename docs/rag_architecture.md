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

İzin değişikliği için güvenilir bir anlık bildirim yoktur. `bookshelf_update` olayı kitap listesini taşımaz; raf üyeliği `book_update` veya tam uzlaştırma ile yenilenir. Kabul edilen eskime süresi `TOKEN_TTL_SECONDS` (varsayılan 900 saniye) kadardır. Süre dolunca arama yapılmaz; kullanıcı sayfayı yenileyince yeni kapsam üretilir. Anında iptal garantisi yoktur.

## Chroma süreç modeli

Kurulu paket: Chroma 1.3.5. `PersistentClient` aynı süreçte iş parçacıkları arasında kullanılabilir; aynı dizini yazan ikinci bir süreç güvenli değildir. Bu servis Uvicorn'u `--workers 1` ile çalıştırır. İndeks işçisi aynı süreçteki bir iş parçacığıdır. Ayrı bir worker prosesi veya birden fazla Uvicorn worker'ı bu dizini paylaşmamalıdır. Çok süreç gerekirse Chroma HTTP sunucusu ayrıca kurulmalıdır; Docker volume paylaşımı bunu kendiliğinden güvenli yapmaz.

Yeni vektörler eski `bookstack_articles` koleksiyonuna yazılmaz. Şema sürümü `pc-v1`, indeks sürümü `adaptive-1`.

## Tokenizer

Chroma'nın varsayılan gömme fonksiyonu `ONNXMiniLM_L6_V2` / `all-MiniLM-L6-v2` kullanır ve diziyi 256 WordPiece tokenında keser. Bu sınır, kurulu `chromadb/utils/embedding_functions/onnx_mini_lm_l6_v2.py` dosyasında `tokenizer.enable_truncation(max_length=256)` satırı ile doğrulanmıştır. `DefaultEmbeddingFunction.max_tokens()` 256 döner.

Yeni child parçalar gömme metnini varsayılan 180 tahmini tokenın altında tutar. Hiyerarşi başlığı bu bütçenin en fazla beşte birini alabilir; sığmazsa gömülmez. Tahmin `max(kelime * 1.4, karakter / 3)` formülüdür ve WordPiece sayımı değildir. Sağlayıcı `usage` döndürürse gerçek prompt/çıktı tokenları tahminle birlikte `usage_log` tablosuna yazılır.

## İndeks durumu

SQLite WAL dosyası `rag_state.sqlite` iş kuyruğu, sayfa durumu, revizyon, lexical FTS5 ve katalogu tutar. Chroma ile ortak bir ACID işlemi yoktur. Sıra `prepared → indexed → published` şeklindedir. Yayın işaretçisi dönmeden süreç kapanırsa sorgu eski aktif revizyonu kullanır. `prepared` kayıt kurtarmada silinir. Yazmaları bitmiş `indexed` kayıt kurtarmada yayınlanır. Eski vektör kimlikleri `vector_gc` üzerinden silinir. Sorgu, adayın revizyonunu aktif revizyonla karşılaştırır.

Aynı içerik ve metadata karması embedding çağırmaz. Yalnız raf veya kitap adı değişirse metadata güncellenir. Bir kitabın bütün rafları `shelf_names` listesinde durur; ilk raf tek gerçek raf sayılmaz.

Silinen sayfa tombstone olur. Daha eski bir iş onu geri getiremez. Tam uzlaştırma, tarama hatasız bitmeden eksik sayfaları silmez.

## Retrieval

Selamlama, katalog, aktif sayfa özeti ve arama önce deterministik ayrılır. Planlayıcı model varsayılan kapalıdır. Açılırsa yalnız soru, kısa geçmiş ve şema doğrulaması görür; katalog görmez. Hatalı plan tek soruya döner. Kitap ipuçları ACL filtresi yapılmaz.

Vektör ve FTS5 ayrı kanaldır. Birleştirme reciprocal rank fusion ile yapılır; cosine mesafesi ile BM25 toplanmaz. Reranker varsayılan kapalıdır ve metrik `fallback_rrf` yazar. İzin listesi Chroma `$in` sınırına göre 200'lük parçalara bölünür. Parent metni token bütçesini aşıyorsa tamamı yüklenmez. Kaynak listesi seçilen kanıtın sayfa kimliğinden kurulur.

Durma: kanıt tamam, yeni tur kanıt getirmez, iki ek tur biter veya süre/çağrı bütçesi dolar. Bütçe bitmesi kanıtın yeterli olduğu anlamına gelmez. Büyük yanıt modeli tur başına tekrar çağrılmaz. İlk sürümde yanıt önbelleği yoktur.

## Görseller

İndirme boyut, süre ve host ile sınırlıdır. BookStack `Authorization` başlığı yalnız BookStack hostuna gider; yönlendirme hedefi dış host ise başlık taşınmaz. Önbellek anahtarı içerik karması, vision modeli ve prompt sürümüdür. Aynı URL yeni bayt veya yeni prompt ile yeniden analiz edilir.
