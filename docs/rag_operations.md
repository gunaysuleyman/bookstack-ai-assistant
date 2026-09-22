# Adaptive RAG işletim notu

## Bayraklar

| Değişken | Anlamı |
| --- | --- |
| `ADAPTIVE_RAG=off` | Kullanıcı eski retrieval yolunu görür. Yeni koleksiyon açılmaz. |
| `ADAPTIVE_RAG=shadow` | Yanıt eski yoldan gelir. Yeni retrieval ayrıca çalışır, yanıt modeli ikinci kez çağrılmaz. |
| `ADAPTIVE_RAG=on` | Yanıt yeni motordan gelir. Bunun için `bookstack_articles_pc_v1` doldurulmuş olmalıdır. |
| `ADAPTIVE_INDEXING=1` | Webhook işleri yeni revizyon yayıncısına gider. |
| `ENABLE_INDEX_WORKER=1` | Kuyruk aynı süreçte işlenir. |

Canlı kesim bu depoda yapılmadı. Değerlendirme belgesindeki ölçülmemiş kapılar dururken bayrağı `on` veya `shadow` yapmayın. `ADAPTIVE_RAG` varsayılanı `off` kalır.

`WEBHOOK_SECRET` boşken yeni imaj webhook isteklerini reddeder. Konteyneri yeniden oluşturmadan önce bu sırrı ve `RAG_SECRET_TOKEN` değerini varsayılan belgedeki örnekten farklı bir sırra çevirin. Çalışan eski imaj bu not yüzünden yeniden başlatılmamalıdır.

## Uçlar

- `GET /health` süreç ayaktadır.
- `GET /ready` aktif koleksiyon, şema ve indeks sürümünü gösterir. İndeksin senkron sağlığı bu uçta iddia edilmez.
- `GET /api/jobs/status` servis sırrı ister. Kuyruk, lease, dead-letter ve tamamlanan sayıları döner.
- `POST /api/sync` tam uzlaştırmayı kuyruğa yazar ve hemen döner. BookStack'e bu istek sırasında gitmez.
- `POST /api/webhook` `?token=` veya `X-Webhook-Token` ile `WEBHOOK_SECRET` bekler. BookStack gövde imzası göndermez. Ağ sınırı kullanıcı yetkisinin yerine geçmez.

Webhook URL örneği:

`http://rag_service:8000/api/webhook?token=WEBHOOK_SECRET`

Olaylar `page_create`, `page_update`, `page_delete` ve `book_update` kabul edilir. Aynı sayfanın bekleyen eski işi `superseded` olur. Gövde yalnız sayfa kimliği taşır. Adaptive indeksleme açıkken iş sayfayı BookStack'ten okur ve yeni koleksiyonu da günceller. Okuma hatası işi başarılı saymaz; kuyruk yeniden dener. Bir okuma hatası varken tam uzlaştırma eksik sayfaları silmez. Durum veritabanındaki işlemler tek kilit altında sıralanır. Süresi dolan `scope_ref` kayıtları okunurken ve worker turunda silinir. Tam uzlaştırmada eski koleksiyon yazısı yeni indeks yayınından önce yapılır; eski yazı hata verirse sayfa değişmiş görünmeye devam eder ve iş yeniden dener. Eski koleksiyondan silme hatası da işi başarısız sayar. Gemini anahtarı istek adresine konmaz. Yanıt modeli hata verirse adaptive arama, bulunan parçaları döndürür. `MAX_MODEL_CALLS` her HTTP denemesini sayar.

## Kuyruk

SQLite dosyası `RAG_STATE_DIR/rag_state.sqlite`. Lease süresi `JOB_LEASE_SECONDS` (varsayılan 120). En fazla `JOB_MAX_ATTEMPTS` (varsayılan 5) denemeden sonra kayıt `dead` olur. Uzun HTTP veya model çağrısı sırasında yazma işlemi açık tutulmaz.

Yerel komutlar, `rag_service` dizininden:

```text
python -m adaptive.cli status
python -m adaptive.cli manifest
python -m adaptive.cli recover
python -m adaptive.cli retry-dead JOB_ID
```

`recover` yarım `prepared` revizyonu siler, yazması bitmiş `indexed` revizyonu yayınlar ve `vector_gc` kimliklerini Chroma'dan temizler. `retry-dead` bir dead-letter işini yeniden kuyruğa alır.

## Yeniden indeks ve geri dönüş

Yeni indeks eski koleksiyonun üstüne yazılmaz. Geri dönüş: `ADAPTIVE_RAG=off` ve `ADAPTIVE_INDEXING=0` yapıp RAG konteynerini yeniden başlatın. Eski `bookstack_articles` koleksiyonu yerinde kalır.

Yeni indeksi doldurmak üretim verisine karşı bu oturumda çalıştırılmadı. Operatör kendi bakım penceresinde, yedeği aldıktan sonra webhook veya kontrollü uzlaştırma ile doldurmalıdır. Başlangıçta otomatik full sync yoktur.

## Yedek ve geri yükleme

Birlikte kopyalayın:

- Chroma dizini (`CHROMA_PERSIST_DIR`)
- `rag_state.sqlite` ve WAL/SHM dosyaları

Geri yüklemeden önce servisi durdurun. Tek süreç sahip olduğu için kopya, çalışan yazıcı varken tutarlı sayılmaz. Testler, kapatılmış SQLite dosyasının kopyasından aktif revizyon metninin okunabildiğini doğruladı. Tam Chroma dosya kopyası bu oturumda üretim volume'u üzerinde denenmedi.

Sır değiştirmek: `RAG_SECRET_TOKEN` ve `WEBHOOK_SECRET` değerlerini BookStack ve RAG ortamında birlikte değiştirin, konteynerleri yeniden başlatın. Eski imzalı sayfa token'ları TTL sonunda zaten reddedilir. Tarayıcı kaynağında sır yoktur; buna rağmen sızıntı şüphesinde her iki sırrı da değiştirin.

## Test komutları

`rag_service` dizininde, ağa ve üretim volume'una gerek olmadan:

```text
python -m pytest -q
```

Küçük kapasite dumanı, varsayılan 40 sayfa, hash gömme:

```text
python benchmarks/capacity_smoke.py --pages 40
```

10.000 sayfalık koşu bu komutun varsayılanı değildir ve bu teslimde çalıştırılmamıştır.
