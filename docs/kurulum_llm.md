# BookStack AI Assistant kurulum notu

Bu dosya yeni bir makinede, özellikle Proxmox LXC içinde, kurulumu yapacak model içindir. Yerel geliştirme makinesinin `.env` dosyasını, Chroma volume'unu veya BookStack volume'unu kopyalama. Sır değerlerini bu dosyaya yazma ve sohbet çıktısına basma.

## Ne kuruluyor

Üç konteyner `docker-compose.yml` ile kalkar:

- `bookstack`: wiki arayüzü, host portu `6875`.
- `bookstack_db`: MariaDB. Veri volume'da kalır.
- `rag_service`: FastAPI. Host tarafında yalnız `127.0.0.1:8000`.

Kod `./rag_service` dizininden konteynere bağlanır. İmajı yeniden derlemek, diskteki kaynak klasörün yerine geçmez. Kurulum, bu deponun güncel çalışma ağacıyla yapılmalıdır. `rag_service/adaptive/tools.py` ve `rag_service/adaptive/provider.py` içinde `https://api.openai.com/v1/responses` yoksa klon eski commit'tedir; o ağaçla yayına alma.

## Model ayrımı

- Cevap modeli OpenAI Responses API: `AI_PROVIDER=openai`, `OPENAI_MODEL=gpt-6-luna`, akıl yürütme `medium`.
- Gömme modeli Gemini'de kalır: `EMBEDDING_MODEL_ID=gemini-embedding-001`. Bunu değiştirmek yeni bir `ADAPTIVE_COLLECTION` adı ve kontrollü tam indeks ister. Mevcut koleksiyonun üstüne yazma.
- Görsel açıklama indeks sırasında Gemini vision ile yapılır. Cevap anında görsel çağrısı yoktur.
- `gpt-6-luna` fiyatı, kayıt anında satıra yazılır: giriş 0,10 USD / 1 milyon token, çıkış 0,50 USD / 1 milyon token. Çıkış tokenine akıl yürütme dahildir. Gemini gömme bu dolar tutarının dışındadır.

## Yapma

- `.env` dosyasını git'e ekleme.
- `RAG_SECRET_TOKEN` için `my_super_secret_local_token_123` kullanma. Servis bu değerle açılmaz.
- `ADAPTIVE_INDEXING=1` iken `WEBHOOK_SECRET` boş bırakma. Servis açılmaz.
- RAG portunu `0.0.0.0` üzerinde yayınlama. Tarayıcı erişimi ters vekil üzerinden, `RAG_SERVICE_PUBLIC_URL` ile verilir.
- BookStack API anahtarını `IMAGE_TRUSTED_HOSTS` içindeki hostlara gönderme. Kod bunu BookStack dışına iletmez; genel dış indirmeyi `IMAGE_ALLOW_EXTERNAL=1` yaparak açma.
- İlk açılışta otomatik tam indeks yoktur. İçerik, token'lar hazır olduktan sonra `POST /api/sync` ile kuyruğa alınır.
- Çalışan bir BookStack volume'unda `APP_KEY` veya yalnız `.env` içinden `DB_PASS` değiştirme. Ayrıntı `docs/rag_operations.md` içindedir.

## Proxmox CT

CT içinde Docker çalışacaksa nesting ve gerekli aygıt izinleri açık olmalıdır. Repoyu CT içine klonla. Wiki ve RAG'ı kullanıcının tarayıcısına ancak CT önündeki ters vekil açar.

Örnek genel adresler, kurulumda gerçek alan adlarıyla değiştirilir:

- Wiki: `https://wiki.ornek.tld`
- RAG arama: `https://wiki.ornek.tld/api/ai-search` veya ayrı bir host. Bu adres tarayıcıdan erişilebilir olmalı ve `RAG_SERVICE_PUBLIC_URL` ile aynı olmalıdır.
- Vekil, RAG'a `127.0.0.1:8000` üzerinden bağlanır.

## `.env`

`.env.example` dosyasını kopyala. Değerleri üret; örnek metinleri bırakma. Aşağıdaki blok bu kurulumun hedefidir.

```env
APP_KEY=<yeni base64 anahtar>
DB_PASS=<yeni veritabanı şifresi>
DB_PASSWORD=<DB_PASS ile aynı>
DB_NAME=bookstackapp
DB_DATABASE=bookstackapp
DB_USER=bookstack
DB_USERNAME=bookstack

BOOKSTACK_EXTERNAL_URL=https://wiki.ornek.tld
BOOKSTACK_TOKEN_ID=<BookStack API token id>
BOOKSTACK_TOKEN_SECRET=<BookStack API token secret>

RAG_SECRET_TOKEN=<uzun benzersiz sır>
RAG_INTERNAL_URL=http://rag_service:8000
RAG_SERVICE_PUBLIC_URL=https://wiki.ornek.tld/api/ai-search
RAG_BIND_HOST=127.0.0.1
WEBHOOK_SECRET=<uzun benzersiz sır>
TOKEN_TTL_SECONDS=900

ADAPTIVE_RAG=on
ADAPTIVE_INDEXING=1
ENABLE_INDEX_WORKER=1
ADAPTIVE_TOOLS=1
MAX_MODEL_CALLS=3
EMBEDDING_MODEL_ID=gemini-embedding-001
ADAPTIVE_COLLECTION=bookstack_articles_pc_v1

IMAGE_TRUSTED_HOSTS=
IMAGE_ALLOW_EXTERNAL=0

AI_PROVIDER=openai
GEMINI_API_KEY=<gemini anahtarı>
GEMINI_MODEL=gemini-3.8-flash
GEMINI_VISION_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-flash-latest,gemini-3.6-flash

OPENAI_API_KEY=<openai anahtarı>
OPENAI_MODEL=gpt-6-luna
RESONING_OPENAI_MODEL=medium

AI_ALLOWED_ROLES=admin,internal
```

`RESONING_OPENAI_MODEL` yazımı kodun okuduğu isimdir. `REASONING_OPENAI_MODEL` de geçerlidir. İkisi de doluysa `REASONING_OPENAI_MODEL` kullanılır.

Görseller BookStack dışı bir hosttaysa yalnız o hostu `IMAGE_TRUSTED_HOSTS` içine virgülle yaz. Örnek ihtiyaç: `odoo.capstan.be`.

## Sıra

1. CT'de depoyu al, `.env` dosyasını yukarıdaki kuralla doldur.
2. `docker compose up -d --build` çalıştır.
3. `http://127.0.0.1:8000/health` cevabında `provider` alanı `openai` olmalı. `GET /ready` 200 dönmeli. 503 ise worker, webhook sırrı, servis sırrı veya indeks erişimini logdan ayır.
4. Wiki'ye genel URL'den gir. İlk BookStack kurulumunda varsayılan yönetici parolasını hemen değiştir.
5. Ayarlar → Kullanıcılar → API Tokens ile token üret. `BOOKSTACK_TOKEN_ID` ve `BOOKSTACK_TOKEN_SECRET` yaz. Sonra yalnız RAG konteynerini yeniden oluştur: `docker compose up -d --no-deps --force-recreate rag_service`. `docker restart` ortam değişkenlerini `.env` dosyasından tekrar okumaz.
6. BookStack webhook:
   - URL: `http://rag_service:8000/api/webhook?token=<WEBHOOK_SECRET>`
   - Olaylar: sayfa oluşturma, güncelleme, silme. Kitap güncellemesi de desteklenir.
   - BookStack gövdeyi imzalamaz. Sır, URL'deki `token` ile aynı olmalıdır.
7. İlk indeks, RAG'ın dinlediği arayüzden:

```bash
curl -X POST http://127.0.0.1:8000/api/sync \
  -H "X-RAG-Token: <RAG_SECRET_TOKEN>"
```

8. `GET /api/jobs/status` aynı başlıkla kuyruğu gösterir. İş `done` olunca ve `dead` sayısı 0 iken widget'tan bir sayfa sorusu sor. Kaynaklar yetkili sayfalardan gelmeli.

Widget dosyası BookStack temasına volume ile bağlıdır: `widget/bookstack_ai_widget.html`. BookStack imajını sırf widget için yükseltme.

## Doğrulama

- Açık sayfada "summarize this page" yalnız o sayfayı kaynak gösterir.
- Katalog sorusu (`kaç kitap`) `catalog_counts` aracını kullanır; uydurma sayı yazılmaz.
- Her başarılı asistan turu `rag_state.sqlite` içindeki `assistant_turns` tablosuna kullanıcı kimliği, soru, cevap, model, token ve dolar olarak yazılır. Kimlik, BookStack kullanıcı numarasıdır.
- Gömme anahtarı istek URL'sine konmaz. OpenAI anahtarı `Authorization` başlığındadır.

## Geri dönüş

Yeni cevap yolunu kapatmak için `ADAPTIVE_RAG=off` ve `ADAPTIVE_INDEXING=0` yap, RAG konteynerini `.env` okunacak şekilde yeniden oluştur. Eski koleksiyon `bookstack_articles` yerinde kalır. Ayrıntılı kuyruk, yedek ve sır değiştirme notu `docs/rag_operations.md` dosyasındadır.
