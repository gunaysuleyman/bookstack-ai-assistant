# Ölçek ve hiyerarşi stres testi

Tarih: 30 Eylül 2026. Ortam: Linux, 4 çekirdek, Python 3.11, Chroma 1.3.5. Ücretli model çağrısı yapılmadı. Üretim Chroma dizini ve BookStack verisi kullanılmadı. Bütün ölçümler geçici dizinde, gerçek `Indexer` ve gerçek `document_search` / `catalog_*` araçları (`ToolRegistry` + `HybridSearcher`) üzerinden alındı.

Tekrar üretmek için:

```bash
cd rag_service
python benchmarks/stress_scale.py --preset quick                 # ~1 dk, 500 sayfa
python benchmarks/stress_scale.py --preset full --report out.md  # ~15 dk, 10.000 sayfa
python -m pytest -q tests/test_hierarchy_scale.py                # birim testleri
```

## Korpus

Sentetik ama BookStack yapısında: 60 raf, 1.000 kitap, 1.941 bölüm (chapter), 10.000 sayfa, ~100.000 chunk. Kitapların %10'u iki rafta, %3'ü hiçbir rafta değil; sayfaların ~%80'i bir bölümde.

Her sayfada 5 başlık × 70 kelime ve **aynı cümle kalıbında** tek bir ayırt edici bilgi var (`<KOD> sisteminin yedekleme sorumlusu <kişi>`). Yani 10.000 sayfanın hepsi soruyla aynı kelimeleri taşıyor; doğru sayfayı yalnız kod ayırıyor.

Raf, kitap, bölüm ve sayfa başlığı adları sayfa gövdesinde **hiç geçmiyor**, dolayısıyla bu adlarla gelen bir isabet yalnız hiyerarşi bilgisinden gelebilir. Hiyerarşi testinde aynı bilgi ("seyahat avansı onay limiti") dört farklı bölüm, kitap ve raftaki dört sayfada farklı değerlerle duruyor; doğru sayfayı yalnız soruda adı geçen kap ayırabilir.

Gömme: `hash-bow` (deterministik, anlamsal değil). "Yalnız vektör" sütunu ve top-1 sıralaması bu yüzden anlamsal kaliteyi ölçmez. Bulgular gömmeden bağımsız mekaniği ölçer: indeks, FTS, ACL, katalog ve hiyerarşi. Üretimdeki Gemini gömmesiyle anlamsal kalite ayrıca ölçülmelidir.

## A. 1.000 kitap / 10.000 sayfada ölçek

| Ölçüm (10.000 sayfa) | Önce | Sonra |
|---|---|---|
| İndeksleme, sayfa başına (embedding hariç) | 65 ms → **230 ms** (büyüdükçe artıyor) | 54 ms → **62 ms** (sabit) |
| Toplam indeksleme | 24,1 dk | **9,8 dk** |
| Dolu indekste tek sayfa güncelleme (p50) | 251 ms | **75 ms** |
| Recall@8 (200 kod sorusu) | 0,99 | **1,00** |
| Arama p50 / p95 / p99 (admin) | 66 / 131 / 154 ms | **28 / 36 / 39 ms** |
| Kısıtlı kullanıcı, 1.000 izinli sayfa (p50) | 455 ms | **80 ms** |
| Kısıtlı kullanıcı, 3.000 izinli sayfa (p50) | 1.281 ms | **62 ms** |
| Kısıtlı kullanıcı, 8.000 izinli sayfa (p50) | 3.049 ms | **68 ms** |
| Kısıtlı kullanıcı, 50–200 izinli sayfa (p50) | 109–114 ms | 94–98 ms |
| 10 eşzamanlı sorgu: sorgu/s, p50 | 13,6 /s, 694 ms | **97 /s, 80 ms** |
| İzinsiz sayfa sızıntısı | 0 | 0 |
| Katalog sayıları (sayfa/kitap/raf) | doğru | doğru (+ bölüm sayısı) |

Yapılan düzeltmeler:

1. **Karesel indeksleme.** `page_id` FTS5'te indekslenmeyen bir sütun; `DELETE FROM chunks_fts WHERE page_id = ?` her yayında 100.000 satırın tamamını tarıyordu (tek tarama 41 ms, bir upsert'in 165/310 ms'si). FTS satırları artık `chunk_records.fts_rowid` üzerinden rowid ile siliniyor. `chunk_records` ve `parent_records` tablolarına `(page_id, revision_id)` ve `revision_id` indeksleri eklendi.
2. **ACL partileri.** İzin listesi FTS'e ve katalog sorgularına tek JSON parametresi (`json_each`) olarak gidiyor; önceden 200 sayfalık her parti için ayrı bir FTS ve ayrı bir Chroma sorgusu çalışıyordu. Chroma'da izinli küme 200 sayfadan büyükse filtresiz, daha geniş bir aday listesi çekilip izne göre süzülüyor (filtreli `$in` 8.000 sayfada 835 ms, filtresiz 200 aday 36 ms). Yetmezse tek bir `$in` sorgusuna düşülüyor. 50–200 sayfalık küçük kümeler hâlâ Chroma'nın filtreli sorgusunu kullanıyor (~95 ms); bu Chroma'nın `$in` maliyeti.
3. **Arama sorguları.** Her aramada 10.000 satırlık `active_revision_map` okuması ve aday başına `get_page_state` sorgusu kalktı; adaylar tek bir birleştirme sorgusuyla geliyor. Birleştirme sırası sabitlendi, çünkü yeni indeks planlayıcıyı `page_state` taramasına yönlendirip sorguyu 0,1 ms'den 50 ms'ye çıkarıyordu. Admin aramasında FTS önce kendi içinde sıralanıyor, sonra yalnız ilk adaylar birleştiriliyor. Chroma'daki sorgu başına `count()` çağrısı kaldırıldı.
4. **Eşzamanlılık.** Arama sorguları iş parçacığı başına salt okunur bir SQLite bağlantısı kullanıyor; WAL okuyucuları yazar kilidini beklemiyor.
5. **Kaybolan kesin eşleşmeler.** Parçalayıcı cümleyi kelime ortasından bölüyordu; kod bir chunk'ta, "yedekleme sorumlusu" sonrakinde kalıyordu. Artık cümle sınırında bölüyor. Yalnız sınırı aşan tek bir cümle kelime düzeyinde kesiliyor ve kalan kısa kuyruk sonraki cümleyle birleşiyor. RRF sonrası her kanalın ilk iki adayı sonuçta tutuluyor, böylece kesin bir FTS eşleşmesi iki kanalda da orta sıradaki adaylarca dışarı itilmiyor.

## B. RAG rafları ve bölümleri algılıyor mu?

| Test (10.000 sayfa) | Önce | Sonra |
|---|---|---|
| Aynı bilgi 4 sayfada; soru **bölüm** adını veriyor → doğru sayfa ilk 8'de | %23 | **%100** |
| … ve model adı `chapter` filtresi olarak da veriyor → doğru sayfa 1. sırada | — | **%96** |
| Aynı test, **kitap** adıyla (filtreli) → 1. sırada | %0 | **%92** (ilk 8'de %100) |
| Aynı test, **raf** adıyla (filtreli) → 1. sırada | %2 | %49 (ilk 8'de %100) |
| "`<sayfa başlığı>` sayfasında ne anlatılıyor?" → sayfa ilk 8'de | %0 | **%99,5** |
| Raf listesi | araç yok | **60/60, tam** |
| Bir raftaki kitaplar / bir kitaptaki bölümler / bir kitaptaki ve bölümdeki sayfalar | araç yok | **%100 tam** (küçük harf, Türkçe karaktersiz adla da) |
| Bölüm sayısı | dönmüyordu | 1.835/1.835 |
| Pasajlarda raf/kitap/bölüm | yok | var; kanıt hakemi konumu görüyor |
| Kısıtlı kullanıcıya gizli raf adının sızması | — | sızmıyor (listede de, öneride de) |

Raf filtresinde top-1'in %49'da kalmasının nedeni şu: filtre aramayı o rafın ~170 sayfasına daraltıyor ve doğru sayfa her zaman ilk 8'de. İlk sırayı ise anlamsal olmayan `hash-bow` vektör kanalı karıştırıyor. Kanıt hakemi pasajları konumlarıyla gördüğü için ilk 8'de olmak yeterli; gerçek gömmeyle top-1 ayrıca ölçülmeli.

Yapılan değişiklikler:

1. **Hiyerarşi indekste.** FTS'e `title` ve `location` (raflar, kitap, bölüm) sütunları eklendi (bm25 ağırlıkları gövde 1,0 / başlık 1,5 / konum 1,0). Gömme metnine `Kitap › Bölüm › Sayfa` öneki eklendi. Raflar gömmeye girmiyor, çünkü bir kitap birden çok rafta olabilir ve raf üyeliği sık değişir. Chroma metadatasına `book_id`, `chapter_id` ve `chapter_name` eklendi. Şema `pc-v3`.
2. **Pasajlarda konum.** `book_name`, `chapter_name` ve `shelf_names` yanıt modeline gidiyor. Kanıt hakemi her pasajı `raf › kitap › bölüm › sayfa` konumuyla görüyor ve soruda adı geçen konumu tercih ediyor.
3. **Katalog araçları.** `catalog_browse` rafları, kitapları, bölümleri veya sayfaları sayfalı olarak (`has_more` ile) listeliyor; raf/kitap/bölüm filtresi alıyor. `catalog_counts` bölüm sayısını da veriyor ve filtre alıyor.
4. **Filtreli arama.** `document_search` isteğe bağlı `shelf`, `book` ve `chapter` alıyor. Ad eşleşirse arama o kapla sınırlanıyor; eşleşmezse sınırlama yapılmıyor ve durum `container.status` alanında bildiriliyor. Takip araması da aynı kapla sınırlı kalıyor.
5. **Ad eşleştirme.** Büyük/küçük harf ve Türkçe karakter farkı yok sayılıyor ("bt el kitabi" → "BT El Kitabı"). Sorgunun her kelimesi addaki bir kelimeyle eşleşmeli. Kısmi eşleşme ("Finans Bursa Rafı" → "Finans Ankara Rafı") sessizce uygulanmıyor; öneri olarak dönüyor, model de kullanıcıya soruyor. Adlar yalnız kullanıcının erişebildiği sayfaların kaplarından çözülüyor.
6. **Değişiklik takibi.** Webhook artık `page_move`, `chapter_*`, `bookshelf_*` ve `book_delete` olaylarını işliyor. Sayfa, kitap veya bölüm adı değişirse sayfa yeniden gömülüyor; yalnız raf, etiket veya URL değişirse etiketler yerinde güncelleniyor. Kitap, bölüm ve raf işleri kuyrukta ayrı negatif anahtar kullanıyor. Önceden `book_update` kitap kimliğini sayfa kimliği gibi kuyruğa yazıyordu, bu yüzden 5 numaralı kitabın güncellemesi 5 numaralı sayfanın bekleyen işini iptal edebiliyordu.

## Geçiş

- Şema `pc-v2` → `pc-v3`. İlk açılışta eski FTS tablosu yeni sütunlarla `chunk_records` üzerinden yeniden kuruluyor (10.000 sayfada birkaç saniye). Böylece yeniden indekslemeden önce de başlık ve konum aramada kullanılabiliyor.
- Gömmelerin güncellenmesi için bir kez tam uzlaştırma çalıştırın: `POST /api/sync`. Şema sürümü farklı her sayfa yeniden parçalanıp gömülüyor. Koleksiyon adı değişmiyor.
- BookStack'te webhook'u kitap, bölüm ve raf olaylarını da gönderecek şekilde genişletin (bkz. README).
- `CHILD_CHUNK_TOKENS` varsayılanı 160'tan 140'a indi; gömme bütçesi (180) içinde konum önekine yer açmak için.

## Bilinen sınırlar

- Sonuçlar sentetik korpus ve `hash-bow` ile alındı. Gemini gömmesiyle anlamsal recall, top-1 sıralaması ve gerçek embedding API süresi ayrıca ölçülmeli. 10.000 sayfanın ilk tam indekslemesinde süreyi Gemini çağrıları belirleyecek.
- 50–200 izinli sayfalı küçük kullanıcılar Chroma'nın filtreli sorgusunu kullanıyor (~95 ms).
- Raf adı değişip aynı anda rafın kitap listesi de değişirse, kitap artık o rafta olmadığı ve eski ad API'den okunamadığı için eski raf etiketi bir sonraki `book_update` olayına kadar kalabilir.
