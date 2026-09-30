# Ölçek ve hiyerarşi stres testi

Tarih: 30 Eylül 2026. Ortam: Linux, 4 çekirdek, Python 3.11, Chroma 1.3.5. Ücretli model çağrısı yapılmadı. Üretim Chroma dizini ve BookStack verisi kullanılmadı; bütün ölçümler geçici dizinde, gerçek `Indexer` ve gerçek `document_search` aracı (`ToolRegistry` + `HybridSearcher`) üzerinden alındı.

Tekrar üretmek için:

```bash
cd rag_service
python benchmarks/stress_scale.py --preset quick                 # ~40 sn, 500 sayfa
python benchmarks/stress_scale.py --preset full --report out.md  # ~35 dk, 10.000 sayfa
```

## Korpus

Sentetik ama BookStack yapısında: 60 raf, 1000 kitap, 1941 bölüm (chapter), 10.000 sayfa, 99.394 chunk. Kitapların %10'u iki rafta, %3'ü hiçbir rafta değil; sayfaların ~%80'i bir bölümde. Her sayfada 5 başlık × 70 kelime ve **aynı cümle kalıbında** tek bir ayırt edici bilgi var (`<KOD> sisteminin yedekleme sorumlusu <kişi>`). Yani 10.000 sayfanın hepsi soruyla aynı kelimeleri taşıyor; doğru sayfayı yalnız kod ayırıyor. Raf, kitap, bölüm ve sayfa başlığı adları sayfa gövdesinde **hiç geçmiyor**, böylece bu adlarla gelen bir isabet yalnız hiyerarşi metadatasından gelebilir.

Gömme: `hash-bow` (deterministik, anlamsal değil). Bu yüzden "yalnız vektör" sütunu anlamsal kaliteyi ölçmez; aşağıdaki bulgular gömmeden bağımsız olan indeks, FTS, ACL, katalog ve hiyerarşi mekaniği içindir. Üretimdeki Gemini gömmesiyle anlamsal kalite ayrıca ölçülmelidir.

## A. 1000 kitap / 10.000 sayfada RAG doğru çalışıyor mu?

### Sonuç tablosu

| Sayfa | Chunk | Recall@8 | Recall@1 | Yalnız FTS R@8 | Arama p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| 1.000 | 9.952 | 1.00 | 0.015 | 1.00 | 10 ms | 14 ms | 16 ms |
| 2.500 | 24.854 | 1.00 | 0.005 | 1.00 | 20 ms | 26 ms | 79 ms |
| 5.000 | 49.703 | 0.985 | 0.00 | 0.985 | 33 ms | 81 ms | 97 ms |
| 10.000 | 99.394 | 0.99 | 0.00 | 0.99 | 66 ms | 131 ms | 154 ms |

| İndekslenen sayfa | 1.000 | 2.000 | 4.000 | 6.000 | 8.000 | 10.000 |
|---|---|---|---|---|---|---|
| ms / sayfa (embedding hariç) | 65 | 80 | 116 | 152 | 188 | 230 |

| Kısıtlı kullanıcının izinli sayfası | 50 | 200 | 1.000 | 3.000 | 8.000 |
|---|---|---|---|---|---|
| ACL partisi (200'lük) | 1 | 1 | 5 | 15 | 40 |
| Arama p50 | 109 ms | 114 ms | 455 ms | 1.281 ms | 3.049 ms |
| İzinsiz sayfa sızıntısı | 0 | 0 | 0 | 0 | 0 |

Diğer ölçümler (10.000 sayfada):

- 10 eşzamanlı sorgu (admin): 13,6 sorgu/s, p50 694 ms, p95 1.204 ms, 0 hata.
- Dolu indekste içerik güncellemesi: upsert p50 251 ms; eski bilgi FTS'ten tamamen silindi, yeni bilgi 30/30 bulundu.
- Katalog: sayfa 10.000/10.000, kitap 1.000/1.000, raf 60/60 doğru; 3.000 sayfalık kısıtlı kullanıcıda da kitap/raf sayıları doğru.
- Depolama: SQLite 251 MB, Chroma 369 MB.

### Değerlendirme

**Doğruluk ve güvenlik ölçekte korunuyor.** İzinsiz sayfa sızıntısı hiçbir boyutta yok, katalog sayıları kesin, güncelleme sonrası eski revizyon geri gelmiyor. Doğru sayfa 10.000 sayfada da %99 oranla ilk 8 pasajda.

**Ama beş ölçek sorunu var:**

1. **İndeksleme karesel yavaşlıyor (O(N²)).** Sayfa başı süre 65 ms'den 230 ms'ye doğrusal artıyor; toplam 10.000 sayfa embedding hariç 24 dakika. Doğrulanmış ana şüpheli (toplam süredeki payı ayrıca ölçülmedi; Chroma HNSW büyümesi de katkı verebilir): `store.publish` ve `tombstone` içindeki `DELETE FROM chunks_fts WHERE page_id = ?`. `page_id` FTS5 tablosunda `UNINDEXED` bir sütun; SQLite bunu indeksle bulamaz ve her yayında 100.000 satırlık FTS tablosunun tamamını tarar (`EXPLAIN QUERY PLAN` → `SCAN chunks_fts VIRTUAL TABLE INDEX 0`). Çözüm: FTS satırlarını `rowid` üzerinden silmek (chunk_records'a fts rowid saklamak) veya `DELETE ... WHERE rowid IN (SELECT ...)` ile normal tablodan eşlemek. İlk tam senkronda Gemini embedding süresi buna eklenir.

2. **Kısıtlı kullanıcıda arama süresi izinli sayfa sayısıyla doğrusal büyüyor.** İzin listesi 200'lük partilere bölünüyor ve her parti için ayrı bir Chroma sorgusu **ve** ayrı bir FTS sorgusu çalışıyor. 8.000 izinli sayfalı sıradan bir kullanıcı için 40+40 sorgu = 3 saniye; tek partide bile Chroma'nın `$in` filtreli sorgusu admin sorgusundan yavaş. 10.000 sayfalık bir kurulumda çoğu kullanıcı binlerce sayfa görür, yani bu **gerçek kullanıcı deneyimini belirleyen yol**. Çözüm önerisi: izinli küme büyükse (ör. > %30 veya > 1.000 sayfa) filtresiz daha geniş bir top-k çekip sonucu Python'da ACL ile süzmek; FTS'te ise izin listesini geçici tabloya yazıp tek sorguda JOIN etmek.

3. **Eşzamanlılık zayıf.** 10 eşzamanlı sorguda p50 66 ms'den 694 ms'ye çıkıyor. SQLite bağlantısı tek kilitle seri hale getiriliyor ve her arama `active_revision_map()` ile 10.000 satırın tamamını okuyor. Kısıtlı kullanıcılar ve gerçek embedding API gecikmesiyle bu daha da artar.

4. **Füzyon kesin eşleşmeyi aşağı itiyor (Recall@1 ≈ 0).** Reciprocal rank fusion'da FTS'te 1. sıradaki tek-kanal aday 1/61 puan alır; iki kanalda da 8. sırada olan zayıf bir aday 2/68 alır ve öne geçer. Hash gömme bu etkiyi abartıyor, fakat kodlar, sürüm numaraları, kişi adları gibi kesin eşleşme gereken sorularda Gemini ile de aynı mekanizma çalışır. 5.000 sayfadan sonra görülen %1–1,5'lik kayıp bu sınırda: doğru chunk ilk 8'in dışına itiliyor. Öneri: kanal ağırlıklı RRF veya FTS'te çok nadir bir terimle eşleşen chunk'a taban puan; ayrıca aday havuzunu `limit`'ten geniş çekip sonra kesmek.

5. **Her aramada `get_page_state` aday başına ayrı sorgu** — şu an küçük ama eşzamanlılıkta kilit süresini uzatıyor.

## B. RAG rafları ve bölümleri algılıyor mu?

**Kısa cevap: hayır.** Sayılar dışında hiyerarşi bilgisi aramaya ve yanıt modeline ulaşmıyor.

| Test (10.000 sayfa) | Sonuç | Rastgele tahmin |
|---|---|---|
| Aynı bilgi 4 sayfada farklı değerle; soru **bölüm** adını veriyor → doğru sayfa 1. sırada | %1,9 | %25 |
| Aynı test, **kitap** adıyla | %0 | %25 |
| Aynı test, **raf** adıyla | %1,9 | %25 |
| "`<sayfa başlığı>` sayfasında ne anlatılıyor?" → sayfa ilk 8'de | %0 | — |
| "`<raf>` içinde hangi kitaplar var?" → dönen sayfaların o rafta olma oranı | %2,2 | — |
| "`<kitap>` kitabında hangi sayfalar var?" | %0 | — |
| "`<bölüm>` bölümündeki sayfalar neler?" | %0 | — |

Nedenleri koddan doğrulandı:

1. **Hiyerarşi adları indekslenmiyor.** Gömme metni yalnız `başlık + chunk` (`chunking._embed_text`), FTS gövdesi yalnız `heading + body` (`store.write_fts`). Sayfa adı, kitap, bölüm ve raf adları ikisinde de yok. Chroma metadatasında `book_name` ve `shelf_name` var ama filtre olarak kullanılmıyor; `chapter_name` hiç yazılmıyor.
2. **Yanıt modeli hiyerarşiyi görmüyor.** `serialize_passages` yalnız `chunk_id, page_id, revision_id, title, url, heading, text` döndürüyor. Model "bu bilgi X rafındaki Y kitabının Z bölümünde" diyemez, aynı bilginin farklı bölümlerdeki sürümlerini ayırt edemez.
3. **Katalog araçları yalnız sayıyor.** `catalog_counts` sayfa/kitap/raf sayısını doğru veriyor, `catalog_list_books` kitapları listeliyor. Ama rafları listeleyen, bir raftaki kitapları, bir kitaptaki bölümleri/sayfaları veya bölüm sayısını veren araç yok. Bu sorular `document_search`'e düşüyor ve yukarıdaki tablo gibi rastgele sonuç veriyor.
4. **Bölüm değişiklikleri takip edilmiyor.** Webhook `chapter_update`/`chapter_move` olaylarını yok sayıyor; `bookshelf_update` da yok sayılıyor. Bölüm adı değişince sayfalar tam uzlaştırmaya kadar eski adı taşır.

## Önerilen düzeltmeler (öncelik sırasıyla)

1. FTS silmelerini `rowid` tabanlı yapmak (indekslemenin karesel yavaşlaması).
2. Büyük ACL kümelerinde parti başına sorgu yerine geniş sorgu + sonradan süzme; FTS'te tek sorgu.
3. Hiyerarşiyi indekse eklemek: FTS'e ayrı `title`, `book`, `chapter`, `shelf` sütunları (bm25 ağırlıklı), gömme metnine kısa bir `Kitap › Bölüm › Sayfa` önekini bütçe içinde eklemek, Chroma metadatasına `chapter_name`/`book_id`/`chapter_id` yazmak. Bu şema değişikliği yeni bir koleksiyon sürümü ve yeniden indeksleme gerektirir.
4. Pasajlara `book_name`, `chapter_name`, `shelf_names` eklemek, böylece yanıt modeli kaynağı konumuyla söyleyebilir.
5. Katalog araçlarını genişletmek: `catalog_list_shelves`, `catalog_list_books(shelf=…)`, `catalog_list_chapters(book=…)`, `catalog_list_pages(book|chapter=…)`, bölüm sayısı; ve `document_search`'e isteğe bağlı `book`/`chapter`/`shelf` kapsam filtresi (ACL ile kesişim).
6. RRF'yi kesin eşleşmelerin kaybolmayacağı şekilde ayarlamak ve aday havuzunu genişletmek.
7. `chapter_update` webhook'unu `book_refresh` benzeri bir yeniden etiketleme işine bağlamak.

Her düzeltmeden sonra aynı betik tekrar çalıştırılarak önce/sonra karşılaştırılabilir.
