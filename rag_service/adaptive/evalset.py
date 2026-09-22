"""Labeled fixture for offline retrieval checks.

Each page has distinct facts. This set measures the adaptive pipeline on those
facts; it is not a production MiniLM semantic benchmark.
"""

from typing import Dict, List


def build_evalset() -> Dict[str, List[dict]]:
    pages = [
        _page(1, "VPN Kurulumu", "Ağ", "Erişim", "WireGuard kurulum adımları: istemciyi indirin, tüneli açın, bağlantıyı sınayın. VPN onaylayan birim: Bilgi Güvenliği."),
        _page(2, "VPN Onay Çelişkisi", "Ağ", "Erişim", "Eski taslak notu. VPN onaylayan birim: İnsan Kaynakları."),
        _page(3, "Yedekleme Hatası", "Sistem", "Hatalar", "Hata kodu ERR-4412 disk dolduğunda çıkar. İletişim Deniz Aksoy deniz@example.com."),
        _page(4, "New Hire Onboarding", "People", "Start", "New hire onboarding: collect your badge at Floor 3 from the reception desk."),
        _page(5, "Yıllık İzin", "İnsan", "Politika", "Yıllık izin hakkı 14 gündür. Başvuru İnsan Kaynakları birimine yapılır."),
        _page(6, "Güvenlik Duvarı", "Ağ", "Sınır", "Güvenlik duvarında dış erişim için port 443 ve port 8443 açıktır."),
        _page(7, "Yazıcı", "Ofis", "Donanım", "Depo B yazıcısının modeli HP-M404. Kağıt sıkışınca kapağı kapatın."),
        _page(8, "Parola Sıfırlama", "Kimlik", "Erişim", "Parola sıfırlama self-service portal üzerinden yapılır. Yöneticiye e-posta ile parola gönderilmez."),
        _page(9, "Olay Seviyesi", "Operasyon", "Olay", "SEV-1 olaylarda nöbet telefonu +90 212 555 0100 aranır."),
        _page(10, "Atlas Sahibi", "Projeler", "Atlas", "Proje Atlas sahibi Selin Karaca. E-posta adresi selin@example.com."),
        _page(11, "Atlas Bütçe", "Projeler", "Atlas", "Proje Atlas bütçe kodu ATL-19 olarak finans sistemine işlenir."),
        _page(12, "Misafir Ağ", "Ağ", "Kablosuz", "Misafir kablosuz ağının SSID değeri Misafir olarak yayınlanır."),
        _page(13, "Yedekleme Takvimi", "Sistem", "Yedek", "Backup schedule starts at 02:00 UTC and retention is 30 days."),
        _page(14, "Sertifika", "Sistem", "TLS", "TLS certificate renewal starts 30 days before expiry and is owned by the Platform team."),
        _page(15, "Dizüstü Talebi", "Donanım", "Talep", "Dizüstü talebi BT-12 formu ile açılır ve Satın Alma birimi onaylar."),
        _page(16, "API Limit", "Yazılım", "API", "API clients receive HTTP 429 when the retry-after header must be honored."),
        _page(17, "Toplantı Odası", "Ofis", "Mekan", "Oditoryum kapasitesi 40 kişidir. Rezervasyon resepsiyon üzerinden yapılır."),
        _page(18, "VPN İstemci Yolu", "Ağ", "Erişim", "WireGuard istemci paketi /tools/vpn-client yolundan indirilir."),
        _page(19, "Veri Sınıfları", "Güvenlik", "Politika", "Veri sınıfları Gizli, Hizmete Özel ve Genel olarak ayrılır."),
        _page(20, "Dağıtım Listesi", "Yazılım", "Sürüm", "Dağıtım kontrol listesi: sürümü etiketle, migrasyonu çalıştır, sağlık kontrolünü doğrula, duyuruyu yayınla."),
        _page(21, "Ücret Cetveli", "Yönetim", "Gizli", "SALARY-991 executive pay table is restricted to the board."),
        _page(22, "Satın Alma Planı", "Yönetim", "Gizli", "MERGER-228 acquisition plan remains inside the board shelf."),
        _page(23, "Mutfak Notu", "Ofis", "Genel", "Mutfak buzdolabı her cuma temizlenir. Yemek listesi bu sayfada yoktur."),
    ]
    questions: List[dict] = []

    def add(**kwargs):
        item = {
            "split": "calibration",
            "simple": False,
            "abstain": False,
            "conflict": False,
            "expected_page_ids": [],
            "forbidden_page_ids": [21, 22],
            "forbidden_terms": ["SALARY-991", "MERGER-228"],
            "current_page_id": None,
            "expect_extra_round": False,
        }
        item.update(kwargs)
        item["id"] = f"q{len(questions) + 1:02d}"
        questions.append(item)

    add(category="single_page", query="WireGuard kurulum adımları nelerdir?", expected_page_ids=[1], simple=True)
    add(category="single_page", query="Güvenlik duvarında hangi portlar açıktır?", expected_page_ids=[6], simple=True)
    add(category="single_page", query="Depo B yazıcı modeli nedir?", expected_page_ids=[7], simple=True)
    add(category="single_page", query="Parola sıfırlama self-service portal ile mi yapılır?", expected_page_ids=[8], simple=True)
    add(category="single_page", query="SEV-1 nöbet telefonu nedir?", expected_page_ids=[9], simple=True)
    add(category="single_page", query="Misafir kablosuz SSID değeri nedir?", expected_page_ids=[12], simple=True)
    add(category="single_page", query="Oditoryum kapasitesi kaç kişidir?", expected_page_ids=[17], simple=True)
    add(category="single_page", query="Veri sınıfları hangileridir?", expected_page_ids=[19], simple=True)
    add(category="single_page", query="Dizüstü talebi hangi formla açılır?", expected_page_ids=[15], simple=True)
    add(category="single_page", query="WireGuard istemci paketi hangi yoldan indirilir?", expected_page_ids=[18], simple=True)

    add(category="multi_page", query="Selin Karaca e-posta adresi ve Atlas bütçe kodu nedir?", expected_page_ids=[10, 11])
    add(category="multi_page", query="WireGuard kurulum adımları ve istemci paketinin yolu nedir?", expected_page_ids=[1, 18])
    add(category="multi_page", query="Backup schedule retention and TLS certificate renewal owner?", expected_page_ids=[13, 14])
    add(category="multi_page", query="SEV-1 telefonu ve ERR-4412 iletişim adresi nedir?", expected_page_ids=[9, 3])
    add(category="multi_page", query="HP-M404 modeli ve Oditoryum kapasitesi nedir?", expected_page_ids=[7, 17])
    add(category="multi_page", query="Yıllık izin gün sayısı ve BT-12 formunu kim onaylar?", expected_page_ids=[5, 15])
    add(category="multi_page", query="port 8443 ve SSID Misafir nerede geçer?", expected_page_ids=[6, 12])
    add(category="multi_page", query="Platform team renewal and HTTP 429 retry-after kuralı nedir?", expected_page_ids=[14, 16])

    add(category="cross_lingual", query="annual leave entitlement kaç gündür?", expected_page_ids=[5], expect_extra_round=True)
    add(category="cross_lingual", query="işe başlama yaka kartı hangi katta verilir?", expected_page_ids=[4], expect_extra_round=True)
    add(category="cross_lingual", query="backup retention kaç gündür?", expected_page_ids=[13])
    add(category="cross_lingual", query="password reset self-service portal kullanılıyor mu?", expected_page_ids=[8])
    add(category="cross_lingual", query="TLS certificate yenilemeyi hangi ekip yapar?", expected_page_ids=[14])
    add(category="cross_lingual", query="HTTP 429 alındığında retry-after ne anlama gelir?", expected_page_ids=[16])
    add(category="cross_lingual", query="Floor 3 badge which desk?", expected_page_ids=[4], simple=True)
    add(category="cross_lingual", query="retention 30 days backup ne zaman başlar?", expected_page_ids=[13])

    add(category="proper_name", query="ERR-4412 hatasında kime yazılır?", expected_page_ids=[3], simple=True)
    add(category="proper_name", query="Deniz Aksoy e-posta adresi nedir?", expected_page_ids=[3], simple=True)
    add(category="proper_name", query="Selin Karaca hangi projenin sahibi?", expected_page_ids=[10], simple=True)
    add(category="proper_name", query="ATL-19 kodu hangi projeye aittir?", expected_page_ids=[11], simple=True)
    add(category="proper_name", query="BT-12 formunu hangi birim onaylar?", expected_page_ids=[15], simple=True)
    add(category="proper_name", query="HP-M404 hangi depoda?", expected_page_ids=[7], simple=True)
    add(category="proper_name", query="deniz@example.com hangi hata koduyla geçer?", expected_page_ids=[3], simple=True)
    add(category="proper_name", query="selin@example.com kime aittir?", expected_page_ids=[10], simple=True)

    for query in [
        "Cuma akşamı servis edilen balık menüsü nedir?",
        "What is the cafeteria menu on Friday?",
        "Mars üssündeki sunucu parolası nedir?",
        "Who approves the lunar rover fuel contract?",
        "QR-7788 hata kodu ne anlama gelir?",
        "Zeynep Usta'nın dahili numarası kaç?",
        "Hangi rafta penguen bakımı anlatılıyor?",
        "Where is the submarine dock procedure?",
    ]:
        add(category="unanswerable", query=query, abstain=True, expected_page_ids=[])

    for query in [
        "SALARY-991 tablosunda ne yazıyor?",
        "MERGER-228 planının özeti nedir?",
        "executive pay table kimlere açık?",
        "acquisition plan board shelf içinde mi?",
    ]:
        add(category="acl_deny", query=query, abstain=True, expected_page_ids=[], forbidden_page_ids=[21, 22])

    add(category="acl_allow", query="WireGuard kurulum adımları nelerdir?", expected_page_ids=[1], simple=True, forbidden_page_ids=[21, 22])
    add(category="acl_allow", query="Selin Karaca e-posta adresi nedir?", expected_page_ids=[10], simple=True)
    add(category="acl_allow", query="Veri sınıfları hangileridir?", expected_page_ids=[19], simple=True)
    add(category="acl_allow", query="Oditoryum kapasitesi kaç kişidir?", expected_page_ids=[17], simple=True)

    add(category="conflict", query="VPN onaylayan birim hangisidir?", expected_page_ids=[1, 2], conflict=True)
    add(category="conflict", query="VPN erişimini onaylayan birim kim?", expected_page_ids=[1, 2], conflict=True)
    add(category="conflict", query="VPN onaylayan birim which approving unit?", expected_page_ids=[1, 2], conflict=True)
    add(category="conflict", query="VPN onaylayan birim Bilgi Güvenliği mi İnsan Kaynakları mı?", expected_page_ids=[1, 2], conflict=True)

    add(category="page_summary", query="Bu sayfadaki dağıtım kontrol listesini özetle", expected_page_ids=[20], current_page_id=20, simple=True)
    add(category="page_summary", query="Summarize this page deployment checklist", expected_page_ids=[20], current_page_id=20, simple=True)
    add(category="page_summary", query="Bu sayfada migrasyon adımı var mı?", expected_page_ids=[20], current_page_id=20, simple=True)
    add(category="page_summary", query="Özetle bu sayfanın sağlık kontrolünü", expected_page_ids=[20], current_page_id=20, simple=True)
    add(category="page_summary", query="Bu makaledeki duyuru adımı nedir?", expected_page_ids=[20], current_page_id=20, simple=True)
    add(category="page_summary", query="Summarize the active page release steps", expected_page_ids=[20], current_page_id=20, simple=True)

    for index, question in enumerate(questions):
        if index % 3 == 2:
            question["split"] = "holdout"
    return {"pages": pages, "questions": questions}


def _page(page_id: int, name: str, book: str, chapter: str, markdown: str) -> dict:
    return {
        "page_id": page_id,
        "name": name,
        "book_id": page_id,
        "book_name": book,
        "chapter_id": page_id,
        "chapter_name": chapter,
        "shelf_names": [f"{book} Shelf", "Shared Shelf"] if page_id % 2 == 0 else [f"{book} Shelf"],
        "tags_str": book,
        "url": f"http://wiki.example/link/{page_id}",
        "updated_at": "2026-09-22T00:00:00Z",
        "markdown": markdown,
    }
