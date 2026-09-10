"""Domain plat nomor Indonesia: normalisasi teks OCR -> nomor plat.

Murni logika teks. Tidak bergantung pada OpenCV, EasyOCR, atau Flask,
sehingga bisa diuji tanpa memuat model apa pun.
"""

import re

# Kode wilayah plat Indonesia. Dipakai sebagai filter keras kandidat.
AREA_CODES = {
    "A", "AA", "AB", "AD", "AE", "AG", "B", "BA", "BB", "BD", "BE", "BG", "BH",
    "BK", "BL", "BM", "BN", "BP", "D", "DA", "DB", "DC", "DD", "DE", "DG", "DH",
    "DK", "DL", "DM", "DN", "DP", "DR", "DS", "DT", "DU", "DW", "E", "EA", "EB",
    "ED", "F", "G", "H", "K", "KB", "KH", "KT", "KU", "L", "M", "N", "P", "PA",
    "PB", "R", "S", "T", "W", "Z",
}

# Confusion OCR: dipakai sesuai posisi (huruf vs angka).
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}
# Posisi kode wilayah: satu digit bisa jadi beberapa huruf ("0" -> D atau O).
# Kandidat disaring keras oleh AREA_CODES, jadi ambigu di sini aman.
AREA_ALT = {"0": "DO", "1": "IT", "2": "Z", "3": "BE", "4": "A", "5": "S",
            "6": "G", "7": "T", "8": "BR", "9": "P"}

MAX_SUFFIX = 3   # panjang maksimum huruf belakang
MAX_NUMBER = 4   # panjang maksimum angka tengah


def _coerce(chunk, table, want_digit):
    """Paksa tiap char ke satu kelas (angka atau huruf).

    Return (hasil, jumlah_char_yang_sudah_benar) atau None kalau ada char
    yang tidak bisa dipetakan.
    """
    out, clean = [], 0
    for ch in chunk:
        if ch.isdigit() == want_digit:
            out.append(ch)
            clean += 1
        elif ch in table:
            out.append(table[ch])
        else:
            return None
    return "".join(out), clean


def _area_candidates(chunk):
    """Semua tafsir kode wilayah yang mungkin dari potongan OCR.

    Digit dipetakan ke beberapa huruf sekaligus (AREA_ALT) karena OCR sering
    membaca "D" sebagai "0". Hasil disaring AREA_CODES oleh pemanggil.
    Return list (teks, jumlah_char_asli).
    """
    outs = [("", 0)]
    for ch in chunk:
        if ch.isalpha():
            opts = [(ch, 1)]
        elif ch in AREA_ALT:
            opts = [(c, 0) for c in AREA_ALT[ch]]
        else:
            return []
        outs = [(p + c, n + k) for p, n in outs for c, k in opts]
    return outs


def _score(area, num, suf, clean, area_clean, num_clean, conf, at_start):
    """Nilai kandidat plat. None = ditolak."""
    if area not in AREA_CODES:
        return None
    if num_clean == 0 or num_clean * 2 < len(num):  # angka harus dominan digit asli
        return None
    # Kode wilayah tanpa huruf asli (mis. OCR baca "D" jadi "0") baru diterima
    # kalau ia di awal teks DAN nomornya 4 digit yang semuanya asli. Tanpa itu
    # plat yang kode wilayahnya gagal terbaca ("1002 SJT") akan dicuri digit
    # dan hurufnya jadi "T 0025 JT". Untuk data pajak, not_found lebih aman
    # daripada nomor yang salah.
    if area_clean == 0 and not (at_start and len(num) == MAX_NUMBER and num_clean == MAX_NUMBER):
        return None
    return clean + conf * 2 + (2 if suf else 0) + (1 if len(num) == MAX_NUMBER else 0) \
        + (1 if area_clean else 0)


def _fragments(texts):
    """Bersihkan fragmen OCR jadi [(teks_A-Z0-9, confidence)]."""
    frags = []
    for t in texts:
        s, c = (t, 1.0) if isinstance(t, str) else (t[0], float(t[1]))
        s = re.sub(r"[^A-Z0-9]", "", s.upper())
        if s:
            frags.append((s, c))
    return frags


def parse_plate(texts):
    """Plat Indonesia terbaik dari fragmen OCR. Item: str atau (str, confidence).

    Gabungkan semua fragmen lalu coba tiap substring sebagai
    KODE_WILAYAH + ANGKA + SUFIKS, ambil skor tertinggi.
    """
    frags = _fragments(texts)
    if not frags:
        return None

    joined = "".join(s for s, _ in frags)
    conf = sum(c for _, c in frags) / len(frags)

    best, best_score = None, 0.0
    n = len(joined)
    for i in range(n):
        for j in range(i + 3, min(i + 10, n + 1)):
            sub = joined[i:j]
            for a in (1, 2):  # panjang kode wilayah
                for c_len in range(MAX_SUFFIX, -1, -1):
                    b = len(sub) - a - c_len
                    if not 1 <= b <= MAX_NUMBER:
                        continue
                    num = _coerce(sub[a:a + b], TO_DIGIT, True)
                    suf = _coerce(sub[a + b:], TO_LETTER, False) if c_len else ("", 0)
                    if not (num and suf):
                        continue
                    for area, area_clean in _area_candidates(sub[:a]):
                        sc = _score(area, num[0], suf[0], area_clean + num[1] + suf[1],
                                    area_clean, num[1], conf, i == 0)
                        if sc is not None and sc > best_score:
                            best_score = sc
                            best = " ".join(p for p in (area, num[0], suf[0]) if p)
    return best


def first_line(raw):
    """Ambil fragmen baris nomor plat saja, urut kiri->kanan.

    Plat Indonesia 2 baris: nomor di atas, masa berlaku (BB.YY) di bawah.
    Tanpa pemisahan ini fragmen tanggal ikut tergabung dan parser bisa
    memilih digit tanggal sebagai nomor (mis. "D 5299 UCD" -> "D 12 S").
    Item raw: (bbox, text, conf) dari EasyOCR detail=1.
    """
    if not raw:
        return []
    items = []
    for box, text, conf in raw:
        ys = [p[1] for p in box]
        items.append((min(ys), max(ys) - min(ys), min(p[0] for p in box), text, conf))

    # toleransi = 60% tinggi karakter median, cukup untuk memisah dua baris
    tol = max(1.0, sorted(i[1] for i in items)[len(items) // 2] * 0.6)
    top = min(i[0] for i in items)
    line = sorted((i for i in items if i[0] - top < tol), key=lambda i: i[2])
    return [(i[3], i[4]) for i in line]
