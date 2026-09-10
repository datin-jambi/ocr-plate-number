"""Self-check parser plat nomor. Jalankan: python test_parser.py

Murni logika teks - tidak memuat model, jadi selesai dalam sekejap.
"""

from plates import first_line, parse_plate

CASES = [
    ([("B", 0.9), ("1234", 0.9), ("ABC", 0.9)], "B 1234 ABC"),
    (["DK9999XYZ"], "DK 9999 XYZ"),
    (["B 1234 ABC"], "B 1234 ABC"),
    (["BH", "1", "AA"], "BH 1 AA"),
    (["HELLO", "WORLD"], None),
    ([], None),
    # kode wilayah tak valid -> tolak
    (["XX 1234 ABC"], None),
    # confusion char: 8->B pada kode wilayah, O->0 & S->5 pada angka
    (["8H", "1O25", "AA"], "BH 1025 AA"),
    # noise di sekeliling plat
    (["INDONESIA", "BE9012XY", "2027"], "BE 9012 XY"),
    # angka semu semua (hasil coerce huruf) -> tolak
    (["B", "OOOO", "ABC"], None),
    # kode wilayah gagal terbaca: lebih baik None daripada "T 0025 JT"
    (["1002SJT"], None),
    (["B1002SJT"], "B 1002 SJT"),
    # "0" di awal boleh jadi D karena 4 digit asli menyusul
    (["05299UCD"], "D 5299 UCD"),
]

for texts, expected in CASES:
    got = parse_plate(texts)
    assert got == expected, f"{texts!r} -> {got!r}, expected {expected!r}"


def box(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


# Plat 2 baris: nomor di atas, masa berlaku di bawah. Baris bawah harus
# dibuang, sisanya urut kiri->kanan berapa pun urutan OCR mengembalikannya.
TWO_LINES = [
    (box(300, 200, 60, 30), "08", 0.9),
    (box(120, 50, 90, 50), "5299", 0.99),
    (box(20, 52, 40, 48), "D", 0.99),
    (box(230, 51, 70, 49), "UCD", 1.0),
    (box(370, 202, 60, 30), "25", 0.4),
]
assert [t for t, _ in first_line(TWO_LINES)] == ["D", "5299", "UCD"]
assert parse_plate(first_line(TWO_LINES)) == "D 5299 UCD"
assert first_line([]) == []
assert len(first_line(TWO_LINES[1:4])) == 3  # satu baris: tak ada yang dibuang

print("OK: semua kasus parser lolos")
