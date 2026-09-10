"""Self-check parser plat nomor. Jalankan: python test_parser.py"""

from plates import parse_plate

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
]

for texts, expected in CASES:
    got = parse_plate(texts)
    assert got == expected, f"{texts!r} -> {got!r}, expected {expected!r}"

print("OK: semua kasus parser lolos")
