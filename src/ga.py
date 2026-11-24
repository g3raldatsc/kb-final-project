import numpy as np
import pandas as pd

def mape_fts(batas, data_latih, acuan, aktual):
    """MAPE Fuzzy Time Series model Chen orde satu dengan cara cepat (tanpa kelas).
    Aturan dipelajari dari data_latih. Peramalan memakai pasangan (acuan -> aktual),
    yaitu acuan adalah harga pencatatan sebelumnya dan aktual adalah harga hari itu.
    """
    k = len(batas) - 1
    himpunan = np.clip(np.searchsorted(batas, data_latih, side="right") - 1, 0, k - 1)
    ada = np.zeros((k, k), dtype=bool)
    ada[himpunan[:-1], himpunan[1:]] = True

    tengah = (batas[:-1] + batas[1:]) / 2
    jumlah_tujuan = ada.sum(axis=1)
    jumlah_tengah = (ada * tengah).sum(axis=1)
    titik_ramal = np.where(jumlah_tujuan > 0, jumlah_tengah / np.maximum(jumlah_tujuan, 1), tengah)

    indeks = np.clip(np.searchsorted(batas, acuan, side="right") - 1, 0, k - 1)
    return float(np.mean(np.abs(aktual - titik_ramal[indeks]) / aktual) * 100)

def lebar_ke_batas(lebar, bawah, atas):
    """Mengubah proporsi lebar interval (jumlah 1) menjadi K+1 titik batas dari bawah sampai atas."""
    kumulatif = np.concatenate([[0.0], np.cumsum(lebar, axis=-1)])
    return bawah + (atas - bawah) * kumulatif

def perbaiki(lebar, lebar_min):
    """Memastikan setiap proporsi lebar >= lebar_min dan jumlah tiap baris tepat 1."""
    lebar = np.atleast_2d(np.asarray(lebar, dtype=float))
    k = lebar.shape[1]
    ekstra = np.maximum(lebar - lebar_min, 0.0)
    jumlah = ekstra.sum(axis=1, keepdims=True)
    ekstra = np.where(jumlah > 0, ekstra, 1.0)
    jumlah = ekstra.sum(axis=1, keepdims=True)
    return lebar_min + ekstra / jumlah * (1.0 - k * lebar_min)

def jalankan_ga(data_latih, k, data_validasi=None, acuan_validasi=None,
                ukuran_populasi=60, jumlah_generasi=150, peluang_crossover=0.9,
                peluang_mutasi=0.2, sigma_mutasi=0.25, ukuran_turnamen=3,
                elitisme=2, faktor_lebar_min=0.25, seed=42):
    """Algoritma Genetika untuk mengoptimasi batas interval Fuzzy Time Series.
    Kromosom adalah K proporsi lebar interval (jumlah 1, masing-masing minimal
    faktor_lebar_min / K). Fitness adalah MAPE pada data latih (diminimalkan).
    Individu pertama populasi awal adalah pembagian sama panjang.
    Data validasi hanya dipakai untuk diagnostik, tidak memengaruhi pemilihan.
    """
    rng = np.random.default_rng(seed)
    data_latih = np.asarray(data_latih, dtype=float)
    bawah, atas = data_latih.min(), data_latih.max()
    lebar_min = faktor_lebar_min / k
    acuan_lat, aktual_lat = data_latih[:-1], data_latih[1:]

    def galat_latih(lebar):
        return mape_fts(lebar_ke_batas(lebar, bawah, atas), data_latih, acuan_lat, aktual_lat)

    populasi = perbaiki(rng.dirichlet(np.full(k, 4.0), size=ukuran_populasi), lebar_min)
    populasi[0] = np.full(k, 1.0 / k)

    riwayat = []
    for generasi in range(jumlah_generasi + 1):
        galat = np.array([galat_latih(individu) for individu in populasi])
        urutan = np.argsort(galat)
        populasi, galat = populasi[urutan], galat[urutan]

        catatan = {"generasi": generasi, "mape_latih_terbaik": galat[0], "mape_latih_rata2": galat.mean()}
        if data_validasi is not None:
            batas_terbaik = lebar_ke_batas(populasi[0], bawah, atas)
            catatan["mape_validasi_terbaik"] = mape_fts(
                batas_terbaik, data_latih,
                np.asarray(acuan_validasi, dtype=float), np.asarray(data_validasi, dtype=float))
        riwayat.append(catatan)
        if generasi == jumlah_generasi:
            break

        # seleksi turnamen
        peserta = rng.integers(0, ukuran_populasi, size=(ukuran_populasi, ukuran_turnamen))
        pemenang = peserta[np.arange(ukuran_populasi), galat[peserta].argmin(axis=1)]
        induk_a = populasi[pemenang]
        induk_b = populasi[rng.permutation(pemenang)]

        # crossover aritmetika
        alfa = rng.random((ukuran_populasi, 1))
        silang = rng.random((ukuran_populasi, 1)) < peluang_crossover
        anak = np.where(silang, alfa * induk_a + (1 - alfa) * induk_b, induk_a)

        # mutasi gaussian pada lebar relatif
        mutasi = rng.random(anak.shape) < peluang_mutasi
        anak = anak * (1.0 + mutasi * rng.normal(0.0, sigma_mutasi, anak.shape))
        anak = perbaiki(anak, lebar_min)

        # elitisme
        anak[:elitisme] = populasi[:elitisme]
        populasi = anak

    batas_terbaik = lebar_ke_batas(populasi[0], bawah, atas)
    return {
        "batas": batas_terbaik,
        "lebar": populasi[0],
        "mape_latih": float(galat[0]),
        "riwayat": pd.DataFrame(riwayat).set_index("generasi"),
    }

def batas_kuantil(data_latih, k):
    """Batas interval berfrekuensi sama: tiap interval berisi kira-kira jumlah data latih yang sama."""
    batas = np.quantile(np.asarray(data_latih, dtype=float), np.linspace(0.0, 1.0, k + 1))
    if np.any(np.diff(batas) <= 0):
        raise ValueError("Batas kuantil tidak menaik tegas (ada nilai kembar pada batas)")
    return batas

def pencarian_acak(data_latih, k, jumlah_sampel, seed=42, faktor_lebar_min=0.25):
    """Pencarian acak murni: mencoba jumlah_sampel pembagian acak dan mengambil yang MAPE latihnya terkecil."""
    rng = np.random.default_rng(seed)
    data_latih = np.asarray(data_latih, dtype=float)
    bawah, atas = data_latih.min(), data_latih.max()
    sampel = perbaiki(rng.dirichlet(np.full(k, 4.0), size=jumlah_sampel), faktor_lebar_min / k)
    galat = np.array([
        mape_fts(lebar_ke_batas(lebar, bawah, atas), data_latih, data_latih[:-1], data_latih[1:])
        for lebar in sampel
    ])
    terbaik = int(galat.argmin())
    return {"batas": lebar_ke_batas(sampel[terbaik], bawah, atas), "mape_latih": float(galat[terbaik])}