import numpy as np
import pandas as pd

def interval_sama_panjang(data, jumlah_interval, margin=0.0):
    """Batas interval (jumlah_interval + 1 titik) sama panjang dari nilai minimum sampai maksimum data."""
    bawah = float(np.min(data)) - margin
    atas = float(np.max(data)) + margin
    return np.linspace(bawah, atas, jumlah_interval + 1)

def fuzzifikasi(nilai, batas):
    """Indeks himpunan fuzzy (0 sampai K-1) untuk tiap nilai.
    Nilai di luar semesta dipetakan ke himpunan terdekat (paling bawah atau paling atas).
    """
    nilai = np.asarray(nilai, dtype=float)
    indeks = np.searchsorted(batas, nilai, side="right") - 1
    return np.clip(indeks, 0, len(batas) - 2)

class FuzzyTimeSeriesChen:
    """Fuzzy Time Series model Chen orde satu dengan batas interval yang diberikan."""
    def __init__(self, batas):
        batas = np.asarray(batas, dtype=float)
        if batas.ndim != 1 or len(batas) < 3 or np.any(np.diff(batas) <= 0):
            raise ValueError("batas harus menaik tegas dan memuat minimal dua interval")
        self.batas = batas
        self.k = len(batas) - 1
        self.titik_tengah = (batas[:-1] + batas[1:]) / 2

    def fit(self, data_latih):
        himpunan = fuzzifikasi(data_latih, self.batas)
        kiri, kanan = himpunan[:-1], himpunan[1:]

        self.jumlah_data = np.bincount(himpunan, minlength=self.k)
        self.jumlah_transisi = np.bincount(kiri, minlength=self.k)
        self.flrg = {i: sorted(set(kanan[kiri == i].tolist())) for i in range(self.k)}

        self.titik_ramal = np.array([
            self.titik_tengah[self.flrg[i]].mean() if self.flrg[i] else self.titik_tengah[i]
            for i in range(self.k)
        ])
        return self

    def ramal(self, nilai_kemarin):
        """Peramalan satu langkah: dari nilai pencatatan sebelumnya ke harga berikutnya."""
        return self.titik_ramal[fuzzifikasi(nilai_kemarin, self.batas)]

    def tabel_aturan(self):
        baris = []
        for i in range(self.k):
            baris.append({
                "himpunan": f"A{i + 1}",
                "batas_bawah": self.batas[i],
                "batas_atas": self.batas[i + 1],
                "jumlah_data": int(self.jumlah_data[i]),
                "jumlah_transisi": int(self.jumlah_transisi[i]),
                "tujuan": ", ".join(f"A{j + 1}" for j in self.flrg[i]) or "(tanpa aturan)",
                "titik_ramal": self.titik_ramal[i],
            })
        return pd.DataFrame(baris).set_index("himpunan")