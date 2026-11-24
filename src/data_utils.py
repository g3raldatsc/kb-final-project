import pandas as pd

def muat_pihps_lebar(path, komoditas="Cabai Rawit Merah"):
    """ini itu untuk membaca tabel PIHPS format lebar (kolom = tanggal) menjadi Series harga harian.

    Nilai kosong ('-') menjadi NaN. Gak ada pengisian atau pembersihan lain.
    """
    mentah = pd.read_excel(path, header=0, dtype=str)
    kolom_nama = mentah.columns[1]

    baris = mentah[mentah[kolom_nama].str.strip() == komoditas]
    if baris.empty:
        raise ValueError(
            f"Komoditas '{komoditas}' tidak ditemukan. Pilihan: {mentah[kolom_nama].tolist()}"
        )
    baris = baris.iloc[0]

    kolom_tanggal = mentah.columns[2:]
    tanggal = pd.to_datetime(
        pd.Series(kolom_tanggal).str.replace(" ", "", regex=False), format="%d/%m/%Y"
    )
    teks_harga = baris[kolom_tanggal].astype(str).str.replace(",", "", regex=False)
    harga = pd.to_numeric(teks_harga, errors="coerce")

    seri = pd.Series(harga.values, index=pd.DatetimeIndex(tanggal, name="tanggal"), name="harga")
    return seri.sort_index()

def muat_data_bersih(path):
    """ini buat baca berkas CSV data bersih (kolom: tanggal, harga) menjadi Series berindeks tanggal."""
    df = pd.read_csv(path, parse_dates=["tanggal"])
    return df.set_index("tanggal")["harga"].sort_index()