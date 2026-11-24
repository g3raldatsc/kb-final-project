import numpy as np

def _ke_array(aktual, prediksi):
    a = np.asarray(aktual, dtype=float)
    p = np.asarray(prediksi, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"Bentuk berbeda: aktual {a.shape}, prediksi {p.shape}")
    if np.isnan(a).any() or np.isnan(p).any():
        raise ValueError("Aktual atau prediksi mengandung nilai kosong")
    return a, p

def mape(aktual, prediksi):
    """Mean Absolute Percentage Error dalam persen."""
    a, p = _ke_array(aktual, prediksi)
    return float(np.mean(np.abs(a - p) / np.abs(a)) * 100)

def rmse(aktual, prediksi):
    """Root Mean Squared Error dalam satuan harga (rupiah)."""
    a, p = _ke_array(aktual, prediksi)
    return float(np.sqrt(np.mean((a - p) ** 2)))

def mae(aktual, prediksi):
    """Mean Absolute Error dalam satuan harga (rupiah)."""
    a, p = _ke_array(aktual, prediksi)
    return float(np.mean(np.abs(a - p)))

def ringkas_metrik(aktual, prediksi):
    return {"MAPE_%": mape(aktual, prediksi), "RMSE": rmse(aktual, prediksi), "MAE": mae(aktual, prediksi)}