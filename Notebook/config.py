from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

DIR_RAW = RAIZ / "Dataset" / "Raw"
DIR_CLEAN = RAIZ / "Dataset" / "Clean"
DIR_MODEL = RAIZ / "Model"
DIR_IMG = RAIZ / "xLain_Lain" / "img"

for d in (DIR_RAW, DIR_CLEAN, DIR_MODEL, DIR_IMG):
    d.mkdir(parents=True, exist_ok=True)

FILE_RAW = DIR_RAW / "dataset_harga_cabai_rawit_merah_jatim_pihps.xlsx"
KOMODITAS = "Cabai Rawit Merah"
SEED = 42