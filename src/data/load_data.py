import os
import pandas as pd

class CSVLoader:
    """
    Charge automatiquement tous les fichiers CSV d'un dossier
    et les stocke dans un dictionnaire {nom: DataFrame}.
    """

    def __init__(self, path="../data/raw/", encoding="latin-1", ignore=None):
        self.path = path
        self.encoding = encoding
        self.ignore = ignore if ignore else []
        self.dataframes = {}

    def _load_csv(self, filepath):
        """Charge un CSV avec fallback d'encodage."""
        try:
            return pd.read_csv(filepath, encoding=self.encoding)
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding="utf-8", errors="replace")

    def load(self):
        """Charge tous les CSV du dossier."""
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                name = file.replace(".csv", "")
                if name not in self.ignore:
                    full_path = os.path.join(self.path, file)
                    df = self._load_csv(full_path)
                    self.dataframes[name] = df
                    print(f"Loaded: {name:<30} shape = {df.shape}")
        return self.dataframes

    def get(self, name):
        """Récupère un DataFrame par son nom."""
        return self.dataframes.get(name)

    def list(self):
        """Liste les noms des datasets chargés."""
        return list(self.dataframes.keys())

    def to_list(self):
        """Retourne une liste de DataFrames dans l'ordre alphabétique."""
        return [self.dataframes[name] for name in sorted(self.dataframes.keys())]