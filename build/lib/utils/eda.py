from IPython.display import Markdown, display
import pandas as pd
import numpy as np
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)


class QuickEDA:
    def __init__(self, df, name="dataset"):
        self.df = df
        self.name = name

        self.summary_num = None
        self.summary_cat = None
        self.problem_matrix = None
        self.duplicate_rows = None

    # ============================
    # DUPLICATE CHECK
    # ============================
    def check_duplicates(self):
        """Détecte les doublons ligne entière."""
        dup = self.df[self.df.duplicated(keep=False)]
        self.duplicate_rows = dup
        return dup

    # ============================
    # NUMERIC SUMMARY
    # ============================
    def compute_numeric_summary(self):
        df = self.df
        rows = []

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns

        for col in num_cols:
            s = df[col]
            nb_na = s.isna().sum()
            tx = 100 * (1 - nb_na / len(s))

            if s.dropna().empty:
                rows.append({
                    "colonne": col,
                    "type_exact": str(s.dtype),
                    "nb_valeurs_uniques": 0,
                    "min": None,
                    "max": None,
                    "moyenne": None,
                    "mediane": None,
                    "mode": None,
                    "ecart_type": None,
                    "q1": None,
                    "q3": None,
                    "iqr": None,
                    "skew": None,
                    "kurtosis": None,
                    "quasi_constante": True,
                    "asymetrique": False,
                    "kurtotique": False,
                    "tx_remplissage (%)": tx,
                    "nb_na": nb_na,
                    "NA_massif": tx < 60
                })
                continue

            mode_val = s.mode().iloc[0] if not s.mode().empty else None
            skew_val = s.skew()
            kurt_val = s.kurtosis()

            rows.append({
                "colonne": col,
                "type_exact": str(s.dtype),
                "nb_valeurs_uniques": s.nunique(),
                "min": s.min(),
                "max": s.max(),
                "moyenne": s.mean(),
                "mediane": s.median(),
                "mode": mode_val,
                "ecart_type": s.std(),
                "q1": s.quantile(0.25),
                "q3": s.quantile(0.75),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "skew": skew_val,
                "kurtosis": kurt_val,
                "quasi_constante": s.nunique() <= 2,
                "asymetrique": abs(skew_val) > 1,
                "kurtotique": kurt_val > 3,
                "tx_remplissage (%)": tx,
                "nb_na": nb_na,
                "NA_massif": tx < 60
            })

        self.summary_num = pd.DataFrame(rows).sort_values("nb_na", ascending=False)
        return self.summary_num

    # ============================
    # CATEGORICAL SUMMARY
    # ============================
    def compute_categorical_summary(self):
        df = self.df
        rows = []

        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            s = df[col]
            nb_na = s.isna().sum()
            tx = 100 * (1 - nb_na / len(s))

            vc = s.value_counts(dropna=True)
            mode_val = vc.index[0] if len(vc) > 0 else None
            mode_freq = vc.iloc[0] / len(s) if len(vc) > 0 else 0

            entropy = -(vc / len(s) * np.log2(vc / len(s))).sum() if len(vc) > 0 else 0

            rows.append({
                "colonne": col,
                "nb_valeurs_uniques": s.nunique(),
                "valeur_plus_frequente": mode_val,
                "pourcentage valeur plus fréquente (%)": round(mode_freq * 100, 2),
                "entropie": entropy,
                "quasi_constante": s.nunique() <= 2,
                "dominance": mode_freq > 0.90,
                "cardinalite_elevee": s.nunique() > 50,
                "asymetrique": mode_freq > 0.70,
                "entropie_faible": entropy < 1,
                "tx_remplissage (%)": tx,
                "nb_na": nb_na,
                "NA_massif": tx < 60
            })

        df_cat = pd.DataFrame(rows)

        # Si aucune colonne catégorielle → DataFrame vide
        if df_cat.shape[0] == 0:
            self.summary_cat = df_cat
            return df_cat

        # Sinon tri normal
        self.summary_cat = df_cat.sort_values("nb_na", ascending=False)
        return self.summary_cat
        return self.summary_cat

    # ============================
    # PROBLEM MATRIX (OUI ONLY)
    # ============================
    def build_problem_matrix(self):
        if self.summary_num is None:
            self.compute_numeric_summary()
        if self.summary_cat is None:
            self.compute_categorical_summary()

        num_problems = ["NA_massif", "quasi_constante", "asymetrique", "kurtotique"]
        df_num = self.summary_num[["colonne"] + num_problems].copy()

        cat_problems = ["NA_massif", "quasi_constante", "dominance",
                        "cardinalite_elevee", "asymetrique", "entropie_faible"]
        if self.summary_cat.empty:
            df_cat = pd.DataFrame(columns=["colonne"] + cat_problems)
        else:
            df_cat = self.summary_cat[["colonne"] + cat_problems].copy()

        all_problems = sorted(set(num_problems + cat_problems))

        for p in all_problems:
            if p not in df_num.columns:
                df_num[p] = ""
            if p not in df_cat.columns:
                df_cat[p] = ""

        for p in all_problems:
            df_num[p] = df_num[p].apply(lambda x: "OUI" if x is True else "")
            df_cat[p] = df_cat[p].apply(lambda x: "OUI" if x is True else "")

        df = pd.concat([df_num, df_cat], ignore_index=True)

        cols_to_keep = ["colonne"] + [p for p in all_problems if df[p].eq("OUI").any()]
        df = df[cols_to_keep]

        df = df[df.drop(columns=["colonne"]).apply(lambda row: row.eq("OUI").any(), axis=1)]

        self.problem_matrix = df
        return df
    
    # ============================
    # LISTE DES COLONNES + TYPES
    # ============================
    def compute_columns_types(self):
        df = self.df

        colonnes_types = pd.DataFrame({
            "colonne": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "nb_valeurs_uniques": [df[col].nunique() for col in df.columns],
            "tx_remplissage (%)": [100 * (1 - df[col].isna().mean()) for col in df.columns],
            "nb_na": [df[col].isna().sum() for col in df.columns],
        })

        return colonnes_types
    
    # ============================
    # EXPORT EXCEL
    # ============================
    def export_excel(self, path="../../reports/eda"):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{self.name}_eda.xlsx")

        # Nouveau tableau colonnes + types
        colonnes_types = self.compute_columns_types()

        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            self.summary_num.to_excel(writer, sheet_name="numerique", index=False)
            self.summary_cat.to_excel(writer, sheet_name="categoriel", index=False)
            self.problem_matrix.to_excel(writer, sheet_name="problemes", index=False)

            # 👉 Nouvel onglet
            colonnes_types.to_excel(writer, sheet_name="colonnes_types", index=False)

        print(f"Export Excel terminé : {file_path}")

    # ============================
    # RUN FULL PIPELINE
    # ============================
    def run(self, export=False):
        display(Markdown(f"## EDA : **{self.name}**"))

        dup = self.check_duplicates()
        display(Markdown(f"###  Doublons détectés : **{len(dup)}**"))
        if len(dup) > 0:
            display(dup.head())

        self.compute_numeric_summary()
        self.compute_categorical_summary()
        self.build_problem_matrix()

        if export:
            self.export_excel()

        return self.summary_num, self.summary_cat, self.problem_matrix

    # ============================
    # SHOW ONE COLUMN
    # ============================
    def show_column(self, col, mode="both"):
        """
        Affiche le résumé et/ou le graphique pour une seule colonne.
        mode : "resume", "graph", "both"
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        s = self.df[col]
        is_num = s.dtype in ["int64", "float64"]

        # Résumé numérique
        if is_num:
            if mode in ["resume", "both"]:
                display(Markdown(f"###  Résumé numérique : **{col}**"))
                summary = {
                    "type": str(s.dtype),
                    "nb_valeurs_uniques": s.nunique(),
                    "min": s.min(),
                    "max": s.max(),
                    "moyenne": s.mean(),
                    "mediane": s.median(),
                    "mode": s.mode().iloc[0] if not s.mode().empty else None,
                    "ecart_type": s.std(),
                    "q1": s.quantile(0.25),
                    "q3": s.quantile(0.75),
                    "iqr": s.quantile(0.75) - s.quantile(0.25),
                    "skew": s.skew(),
                    "kurtosis": s.kurtosis(),
                    "nb_na": s.isna().sum(),
                    "tx_remplissage (%)": round(100 * (1 - s.isna().mean()), 2)
                }
                display(pd.DataFrame([summary]))

            if mode in ["graph", "both"]:
                plt.figure(figsize=(10, 4))
                sns.histplot(s, kde=True, bins=40)
                plt.title(f"Distribution de {col}")
                plt.xlabel(col)
                plt.ylabel("Fréquence")
                plt.tight_layout()
                plt.show()

        # Résumé catégoriel
        else:
            if mode in ["resume", "both"]:
                display(Markdown(f"###  Résumé catégoriel : **{col}**"))
                vc = s.value_counts(dropna=False)
                summary = {
                    "type": str(s.dtype),
                    "nb_valeurs_uniques": s.nunique(),
                    "valeur_plus_frequente": vc.index[0],
                    "freq (%)": round(100 * vc.iloc[0] / len(s), 2),
                    "nb_na": s.isna().sum(),
                    "tx_remplissage (%)": round(100 * (1 - s.isna().mean()), 2)
                }
                display(pd.DataFrame([summary]))

            if mode in ["graph", "both"]:
                vc = s.value_counts(dropna=False)
                plt.figure(figsize=(10, 4))
                sns.countplot(data=self.df, x=col, order=vc.index)
                plt.title(f"Distribution de {col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()



        # ============================
    # BIVARIATE ANALYSIS WITH TARGET
    # ============================
    def bivariate_with_target(self, col, target_col='TARGET'):
        """
        Analyse bivariée entre une variable et la TARGET avec graphiques.
        - Numériques : boxplot + pie chart du % TARGET
        - Catégorielles : barplot des taux de défaut
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        if target_col not in self.df.columns:
            raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans le DataFrame.")

        if col not in self.df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

        s = self.df[col]
        is_num = s.dtype in ["int64", "float64"]

        display(Markdown(f"## Analyse bivariée : **{col}** vs **{target_col}**"))

        # ============================
        # NUMÉRIQUE
        # ============================
        if is_num:
            # Résumé statistique par TARGET
            display(Markdown("### Résumé statistique par TARGET"))
            summary = self.df.groupby(target_col)[col].describe().T
            display(summary)

            # Boxplot
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=self.df, x=target_col, y=col)
            plt.title(f"{col} - Boxplot par TARGET")
            plt.tight_layout()
            plt.show()

        # ============================
        # CATÉGORIEL
        # ============================
        else:
            display(Markdown("### Taux de défaut par catégorie"))

            rates = (
                self.df.groupby(col)[target_col]
                .mean()
                .sort_values(ascending=False)
            )

            display(rates.to_frame("taux_defaut"))

            # Barplot
            plt.figure(figsize=(12, 5))
            sns.barplot(x=rates.index, y=rates.values)
            plt.title(f"Taux de défaut par catégorie - {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()



        # ============================
    # RELATION WITH TARGET (NORMALIZED ONLY)
    # ============================
    def relation_with_target(self, target_col='TARGET', top=None):
        """
        Analyse la relation entre chaque variable et la TARGET.
        Méthode unique : différence normalisée
        (mean(X|1) - mean(X|0)) / std(X)

        top : nombre de variables à retourner pour les plus fortes et plus faibles
        """

        if target_col not in self.df.columns:
            raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans le DataFrame.")

        summary = {}

        for col in self.df.columns:
            if col == target_col:
                continue

            s = self.df[col]

            # ============================
            # NUMÉRIQUES
            # ============================
            if s.dtype in ['int64', 'float64']:
                mean_1 = self.df[self.df[target_col] == 1][col].mean()
                mean_0 = self.df[self.df[target_col] == 0][col].mean()
                std = s.std()

                value = (mean_1 - mean_0) / std if std != 0 else 0
                summary[col] = value

            # ============================
            # CATÉGORIELLES
            # ============================
            else:
                rates = self.df.groupby(col)[target_col].mean()
                summary[col] = rates.max() - rates.min()

        summary_df = (
            pd.DataFrame.from_dict(summary, orient='index', columns=['relation_strength'])
            .sort_values('relation_strength', ascending=False)
        )

        # ============================
        # EXTRACTION TOP + BOTTOM
        # ============================
        if top:
            top_pos = summary_df.head(top)
            top_neg = summary_df.tail(top)
            return top_pos, top_neg

        return summary_df
    
    def compute_correlations(self, threshold=0.3, plot=False):
        """
        Calcule les corrélations entre variables numériques
        et retourne uniquement celles dont |corr| >= threshold.
        - threshold : seuil minimal de corrélation absolue
        - plot : True pour afficher une heatmap
        """

        import seaborn as sns
        import matplotlib.pyplot as plt

        # Sélection des colonnes numériques
        df_num = self.df.select_dtypes(include=["int64", "float64"])

        # Matrice de corrélation
        corr_matrix = df_num.corr()

        # Extraction des corrélations fortes
        corr_list = []

        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 < col2:  # éviter doublons + diagonale
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= threshold:
                        corr_list.append({
                            "var1": col1,
                            "var2": col2,
                            "correlation": corr_value
                        })

        corr_df = pd.DataFrame(corr_list).sort_values(
            "correlation", key=lambda x: abs(x), ascending=False
        )

        # Heatmap optionnelle
        if plot:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
            plt.title(f"Heatmap des corrélations (seuil = {threshold})")
            plt.tight_layout()
            plt.show()

        return corr_df