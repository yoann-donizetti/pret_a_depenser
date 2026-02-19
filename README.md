# Prêt à Dépenser — Credit Scoring (MLOps)

Ce projet consiste à développer un modèle de **credit scoring** permettant d’estimer la probabilité de défaut d’un client.

Il s’inscrit dans une démarche **MLOps complète**, allant du tracking d’expériences (**MLflow**) jusqu’au déploiement d’une **API FastAPI**, prête à être conteneurisée et déployée (Hugging Face / Docker).

Le projet est organisé en deux parties :

- **Partie 1/2 : Modélisation + tracking MLflow**
- **Partie 2/2 : Déploiement API + packaging + monitoring**

---
## Sommaire

- [Prêt à Dépenser — Credit Scoring (MLOps)](#prêt-à-dépenser--credit-scoring-mlops)
- [Objectif du projet](#objectif-du-projet)
- [Enjeux métier](#enjeux-métier)
- [Partie 1/2 — Modélisation & Tracking MLflow](#partie-12--modélisation--tracking-mlflow)
  - [Travaux réalisés](#travaux-réalisés)
  - [Modèle final retenu](#modèle-final-retenu)
- [Partie 2/2 — Déploiement API (FastAPI)](#partie-22--déploiement-api-fastapi)
- [Reproductibilité](#reproductibilité)
- [Structure du projet](#structure-du-projet)
- [Installation rapide](#installation-rapide)
- [Lancer MLflow UI (optionnel)](#lancer-mlflow-ui-optionnel)
- [Lancer l’API en local](#lancer-lapi-en-local)
- [Exemple d'appel API](#exemple-dappel-api)
  - [Healthcheck](#healthcheck)
  - [Prédiction](#prédiction)
- [Chargement du modèle (local ou Hugging Face)](#chargement-du-modèle-local-ou-hugging-face)
  - [Mode local](#mode-local)
  - [Mode Hugging Face](#mode-hugging-face)
- [Tests unitaires](#tests-unitaires)
- [Résultat final](#résultat-final)

---

## Objectif du projet

L’entreprise *Prêt à Dépenser* propose des crédits à la consommation à des clients ayant peu ou pas d’historique bancaire.

L’objectif est de construire un modèle capable de :

- prédire la **probabilité de défaut**
- produire une décision automatique (**ACCEPTED / REFUSED**)
- respecter une contrainte métier : **FN >> FP**

---

## Enjeux métier

Le dataset est fortement déséquilibré et les erreurs n’ont pas le même impact :

- **FN (mauvais client prédit bon)** = perte financière importante
- **FP (bon client prédit mauvais)** = manque à gagner

Mise en place d’une fonction de coût métier :

**Coût = 10 × FN + 1 × FP**

---

## Partie 1/2 — Modélisation & Tracking MLflow

### Travaux réalisés

- Fusion et agrégation de tables multi-sources
- Dataset initial : **1658 variables**
- Benchmark de plusieurs modèles :
  - Logistic Regression
  - Random Forest
  - LightGBM
  - XGBoost
  - CatBoost
  - MLP
- Sélection progressive des variables :
  - réduction de **1658 → 125 features**
- Tuning d’hyperparamètres
- Optimisation du seuil métier
- Interprétabilité avec SHAP :
  - explication globale
  - explication locale

### Modèle final retenu

Le modèle **CatBoostClassifier** est retenu car il obtient le **meilleur coût métier** sur le jeu de test tout en conservant des performances stables.

---

## Partie 2/2 — Déploiement API (FastAPI)

L’API permet de prédire le risque de défaut à partir d’un JSON représentant un client.

Endpoints :

- `GET /health` → état de disponibilité
- `POST /predict` → prédiction score + décision

---

## Reproductibilité

Le projet est reproductible en exécutant les notebooks dans l’ordre :

- `notebooks/01_data_preparation/` : préparation et agrégation des tables
- `notebooks/02_benchmark/` : benchmark des modèles
- `notebooks/03_modeling/` : sélection features, tuning, seuil métier, modèle final
- `notebooks/04_preparation_API/` : génération du modèle et artifacts pour l’API

---

## Structure du projet

```bash
pret-a-depenser/

│
├── app/                          # Code API FastAPI (déploiement)
│   ├── assets/                   # Artifacts locaux (gitignored)
│   │   ├── model/
│   │   │   └── model.cb
│   │   └── api_artifacts/
│   │       ├── kept_features_top125_nocorr.txt
│   │       ├── cat_features_top125_nocorr.txt
│   │       └── threshold_catboost_top125_nocorr.json
│   │
│   ├── model/                    # Fonctions liées au modèle (loader/predict)
│   │   ├── loader.py
│   │   └── predict.py
│   │
│   ├── utils/                    # Fonctions utilitaires (validation, erreurs, IO…)
│   │   ├── errors.py
│   │   ├── io.py
│   │   └── validation.py
│   │
│   ├── config.py                 # Configuration (env vars / chemins / HF)
│   ├── main.py                   # Point d’entrée FastAPI
│   └── schemas.py                # Schémas Pydantic (PredictRequest, PredictResponse…)
│
├── artifacts/                    # Export notebook (artifacts API, modèle, seuil)
│
├── data/                         # Données (non versionnées dans GitHub)
│   ├── raw/                      # Données brutes
│   ├── clean/                    # Données nettoyées
│   └── processed/                # Dataset final prêt pour entraînement
│
├── examples/                     # Exemples d’inputs JSON pour tester l’API
│   └── input_example.json
│
├── htmlcov/                      # Rapport coverage pytest (auto généré)
│
├── mlruns/                       # Dossier MLflow local (tracking runs)
│
├── notebooks/                    # Notebooks du projet (pipeline complet)
│   ├── 01_data_preparation/      # Préparation + agrégation multi-tables
│   ├── 02_benchmark/             # Benchmark des modèles
│   ├── 03_modeling/              # Feature selection + tuning + seuil métier
│   └── 04_preparation_API/       # Packaging modèle + artifacts pour API
│
├── reports/                      # Rapports, figures, exports (EDA, SHAP…)
│
├── src/                          # Code Python "métier" utilisé par les notebooks
│   ├── data/                     # Fonctions de préparation des données
│   ├── modeling/                 # Entraînement + évaluation modèles
│   ├── tracking/                 # MLflow logging (params, metrics, artifacts)
│   └── utils/                    # Helpers génériques
│
├── tests/                        # Tests unitaires (pytest)
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_io.py
│   ├── test_lifespan.py
│   ├── test_loader.py
│   ├── test_predict.py
│   └── test_validation.py
│
├── .env                          # Variables d'environnement (local uniquement)
├── .gitignore                    # Exclusion fichiers lourds / secrets
├── pytest.ini                    # Config pytest
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation principale
```


## Installation rapide
Créer un environnement virtuel puis installer les dépendances :
```bash
pip install -r requirements.txt
```

## Lancer MLflow UI (optionnel)
MLflow permet de visualiser les runs et comparer les expérimentations :
```bash
mlflow ui
```
Puis ouvrir :
http://127.0.0.1:5000

## Lancer l'API en local
```bash
uvicorn app.main:app --reload
```

API accessible sur :
http://127.0.0.1:8000
Documentation Swagger :
http://127.0.0.1:8000/docs

## Exemple d'appel API

### Healthcheck
```bash
curl http://127.0.0.1:8000/health
```
### Prédiction

Un exemple de payload est disponible ici :
examples/input_example.json
Commande :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/input_example.json
```

## Chargement du modèle (local ou Hugging Face)
L’API supporte deux modes de chargement des artifacts :
### Mode local
Les fichiers sont stockés dans :
app/assets/
Ce dossier est **gitignored.**
Variables :

- BUNDLE_SOURCE=local
- LOCAL_MODEL_PATH
- LOCAL_KEPT_PATH
- LOCAL_CAT_PATH
- LOCAL_THRESHOLD_PATH

### Mode Hugging Face
Les fichiers sont téléchargés depuis un repo Hugging Face.
Variables :
- BUNDLE_SOURCE=hf
- HF_REPO_ID
- HF_TOKEN (optionnel si repo public)
- HF_MODEL_PATH
- HF_KEPT_PATH
- HF_CAT_PATH
- HF_THRESHOLD_PATH

## Tests unitaires
Lancer les tests :

```bash
pytest --cov=app
```
## Résultat final
Ce projet fournit :
- un modèle de scoring optimisé selon un coût métier
- un tracking complet des expérimentations via MLflow
- une API FastAPI testée (pytest)
- une architecture prête pour le déploiement Docker / Hugging Face
## Structure du projet

```bash
project/
│
├── data/
│   ├── raw/
│   ├── clean/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_preparation/
│   ├── 02_benchmark/
│   └── 03_modeling/
│
├── outputs/
│
├── src/
│   ├── data/
│   ├── modeling/
│   ├── tracking/
│   └── utils/
│
├── requirements.txt
└── README.md
```
## Installation rapide

```bash
pip install -r requirements.txt
```
