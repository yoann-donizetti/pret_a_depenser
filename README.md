---
title: Pret a Dépenser API
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Prêt à Dépenser — Credit Scoring (MLOps)

Ce projet consiste à développer un modèle de **credit scoring** permettant d'estimer la probabilité de défaut d'un client.

Il s'inscrit dans une démarche **MLOps complète**, allant du tracking d'expériences (**MLflow**) jusqu'au déploiement d'une **API FastAPI**, prête à être conteneurisée et déployée (Hugging Face / Docker).

Le projet est organisé en deux parties :

- **Partie 1/2 : Modélisation + tracking MLflow**
- **Partie 2/2 : Déploiement API + packaging + monitoring**

---

## Architecture simplifiée

```
Client
   ↓
FastAPI
   ↓
PostgreSQL (logging + features)
   ↓
Modèle CatBoost
   ↓
Monitoring Streamlit

CI/CD GitHub → Déploiement automatique HF Spaces
```

---

## Sommaire

1. [Objectif du projet](#objectif-du-projet)
2. [Enjeux métier](#enjeux-métier)
3. [Partie 1/2 — Modélisation & Tracking MLflow](#partie-12--modélisation--tracking-mlflow)
4. [Partie 2/2 — Déploiement API (FastAPI)](#partie-22--déploiement-api-fastapi)
5. [Reproductibilité](#reproductibilité)
6. [Structure du projet](#structure-du-projet)
7. [Couche Core & Base de données](#couche-core--base-de-données)
8. [Déploiement](#déploiement)
9. [Installation rapide](#installation-rapide)
10. [Lancer MLflow UI](#lancer-mlflow-ui-optionnel)
11. [Lancer l'API en local](#lancer-lapi-en-local)
12. [Exemple d'appel API](#exemple-dappel-api)
13. [Chargement du modèle](#chargement-du-modèle-local-ou-hugging-face)
14. [Tests unitaires](#tests-unitaires)
15. [Run (Production / Hugging Face Space)](#run-production--hugging-face-space)
16. [Secrets & Sécurité](#secrets--sécurité)
17. [Validation des entrées API](#validation-des-entrées-api)
18. [Monitoring opérationnel](#monitoring-opérationnel)
19. [Logging structuré](#logging-structuré)
20. [Monitoring Data (Drift)](#monitoring-data-drift)
21. [Résultat final](#résultat-final)

---

## Objectif du projet

L'entreprise *Prêt à Dépenser* propose des crédits à la consommation à des clients ayant peu ou pas d'historique bancaire.

L'objectif est de construire un modèle capable de :

-  Prédire la **probabilité de défaut**
-  Produire une décision automate (**ACCEPTED / REFUSED**)
-  Respecter une contrainte métier : **FN >> FP**

---

## Enjeux métier

Le dataset est fortement déséquilibré et les erreurs n'ont pas le même impact :

- **FN (mauvais client prédit bon)** = 💰 perte financière importante
- **FP (bon client prédit mauvais)** = 📉 manque à gagner

### Fonction de coût métier

```
Coût = 10 × FN + 1 × FP
```

---

## Partie 1/2 — Modélisation & Tracking MLflow

###  Travaux réalisés

-  Fusion et agrégation de tables multi-sources
-  Dataset initial : **1658 variables**
-  Benchmark de plusieurs modèles :
  - Logistic Regression
  - Random Forest
  - LightGBM
  - XGBoost
  - CatBoost
  - MLP
-  Sélection progressive des variables : **1658 → 125 features**
-  Tuning d'hyperparamètres
-  Optimisation du seuil métier
-  Interprétabilité avec SHAP (globale + locale)

###  Modèle final retenu

**CatBoostClassifier** - Meilleur coût métier avec stabilité des performances

---

## Partie 2/2 — Déploiement API (FastAPI)

L'API prédit le risque de défaut à partir d'un **identifiant client (client_id)**.

###  Flux de prédiction

```
1. Récupération des features depuis la base (features_store)
   ↓
2. Application du modèle CatBoost
   ↓
3. Retour score + décision
```

###  Endpoints disponibles

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | État de disponibilité |
| `POST` | `/predict` | Prédiction à partir d'un client_id |

---

## Reproductibilité

Exécutez les notebooks dans l'ordre :

```
notebooks/
├── 01_data_preparation/     → préparation et agrégation des tables
├── 02_benchmark/            → benchmark des modèles
├── 03_modeling/             → sélection features, tuning, seuil métier
└── 04_preparation_API/      → génération du modèle et artifacts
```

---

## Structure du projet
```
pret-a-depenser/
│
├── app/                           # API FastAPI (couche production)
│   ├── main.py                    # Point d'entrée de l'API FastAPI
│   ├── config.py                  # Configuration de l'application (variables d'environnement, chemins)
│   ├── schemas.py                 # Définition des schémas Pydantic pour la validation des données
│   ├── model/                     # Dossier contenant le modèle de machine learning
│   │   ├── loader.py              # Script pour charger le modèle (local ou Hugging Face)
│   │   └── predict.py             # Script pour effectuer des prédictions avec le modèle
│   ├── utils/                     # Fonctions utilitaires pour l'API
│   │   ├── validation.py          # Validation des entrées utilisateur
│   │   ├── errors.py              # Gestion des erreurs et exceptions
│   │   └── io.py                  # Gestion des fichiers (lecture/écriture)
│   └── assets/                    # Artifacts locaux (modèle, seuils, etc.) - ignorés par Git
│
├── core/                          # Accès à la base de données et gestion des requêtes SQL
│   ├── config.py                  # Configuration de la base de données
│   └── db/                        # Dossier pour les interactions avec PostgreSQL
│       ├── conn.py                # Gestion de la connexion à la base de données
│       ├── repo_features_store.py # Requêtes pour récupérer les features des clients
│       ├── repo_prod_requests.py  # Requêtes pour logger les prédictions en production
│       ├── repo_ref_dist.py       # Requêtes pour gérer les distributions de référence (monitoring)
│       ├── migrations/            # Scripts SQL pour initialiser ou migrer la base
│       └── sql/                   # Requêtes SQL paramétrées
│
├── monitoring/                    # Dashboard Streamlit pour le monitoring
│   └── streamlit_app.py           # Application Streamlit pour visualiser les métriques et le drift
│
├── src/                           # Scripts de modélisation et préparation des données
│   ├── data/                      # Scripts pour la préparation des données
│   ├── modeling/                  # Scripts pour entraîner et évaluer les modèles
│   ├── tracking/                  # Scripts pour le suivi des expériences avec MLflow
│   └── utils/                     # Fonctions utilitaires pour la modélisation
│
├── notebooks/                     # Notebooks Jupyter pour le pipeline complet
│   ├── 01_data_preparation/       # Préparation et agrégation des données
│   ├── 02_benchmark/              # Benchmark des modèles
│   ├── 03_modeling/               # Sélection des features, tuning, seuil métier
│   └── 04_preparation_API/        # Génération des artifacts pour l'API
|   |__05_monitoring_drift/

│
├── tests/                         # Tests unitaires pour valider le fonctionnement du projet
├── scripts/                       # Scripts utilitaires pour des tâches spécifiques
├── artifacts/                     # Artifacts générés (modèle, seuils, features) - non versionnés
├── data/                          # Données locales (non versionnées dans Git)
│
├── docker-compose.yml             # Configuration Docker pour PostgreSQL en local
├── Dockerfile                     # Fichier Docker pour containeriser l'API
├── pyproject.toml                 # Fichier de configuration des dépendances Python
├── pytest.ini                     # Configuration pour les tests avec pytest
├── .env                           # Fichier pour les variables d'environnement locales
└── README.md                      # Documentation du projet
```

---

## Couche Core & Base de données

La couche *core/* gère l'accès base de données et les requêtes production.

###  Structure

```bash
core/
├── config.py
└── db/
    ├── conn.py                    # Gestion connexion PostgreSQL
    ├── repo_features_store.py     # Récupération features par client_id
    ├── repo_prod_requests.py      # Logging requêtes production
    ├── repo_ref_dist.py           # Stockage distributions référence (drift)
    ├── migrations/                # Scripts SQL init base
    └── sql/                       # Requêtes paramétrées
```

###  Tables PostgreSQL

La base stocke :

| Table | Contenu |
|-------|---------|
| `features_store` | Features clients |
| `prod_requests` | Requêtes de prédiction + scores + latence |
| `ref_feature_dist` | Distributions de référence (monitoring drift) |

###  Connexion

```bash
DATABASE_URL=postgresql://user:password@host:port/dbname
```

###  Initialisation de la base

Les migrations SQL sont situées dans :

```
core/db/migrations/
```

Exécution manuelle via PostgreSQL en environnement local.


### Scripts d’administration de la base et de monitoring
Le dossier scripts/ contient les outils opérationnels permettant :
- d’initialiser la base
- d’alimenter le feature store
- de construire les distributions de référence (drift)
- de simuler du trafic production

#### Charger les features en base
Remplit la table *features_store* à partir d’un CSV API-ready.
```bash
python scripts/01_load_features_store.py --csv examples/X_api.csv
```
- Exécute automatiquement la migration SQL si nécessaire
- Insert ou update via UPSERT
- Stockage en JSONB

#### Contruire la distribution de référence (drift)
Calcule les distributions de référence utilisées pour le PSI.
```bash
python scripts/02_build_reference_dist.py --csv examples/X_api.csv
```
- Génère les bins numériques
- Stocke les proportions catégorielles
- Remplit la table ref_feature_dist

#### Simuler des requêtes production
Permet de générer du trafic afin de :
- remplir prod_requests
- tester la latence
- alimenter le monitoring

**Exécution locale**
```bash
python scripts.03_simulate_requests --base-url "http://127.0.0.1:8000" --csv examples/X_api.csv --n 2000
```
**Exécution HF**
```bash
python scripts.03_simulate_requests --base-url "https://donizetti-yoann-pret-a-depenser-api.hf.space"  --csv examples/X_api.csv --n 2000
```
---

## Déploiement

###  Architecture de déploiement

Le projet est déployé en **deux espaces Hugging Face distincts** :

#### 1️. API Space (Docker)

```
✓ SDK : docker
✓ Port : 7860
✓ Charge : modèle + DB
✓ Endpoint : /predict
✓ Déploiement : GitHub Actions (CD automatique)
```

#### 2️ Monitoring Space (Streamlit)

```
✓ SDK : streamlit
✓ Dashboard : drift + métriques
✓ Connexion : PostgreSQL
✓ Visualisation : PSI + latence + taux erreur
✓ Sync : automatique via GitHub Actions
```

---

## Installation rapide

Créer un environnement virtuel puis installer les dépendances :

###  - API seule (prod / HF API Space)

```bash
pip install .
```

###  - API + PostgreSQL + tests (dev complet)

```bash
pip install ".[dev]"
docker-compose up -d
```

Démarre :
- PostgreSQL (logging production)
- Base de monitoring locale

### -  Notebooks + MLflow

```bash
pip install ".[notebooks]"
```

### -  Monitoring (dashboard Streamlit)

```bash
pip install ".[monitoring]"
```

### -  Tout (dev + notebooks + monitoring)

```bash
pip install ".[dev,notebooks,monitoring]"
```

---

## Lancer MLflow UI (optionnel)

Visualisez les runs et comparez les expérimentations :

```bash
mlflow ui
```

Puis ouvrir : **http://127.0.0.1:5000**

---

## Lancer l'API en local

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Accès

| Ressource | URL |
|-----------|-----|
| API | http://127.0.0.1:8000 |
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |

---

## Exemple d'appel API

###  Healthcheck

```bash
curl http://127.0.0.1:8000/health
```

###  Prédiction

**Payload minimal :**

```json
{
  "client_id": 123456
}
```

**Commande :**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"client_id": 123456}'
```

**Réponse :**

```json
{
  "client_id": 123456,
  "score": 0.35,
  "decision": "ACCEPTED",
  "latency_ms": 45.2
}
```

---

## Chargement du modèle (local ou Hugging Face)

L'API supporte deux modes de chargement des artifacts :

###  Mode local

Les fichiers sont stockés dans : `app/assets/` **(gitignored)**

**Variables d'environnement :**

```bash
BUNDLE_SOURCE=local
LOCAL_MODEL_PATH=app/assets/model/catboost_model.pkl
LOCAL_KEPT_PATH=app/assets/model/kept_features.pkl
LOCAL_CAT_PATH=app/assets/model/cat_features.json
LOCAL_THRESHOLD_PATH=app/assets/model/threshold.pkl
DATABASE_URL=postgresql://user:password@host:port/dbname
```

###  Mode Hugging Face

Les fichiers sont téléchargés depuis un repo Hugging Face.

**Variables d'environnement :**

```bash
BUNDLE_SOURCE=hf
HF_REPO_ID=your-username/your-repo
HF_TOKEN=hf_xxxxxxxxxxxxx  # optionnel si repo public
HF_MODEL_PATH=artifacts/catboost_model.pkl
HF_KEPT_PATH=artifacts/kept_features.pkl
HF_CAT_PATH=artifacts/cat_features.json
HF_THRESHOLD_PATH=artifacts/threshold.pkl
```

---

## Tests unitaires

Lancer les tests :

```bash
pytest --cov=app
```

Génère un rapport de coverage dans :

```
htmlcov/index.html
```

---

## Run (Production / Hugging Face Space)

###  Lancer en local

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger : **http://127.0.0.1:8000/docs**

###  Lancer via Docker

```bash
docker build -f Dockerfile -t pad-api .
docker run -p 7860:7860 pad-api
```

###  Déploiement automatique

Le pipeline **GitHub Actions** exécute :

```
1. Tests unitaires + coverage
   ↓
2. Build Docker image
   ↓
3. Synchronisation vers Space API
   ↓
4. Synchronisation vers Space Monitoring
```

**Le déploiement est conditionné à la réussite complète des tests.**

---

## Secrets & Sécurité

###  Variables d'environnement

Configurez via :

- **Hugging Face Spaces** → Settings → Variables and secrets
- **GitHub** → Settings → Secrets and variables → Actions

**Exemples :**

```
BUNDLE_SOURCE
HF_REPO_ID
HF_MODEL_PATH
HF_KEPT_PATH
HF_CAT_PATH
HF_THRESHOLD_PATH
DATABASE_URL
```

###  Gestion des tokens

- Le token Hugging Face (**HF_TOKEN**) est stocké **uniquement en tant que Secret**
-  **Aucun token** n'est présent dans le code source ou GitHub

---

## Validation des entrées API

Les inputs sont validés via :

- **Pydantic** (schéma d'entrée)
- **validate_payload()** (contrôle strict)

Le système rejette automatiquement :

-  Champs manquants
-  Mauvais types
-  Champs inconnus (protection contre payload invalide / injection)

---

## Monitoring opérationnel

###  Suivi en temps réel

| Métrique | Description |
|----------|-------------|
| **Taux d'erreur HTTP** | % d'erreurs 4xx/5xx |
| **Latence médiane** | Temps moyen de réponse |
| **p95 / p99** | Percentiles de latence |
| **Volume requêtes** | Nombre de prédictions/jour |
| **Distribution ACCEPTED/REFUSED** | Ratio acceptations/refus |

Les données proviennent directement de la **table `prod_requests`**.

---

## Logging structuré

Chaque requête `/predict` génère un log structuré contenant :

```json
{
  "timestamp": "2026-02-24T10:30:45.123Z",
  "client_id": 123456,
  "score": 0.35,
  "decision": "ACCEPTED",
  "latency_ms": 45.2,
  "status_code": 200
}
```

###  Utilisation

-  Persistés en base
-  Exploitables pour audit
-  Utilisables pour monitoring

---

## Monitoring Data (Drift)

###  Comparaison

```
Distribution production (features_store)
         ↓
Comparaison
         ↓
Distribution référence (ref_feature_dist)
```

###  Calcul

**Population Stability Index (PSI)**

```
PSI = Σ (production% - référence%) × ln(production% / référence%)
```

###  Interprétation

| PSI | Niveau | Action |
|-----|--------|--------|
| < 0.1 |  Stable | Aucune |
| 0.1 – 0.25 |  Dérive modérée | Alerte |
| > 0.25 |  Drift significatif | Investigation |

###  Lancer le monitoring

```bash
streamlit run monitoring/streamlit_app.py
```

---

## Load Testing

Un test de charge simulé a été réalisé afin d’évaluer la stabilité sous contrainte.(2000 requetes)
### Exécution locale
```bash
python scripts.03_simulate_requests --base-url "http://127.0.0.1:8000" --csv examples/X_api.csv --n 2000
```
### Exécution HF
```bash
python scripts.03_simulate_requests --base-url "https://donizetti-yoann-pret-a-depenser-api.hf.space"  --csv examples/X_api.csv --n 2000
```

## Optimisations post-déploiement
### Identification des goulots d’étranglement

Les métriques issues du monitoring (*table prod_requests*) ont montré que la latence provenait principalement de :
- Conversion payload → pandas.DataFrame
- Casting répété des variables catégorielles
- Overhead Python inutile pour une seule ligne d’inférence

Profiling réalisé via cProfile.

### Optimisations mises en œuvre
- Suppression complète de pandas côté inference
- Construction directe du vecteur d’entrée via build_row (liste Python native)
- Pré-calcul des colonnes catégorielles au démarrage de l’API
- thread_count=1 pour stabiliser la latence en environnement CPU partagé (HF Spaces)

### Résultats mesurés

| Métrique            | Avant | Après | Gain   |
|---------------------|-------|-------|--------|
| p50 latency (ms)    | 9,64ms   | 2,85ms    | - 70,44%   |
| p95 latency (ms)    | 11,22ms    | 3,97ms  | - 64,6%   |
| inference_p50 (ms)   | 7,56ms    | 1,03   | --86,4%   |
 inference_p96 (ms)   | 8,86   | 1,57    | -82,3%   |


Latence divisée par ~3 sans modification du modèle.
Les optimisations portent uniquement sur la couche d’inférence API.

## Résultat final

Ce projet fournit une **architecture MLOps complète** avec :

###  Modélisation

-  Modèle de scoring optimisé selon un coût métier
-  Tracking complet via **MLflow**
-  Interprétabilité avec **SHAP**

###  Déploiement

-  **API FastAPI** robuste
-  Base de données **PostgreSQL** de logging production
-  **Containerisation Docker**
-  **CI/CD automatisé** (GitHub Actions)

###  Monitoring

-  Monitoring technique : latence / erreurs
-  Monitoring data : drift via **PSI**
-  **Dashboard Streamlit** interactif

###  Architecture

-  2 environnements isolés (API + Monitoring)
-  Déploiement sur **Hugging Face Spaces**
-  Gestion sécurisée des secrets

---



