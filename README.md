# Prêt à Dépenser — Credit Scoring (MLOps)

Ce projet consiste à développer un modèle de **credit scoring** permettant d’estimer la probabilité de défaut d’un client.  
Il s’inscrit dans une démarche **MLOps complète**, allant du tracking d’expériences (MLflow) jusqu’au déploiement d’une API et au monitoring en production.

Le projet est organisé en deux parties : **Partie 1/2 (MLflow + entraînement)** et **Partie 2/2 (déploiement + monitoring)**.

---

## Partie 1/2 — Etablir un modèle de scoring


Cette première partie du projet s’inscrit dans un scénario professionnel de **credit scoring**.  
L’objectif est de construire un modèle capable d’estimer la probabilité de défaut d’un client afin d’aider à la décision d’octroi de crédit.

### Contexte métier
L’entreprise *Prêt à Dépenser* propose des crédits à la consommation à des clients ayant peu ou pas d’historique bancaire.  
Le modèle doit prédire le risque de faillite (défaut de remboursement) à partir de plusieurs sources de données (données comportementales, historiques financiers, données externes…).

---

### Travaux réalisés

####  Compréhension du besoin métier
- Mise en place d’un modèle de scoring pour prédire la **probabilité de défaut** d’un client.
- Objectif final : automatiser une décision **crédit accordé / refusé**.
- Intégration d’une démarche MLOps (tracking + gestion des versions).

####  Préparation des données multi-sources
- Fusion de plusieurs tables autour d’une table principale (**1 ligne = 1 client**).
- Agrégation des tables secondaires au niveau client (moyenne, somme, min/max, écart-type…).
- Dataset final initial : **1658 variables**.

####  Prise en compte des enjeux métier (FN >> FP)
Le jeu de données est fortement déséquilibré et l’erreur n’a pas le même impact métier :
- **FN (mauvais client prédit bon)** = perte financière importante
- **FP (bon client prédit mauvais)** = manque à gagner

Mise en place d’une fonction de coût métier :
- **Coût = 10 × FN + 1 × FP**

####  Benchmark de modèles et sélection progressive
- Comparaison de plusieurs modèles de classification (Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost, MLP).
- Évaluation via validation croisée stratifiée.
- Gestion du déséquilibre via **pondération des classes**.
- Sélection des modèles les plus performants pour la phase finale.

####  Réduction du périmètre des variables
- Problème initial : modèle lourd et difficile à interpréter.
- Sélection progressive des variables selon leur importance globale.
- Réduction du dataset de **1658 à 125 variables**, afin d’améliorer :
  - la stabilité,
  - le temps de calcul,
  - l’interprétabilité.

####  Benchmark final et choix du modèle
- Entraînement final sur **Train + Valid**
- Évaluation sur le jeu de **Test**
- Le modèle **CatBoost** est retenu car il obtient le **meilleur coût métier** sur test, tout en conservant des performances stables (AUC, recall, F-beta).

####  Interprétabilité avec SHAP
- Analyse SHAP globale (importance globale des variables).
- Analyse SHAP locale (explication au niveau d’un client).
- Identification des variables les plus influentes (notamment `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`).

####  Mise en place de la démarche MLOps avec MLflow
- Tracking automatique des expérimentations (hyperparamètres, métriques, artefacts).
- Utilisation de l’interface **MLflow UI** pour comparer les runs.
- Enregistrement des modèles dans un **Model Registry** (versioning V1, V2, …).
- Sauvegarde des résultats d’interprétabilité (graphiques SHAP) comme artefacts.
---

### Résultat
Cette première partie a permis de produire un modèle de scoring versionné et traçable, prêt à être utilisé comme base pour une mise en production (API + monitoring), réalisée dans la **Partie 2/2**.

--- 

## Partie 2/2 — Confirmez vos compétences en MLOps (Déploiement & Monitoring)


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
