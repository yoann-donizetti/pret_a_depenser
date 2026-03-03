
"""
Module de chargement du modèle CatBoost et de ses artefacts associés (features, catégories, seuil) depuis le disque local ou HuggingFace Hub.

Permet de charger facilement un modèle entraîné, la liste des features conservées, des features catégorielles et le seuil de décision, pour une utilisation en production ou en évaluation.
fonctions principales :
    - load_bundle_from_local: Charge le modèle et ses artefacts depuis des fichiers locaux.     
    - load_bundle_from_hf: Charge le modèle et ses artefacts depuis un dépôt HuggingFace Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

from huggingface_hub import hf_hub_download
from catboost import CatBoostClassifier

from app.utils.io import load_txt_list, parse_json


def _load_catboost_from_file(model_file: Path) -> CatBoostClassifier:
    """
    Charge un modèle CatBoostClassifier à partir d'un fichier local.

    Args:
        model_file (Path): Chemin du fichier modèle CatBoost.

    Returns:
        CatBoostClassifier: Modèle CatBoost chargé.
    

    """
    model = CatBoostClassifier()
    model.load_model(str(model_file))
    return model


def load_bundle_from_local(
    *,
    model_path: Path,
    kept_path: Path,
    cat_path: Path,
    threshold_path: Path,
) -> Tuple[CatBoostClassifier, List[str], List[str], float]:
    """
    Charge le modèle CatBoost et ses artefacts (features, catégories, seuil) depuis des fichiers locaux.

    Args:
        model_path (Path): Chemin du fichier modèle.
        kept_path (Path): Chemin du fichier des features conservées.
        cat_path (Path): Chemin du fichier des features catégorielles.
        threshold_path (Path): Chemin du fichier du seuil de décision.

    Returns:
        Tuple[CatBoostClassifier, List[str], List[str], float]:
            - Modèle CatBoost
            - Liste des features conservées
            - Liste des features catégorielles
            - Seuil de décision (float)
    """
    model_file = Path(model_path)
    kept_file = Path(kept_path)
    cat_file = Path(cat_path)
    thr_file = Path(threshold_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Local model file not found: {model_file}")
    if not kept_file.exists():
        raise FileNotFoundError(f"Local kept features file not found: {kept_file}")
    if not cat_file.exists():
        raise FileNotFoundError(f"Local cat features file not found: {cat_file}")
    if not thr_file.exists():
        raise FileNotFoundError(f"Local threshold file not found: {thr_file}")

    model = _load_catboost_from_file(model_file)
    kept = load_txt_list(kept_file)
    cat = load_txt_list(cat_file)

    thr_obj = parse_json(thr_file.read_text(encoding="utf-8"))
    threshold = float(thr_obj["threshold"])

    return model, kept, cat, threshold


def load_bundle_from_hf(
    *,
    repo_id: str,
    model_path: str,
    kept_path: str,
    cat_path: str,
    threshold_path: str,
    token: str | None = None,
) -> Tuple[CatBoostClassifier, List[str], List[str], float]:
    """
    Charge le modèle CatBoost et ses artefacts depuis un dépôt HuggingFace Hub.

    Args:
        repo_id (str): Identifiant du repo HuggingFace.
        model_path (str): Nom du fichier modèle dans le repo.
        kept_path (str): Nom du fichier des features conservées dans le repo.
        cat_path (str): Nom du fichier des features catégorielles dans le repo.
        threshold_path (str): Nom du fichier du seuil de décision dans le repo.
        token (str | None): Jeton d'accès HuggingFace (optionnel).

    Returns:
        Tuple[CatBoostClassifier, List[str], List[str], float]:
            - Modèle CatBoost
            - Liste des features conservées
            - Liste des features catégorielles
            - Seuil de décision (float)
    """
    model_file = Path(hf_hub_download(repo_id=repo_id, filename=model_path, token=token))
    kept_file = Path(hf_hub_download(repo_id=repo_id, filename=kept_path, token=token))
    cat_file = Path(hf_hub_download(repo_id=repo_id, filename=cat_path, token=token))
    thr_file = Path(hf_hub_download(repo_id=repo_id, filename=threshold_path, token=token))

    model = _load_catboost_from_file(model_file)
    kept = load_txt_list(kept_file)
    cat = load_txt_list(cat_file)

    thr_obj = parse_json(thr_file.read_text(encoding="utf-8"))
    threshold = float(thr_obj["threshold"])

    return model, kept, cat, threshold