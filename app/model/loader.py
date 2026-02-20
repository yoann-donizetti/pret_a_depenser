from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

from huggingface_hub import hf_hub_download
from catboost import CatBoostClassifier

from app.utils.io import load_txt_list, parse_json


def _load_catboost_from_file(model_file: Path) -> CatBoostClassifier:
    """
    Charge un modèle de classifieur CatBoost à partir d'un fichier.

    Args:
        model_file (Path): Le chemin du fichier du modèle CatBoost enregistré.

    Returns:
        CatBoostClassifier: Le modèle de classifieur CatBoost chargé.
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