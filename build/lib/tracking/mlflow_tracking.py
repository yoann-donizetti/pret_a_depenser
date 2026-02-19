from pathlib import Path
import mlflow
def get_or_create_experiment(name: str, artifact_root: Path) -> str:
    """
    Ensure the experiment exists AND stores artifacts under artifact_root.
    Returns the experiment_id.
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            name=name,
            artifact_location=artifact_root.as_uri(),  # file://... (compatible Windows/Linux)
        )
        return exp_id

    # Si l'expérience existe déjà, on ne peut pas changer artifact_location via l'API MLflow.
    # On avertit juste si elle ne pointe pas vers ton dossier artifacts.
    expected = artifact_root.as_uri()
    if exp.artifact_location != expected:
        print(
            "[WARN] L'expérience existe déjà avec un artifact_location différent:\n"
            f"  actuel   = {exp.artifact_location}\n"
            f"  attendu  = {expected}\n"
            "=> Pour forcer, crée une nouvelle expérience (nouveau nom) ou supprime/recrée celle-ci."
        )
    return exp.experiment_id