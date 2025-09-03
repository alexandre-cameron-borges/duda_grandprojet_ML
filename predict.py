import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# --- Chargement du modèle ---
# Le modèle est chargé une seule fois au démarrage du script.
REPO_ID = "alexandre-cameron-borges/churn" # Remplacé par votre repo_id
FILENAME = "churn_prediction_model.joblib"

try:
    model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
except Exception as e:
    # Si le modèle ne peut pas être chargé, on met une variable à None
    # L'application Streamlit pourra gérer cette erreur.
    model = None
    print(f"Erreur lors du chargement du modèle : {e}")


def make_prediction(input_data: pd.DataFrame) -> float:
    """
    Fait une prédiction de churn en utilisant le modèle chargé.

    Args:
        input_data (pd.DataFrame): Un DataFrame contenant les données d'un client.
                                   Les colonnes doivent correspondre à celles attendues
                                   par le modèle.

    Returns:
        float: La probabilité de churn (entre 0 et 1).
    """
    if model is None:
        raise RuntimeError("Le modèle n'a pas pu être chargé. Impossible de faire une prédiction.")

    # Prédiction des probabilités (classe 1 = churn)
    prediction_proba = model.predict_proba(input_data)[:, 1]
    
    # Retourne la première (et unique) probabilité
    return prediction_proba[0]

