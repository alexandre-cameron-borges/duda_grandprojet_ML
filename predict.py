import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
import streamlit as st

# --- Configuration ---
# Votre identifiant de dépôt sur Hugging Face
REPO_ID = "alexandre-cameron-borges/churn"
# Le nom du fichier de votre modèle dans le dépôt
FILENAME = "churn_prediction_model.joblib"

# --- Chargement du modèle (avec gestion des erreurs et cache) ---
@st.cache_resource
def load_model_from_hf():
    """
    Charge le modèle depuis Hugging Face Hub de manière sécurisée.
    Retourne l'objet modèle ou None en cas d'erreur.
    """
    try:
        st.write(f"⏳ Tentative de téléchargement du modèle depuis '{REPO_ID}'...")
        # Télécharge le fichier du modèle
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        st.write(f"✅ Modèle téléchargé avec succès : '{model_path}'")
        
        st.write("⏳ Tentative de chargement du modèle en mémoire...")
        # Charge le modèle avec joblib
        model = joblib.load(model_path)
        st.write("✅ Modèle chargé avec succès.")
        return model

    except Exception as e:
        # Affiche une erreur claire dans l'application et dans les logs
        st.error(f"Erreur critique lors du chargement du modèle : {e}")
        return None

# On charge le modèle au démarrage de l'application
model = load_model_from_hf()

def make_prediction(input_data):
    """
    Fait une prédiction en utilisant le modèle chargé.
    'input_data' doit être un DataFrame pandas avec les bonnes colonnes.
    Retourne la probabilité de churn (classe 1).
    """
    if model is not None:
        try:
            # S'assurer que les colonnes sont dans le bon ordre
            # Le pipeline de pré-traitement s'occupe de la transformation
            prediction_proba = model.predict_proba(input_data)
            
            # Retourne la probabilité de la classe positive (churn = 1)
            return prediction_proba[0][1]
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return None
    else:
        st.error("Le modèle n'est pas disponible, impossible de faire une prédiction.")
        return None
