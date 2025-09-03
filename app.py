import streamlit as st
import pandas as pd
from predict import make_prediction, model # On importe la fonction et la variable 'model'

# --- Configuration de la page ---
st.set_page_config(
    page_title="Prédiction de Churn Client",
    page_icon="🔮",
    layout="wide"
)

# --- Interface Utilisateur ---
st.title("Prédiction du Churn Client 🔮")
st.markdown("""
    Cette application utilise un modèle de Machine Learning (Random Forest) pour prédire si un client
    est susceptible de résilier son abonnement. Entrez les caractéristiques les plus importantes du client
    ci-dessous pour obtenir une prédiction.
""")

st.divider()

# On vérifie si le modèle a bien été chargé depuis predict.py
if model is not None:
    # Création de colonnes pour une mise en page claire du formulaire
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Comportement & Ancienneté")
        tenure = st.slider("Ancienneté du client (en mois)", 0, 70, 10, key="tenure")
        hour_spend_on_app = st.slider("Heures moyennes passées sur l'app", 0.0, 10.0, 2.5, 0.1, key="hours")
        day_since_last_order = st.slider("Jours depuis la dernière commande", 0, 50, 5, key="last_order")
    
    with col2:
        st.subheader("Satisfaction & Service")
        satisfaction_score = st.select_slider(
            "Score de satisfaction",
            options=[1, 2, 3, 4, 5],
            value=3, key="satisfaction"
        )
        complain = st.selectbox("A déposé une réclamation (dernier mois) ?", (0, 1), format_func=lambda x: "Oui" if x == 1 else "Non", key="complain")
        order_count = st.number_input("Nombre de commandes (dernier mois)", min_value=0, value=2, step=1, key="order_count")

    with col3:
        st.subheader("Informations personnelles")
        marital_status = st.selectbox("Statut marital", options=["Single", "Married", "Divorced"], key="marital")
        gender = st.selectbox("Genre", options=["Male", "Female"], key="gender")
        number_of_address = st.number_input("Nombre d'adresses enregistrées", min_value=1, value=2, step=1, key="address")


    # Bouton pour lancer la prédiction
    if st.button("Lancer la Prédiction", type="primary", use_container_width=True):
        
        # Création du DataFrame avec les entrées utilisateur
        # Les autres colonnes moins importantes sont fixées à des valeurs moyennes pour simplifier l'interface
        input_data = pd.DataFrame({
            'Tenure': [tenure],
            'PreferredLoginDevice': ['Mobile Phone'],
            'CityTier': [1],
            'WarehouseToHome': [20.0],
            'PreferredPaymentMode': ['Credit Card'],
            'Gender': [gender],
            'HourSpendOnApp': [hour_spend_on_app],
            'NumberOfDeviceRegistered': [3],
            'PreferredOrderCat': ['Laptop & Accessory'],
            'SatisfactionScore': [satisfaction_score],
            'MaritalStatus': [marital_status],
            'NumberOfAddress': [number_of_address],
            'Complain': [complain],
            'OrderAmountHikeFromLastYear': [15.0],
            'CouponUsed': [1.0],
            'OrderCount': [order_count],
            'DaySinceLastOrder': [day_since_last_order],
            'CashbackAmount': [150.0],
            'Cluster_RFM': [0]
        })
        
        st.write("---")
        
        # Appel de la fonction de prédiction
        churn_probability = make_prediction(input_data)

        # Affichage du résultat
        st.subheader("Résultat de la Prédiction")
        
        if churn_probability > 0.5:
            st.error(f"🔴 Risque de Churn Élevé (Probabilité : {churn_probability:.2%})")
            st.warning("Actions recommandées : Contacter le client, proposer une offre promotionnelle, ou analyser les réclamations récentes pour comprendre son insatisfaction.")
        else:
            st.success(f"✅ Risque de Churn Faible (Probabilité de churn : {churn_probability:.2%})")
            st.info("Actions recommandées : Maintenir une bonne relation client, proposer des produits complémentaires via des newsletters ciblées.")
else:
    st.error("❌ Le modèle n'a pas pu être chargé. L'application ne peut pas fonctionner. Veuillez vérifier le fichier predict.py et la connexion à Hugging Face Hub.")
