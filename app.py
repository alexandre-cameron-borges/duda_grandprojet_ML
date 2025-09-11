import streamlit as st
import pandas as pd
# Assurez-vous d'avoir le fichier predict.py avec la fonction make_prediction dans le même dossier
from predict import make_prediction

# --- Configuration de la page ---
st.set_page_config(
    page_title="Prédiction du Churn Client Ecommerce SDUDA25",
    page_icon="🔮",
    layout="wide"
)

# --- Interface principale ---
st.title("Prédiction du Churn Client Ecommerce SDUDA25 🔮")
st.write("""
Cette application de Alexandre, Ryad & Jordan prédit la probabilité qu'un client résilie son abonnement (churn).
Ajustez les caractéristiques du client ci-dessous et cliquez sur "Lancer la Prédiction" pour voir le résultat.
""")
st.markdown("---")


# --- Section pour la saisie des données au centre ---
st.subheader("Paramètres du Client")

# On utilise des colonnes pour organiser les widgets
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Comportement & Satisfaction")
    tenure = st.slider("Ancienneté (mois)", 0, 70, 10, key="tenure")
    satisfaction_score = st.slider("Score de Satisfaction", 1, 5, 3, key="satisfaction_score")
    hour_spend_on_app = st.slider("Heures passées sur l'app/site", 0.0, 10.0, 3.0, 0.1, key="hour_spend_on_app")
    day_since_last_order = st.slider("Jours depuis la dernière commande", 0, 50, 5, key="day_since_last_order")
    cashback_amount = st.slider("Montant du Cashback", 0.0, 500.0, 150.0, 1.0, key="cashback_amount")
    complain = st.selectbox("Réclamation (dernier mois)", [0, 1], key="complain", help="0 = Non, 1 = Oui")


with col2:
    st.markdown("##### Habitudes d'Achat")
    order_amount_hike_from_last_year = st.slider("Augmentation des commandes (%)", 0.0, 50.0, 15.0, 0.5, key="order_amount_hike")
    coupon_used = st.slider("Coupons utilisés", 0, 20, 1, key="coupon_used")
    order_count = st.slider("Nombre de commandes", 0, 20, 2, key="order_count")
    warehouse_to_home = st.slider("Distance entrepôt-maison (km)", 5, 50, 15, key="warehouse_to_home")
    preferred_payment_mode = st.selectbox("Mode de Paiement Préféré", ['Debit Card', 'Credit Card', 'E wallet', 'Cash on Delivery', 'UPI'], key="payment_mode")
    preferred_order_cat = st.selectbox("Catégorie Préférée", ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'], key="order_cat")

with col3:
    st.markdown("##### Informations Personnelles & Techniques")
    gender = st.selectbox("Genre", ["Male", "Female"], key="gender")
    marital_status = st.selectbox("Statut Marital", ["Single", "Married", "Divorced"], key="marital_status")
    city_tier = st.selectbox("Catégorie de Ville", [1, 2, 3], key="city_tier")
    preferred_login_device = st.selectbox("Appareil de Connexion", ['Mobile Phone', 'Computer'], key="login_device")
    number_of_device_registered = st.selectbox("Nombre d'appareils enregistrés", [1, 2, 3, 4, 5, 6], key="num_devices")
    number_of_address = st.selectbox("Nombre d'adresses", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key="num_address")


# --- Création du DataFrame pour la prédiction ---
# Cette partie est maintenant directement dans le script principal
data = {
    'Tenure': tenure,
    'PreferredLoginDevice': preferred_login_device,
    'CityTier': city_tier,
    'WarehouseToHome': warehouse_to_home,
    'PreferredPaymentMode': preferred_payment_mode,
    'Gender': gender,
    'HourSpendOnApp': hour_spend_on_app,
    'NumberOfDeviceRegistered': number_of_device_registered,
    'PreferedOrderCat': preferred_order_cat, # CORRIGÉ: Un seul 'r'
    'SatisfactionScore': satisfaction_score,
    'MaritalStatus': marital_status,
    'NumberOfAddress': number_of_address,
    'Complain': complain,
    'OrderAmountHikeFromlastYear': order_amount_hike_from_last_year, # CORRIGÉ: 'l' minuscule
    'CouponUsed': coupon_used,
    'OrderCount': order_count,
    'DaySinceLastOrder': day_since_last_order,
    'CashbackAmount': cashback_amount,
    # On peut laisser une valeur par défaut pour Cluster_RFM car ce n'est pas une entrée utilisateur directe
    'Cluster_RFM': 0
}
input_df = pd.DataFrame(data, index=[0])


# Afficher les données d'entrée dans un expander pour ne pas surcharger l'interface
with st.expander("Voir les caractéristiques du client sélectionnées"):
    st.write(input_df)

st.markdown("---")

# Bouton pour lancer la prédiction, centré
# Pour centrer le bouton, nous utilisons des colonnes vides comme marges
_, col_btn, _ = st.columns([2, 1, 2])
with col_btn:
    predict_button = st.button("Lancer la Prédiction", key="predict_button", use_container_width=True)


# --- Section des Résultats ---
if predict_button:
    try:
        # On récupère une seule valeur (la probabilité)
        prediction_proba = make_prediction(input_df)

        # On déduit la classe de prédiction (0 ou 1) à partir de la probabilité
        prediction = 1 if prediction_proba > 0.5 else 0

        st.subheader("Résultat de la Prédiction")

        # Affichage avec des colonnes pour une meilleure mise en page
        col1_res, col2_res = st.columns(2)

        with col1_res:
            if prediction == 1:
                st.error("### Risque de Churn : Élevé")
            else:
                st.success("### Risque de Churn : Faible")

        with col2_res:
            st.metric(label="Probabilité de Churn", value=f"{prediction_proba:.2%}")

        if prediction == 1:
            st.warning("**Actions recommandées :** Envisager une offre promotionnelle, un contact proactif du service client ou une enquête de satisfaction pour comprendre les points de friction.")

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")

