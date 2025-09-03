import streamlit as st
import pandas as pd
from predict import make_prediction

# --- Configuration de la page ---
st.set_page_config(
    page_title="Prédiction de Churn Client",
    page_icon="👋",
    layout="wide"
)

# --- Barre latérale pour la saisie des données ---
st.sidebar.header("Paramètres du Client")

def user_input_features():
    """
    Crée les widgets Streamlit pour la saisie des données utilisateur.
    """
    tenure = st.sidebar.slider("Ancienneté (mois)", 0, 70, 10, key="tenure")
    satisfaction_score = st.sidebar.slider("Score de Satisfaction", 1, 5, 3, key="satisfaction_score")
    hour_spend_on_app = st.sidebar.slider("Heures passées sur l'app/site", 0.0, 10.0, 3.0, 0.1, key="hour_spend_on_app")
    day_since_last_order = st.sidebar.slider("Jours depuis la dernière commande", 0, 50, 5, key="day_since_last_order")
    cashback_amount = st.sidebar.slider("Montant du Cashback", 0.0, 500.0, 150.0, 1.0, key="cashback_amount")
    order_amount_hike_from_last_year = st.sidebar.slider("Augmentation des commandes (%)", 0.0, 50.0, 15.0, 0.5, key="order_amount_hike")
    coupon_used = st.sidebar.slider("Coupons utilisés", 0, 20, 1, key="coupon_used")
    order_count = st.sidebar.slider("Nombre de commandes", 0, 20, 2, key="order_count")
    warehouse_to_home = st.sidebar.slider("Distance entrepôt-maison", 5, 50, 15, key="warehouse_to_home")
    
    complain = st.sidebar.selectbox("Réclamation (dernier mois)", [0, 1], key="complain")
    gender = st.sidebar.selectbox("Genre", ["Male", "Female"], key="gender")
    marital_status = st.sidebar.selectbox("Statut Marital", ["Single", "Married", "Divorced"], key="marital_status")
    preferred_payment_mode = st.sidebar.selectbox("Mode de Paiement Préféré", ['Debit Card', 'Credit Card', 'E wallet', 'Cash on Delivery', 'UPI'], key="payment_mode")
    preferred_order_cat = st.sidebar.selectbox("Catégorie Préférée", ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'], key="order_cat")
    preferred_login_device = st.sidebar.selectbox("Appareil de Connexion", ['Mobile Phone', 'Computer'], key="login_device")
    
    city_tier = st.sidebar.selectbox("Catégorie de Ville", [1, 2, 3], key="city_tier")
    number_of_device_registered = st.sidebar.selectbox("Nombre d'appareils enregistrés", [1, 2, 3, 4, 5, 6], key="num_devices")
    number_of_address = st.sidebar.selectbox("Nombre d'adresses", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key="num_address")

    # Création du DataFrame avec les noms de colonnes corrects
    data = {
        'Tenure': tenure,
        'PreferredLoginDevice': preferred_login_device,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'PreferredPaymentMode': preferred_payment_mode,
        'Gender': gender,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': number_of_device_registered,
        'PreferredOrderCat': preferred_order_cat,
        'SatisfactionScore': satisfaction_score,
        'MaritalStatus': marital_status,
        'NumberOfAddress': number_of_address,
        'Complain': complain,
        'OrderAmountHikeFromLastYear': order_amount_hike_from_last_year,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount,
        # On peut laisser une valeur par défaut pour Cluster_RFM car ce n'est pas une entrée utilisateur directe
        'Cluster_RFM': 0 
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# --- Interface principale ---
st.title("Prédiction du Churn Client 🔮")
st.write("""
Cette application prédit la probabilité qu'un client résilie son abonnement (churn).
Utilisez les options dans la barre latérale pour ajuster les caractéristiques du client et voir l'impact sur le risque de churn.
""")

# Récupérer les données de l'utilisateur
input_df = user_input_features()

# Afficher les données d'entrée
st.subheader("Caractéristiques du client sélectionnées")
st.write(input_df)

# Bouton pour lancer la prédiction
if st.button("Lancer la Prédiction", key="predict_button"):
    try:
        prediction_proba, prediction = make_prediction(input_df)
        
        st.subheader("Résultat de la Prédiction")
        
        # Affichage avec des colonnes pour une meilleure mise en page
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("Risque de Churn : Élevé")
            else:
                st.success("Risque de Churn : Faible")
        
        with col2:
            st.metric(label="Probabilité de Churn", value=f"{prediction_proba:.2%}")
            
        if prediction == 1:
            st.warning("Actions recommandées : Envisager une offre promotionnelle, un contact proactif du service client ou une enquête de satisfaction pour comprendre les points de friction.")

    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")

