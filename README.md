# Prédiction du Churn Client - Projet de Machine Learning - SDUDA25
Plateforme e-commerce — prédire la probabilité de résiliation (churn), comprendre **pourquoi** et **agir** via une app web interactive et des playbooks business.

**Présentation (slides)** : https://docs.google.com/presentation/d/1RMP6nOubCtmiaKo5-O1d-b62YopHXdX40PQx8i74r8E/edit?usp=sharing  
**App web (Streamlit)** : https://acb-churn-dudagrandprojetml.streamlit.app/

**Code Google Colab** : https://colab.research.google.com/drive/1nhwqL_EXHR2y_li6RrHbbJ-c9S6keSon?usp=sharing

---

## 🎯 Objectif du Projet
> Quelles caractéristiques et quels comportements clients contribuent le plus à l’attrition, et comment utiliser ces informations pour la **réduire** ?

---

## 📊 Dataset
Kaggle — *E-Commerce Customer Churn Analysis and Prediction*  
Source : https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

**Champs clés**
- **Démographie** : Sexe, Statut marital, Ville (CityTier)
- **Comportement** : **Tenure**, appareil préféré, heures passées sur l’app
- **Achats** : **PreferedOrderCat**, **OrderCount**, **DaySinceLastOrder (Récence)**
- **Satisfaction & Care** : **SatisfactionScore**, **Complain**
- **Logistique & Paiement** : **WarehouseToHome**, **PreferredPaymentMode**, **NumberOfAddress**
- **Monétaire** : **CashbackAmount**
- **RFM proxy** : R = DaySinceLastOrder, F = OrderCount, M = CashbackAmount/OrderCount → **Cluster_RFM**

---

## 🛠️ Méthodologie
1. **EDA** : distributions, corrélations, valeurs manquantes, fuites évitées  
2. **Prétraitement** : `StandardScaler` (num), `OneHotEncoder` (cat) via `ColumnTransformer`  
3. **Modélisation** : `LogisticRegression`, `DecisionTree`, **`RandomForest` (retenu)**  
4. **Validation** : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`  
5. **Interprétabilité** : **SHAP TreeExplainer** (classe 1 = churn)

---

## ✅ Résultats (CV 5-folds)
| Modèle | PR-AUC | ROC-AUC | Balanced Acc | F1 (classe=1) |
|---|---:|---:|---:|---:|
| **RandomForest** | **0,946** | **0,986** | **0,913** | **0,870** |
| DecisionTree | 0,733 | 0,905 | 0,897 | 0,802 |
| LogisticRegression | 0,696 | 0,889 | 0,803 | 0,578 |

**Top drivers SHAP (importance globale, % approx.)**
| Rang | Driver | % |
|---:|---|---:|
| 1 | **Tenure** | **25,8** |
| 2 | **Complain** | **12,0** |
| 3 | **PreferedOrderCat** | **8,8** |
| 4 | **MaritalStatus** | **8,2** |
| 5 | **DaySinceLastOrder** | **6,1** |
| 6 | **CashbackAmount** | **5,8** |
| 7 | **NumberOfAddress** | **4,4** |
| 8 | **PreferredPaymentMode** | **4,3** |
| 9 | **WarehouseToHome** | **4,0** |
| 10 | **SatisfactionScore** | **4,0** |

**Lecture** : le churn se joue tôt (**Tenure**), sur la **qualité de service** (**Complain**), la **récence**, la **catégorie d’achat**, la **logistique/paiement** et la **satisfaction**.

---

## 🧭 Playbooks Business (à tester)
**1) Onboarding “100 jours d’or” (Tenure)**  
- **Ce qu’on fait** : 3 touchpoints **J0/J7/J30** + page **“Mes avantages”** (points/cashback visibles)  
- **Pourquoi** : ancrer la valeur tôt → ↓ churn 60j  

**2) Care “24h ou geste” (Complain)**  
- **Ce qu’on fait** : **TTR < 24h**, **owner unique**, message de résolution **personnalisé** ; **crédit auto** (5–8€) si >24h ; **squads** dédiées logistique/paiement  
- **Pourquoi** : transformer un incident en **fidélité**  

**3) Triggers 14/30/60 + cashback dynamique (Récence/Monétaire)**  
- **Ce qu’on fait** : 14j rappel valeur (sans promo) → 30j **petite offre** → 60j **offre forte** **uniquement** top-risque (score) ; **holdout 10–15%**  
- **Pourquoi** : réactiver sans **cannibaliser** la marge  


---

## 🚀 Application Web (Streamlit)
- **Formulaire** : saisie des features client  
- **Prédiction** : probabilité de churn + interprétabilité locale (SHAP)  
- **Usage** : prioriser les actions (playbooks ci-dessus)

App : https://acb-churn-dudagrandprojetml.streamlit.app/

---

## ⚙️ Lancer le projet en local
```bash
git clone https://github.com/alexandre-cameron-borges/duda_grandprojet_ML.git
cd duda_grandprojet_ML
pip install -r requirements.txt
streamlit run app.py
