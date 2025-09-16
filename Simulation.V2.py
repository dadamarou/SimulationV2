"""
Simulation.V2.py — Dashboard Streamlit pour simulations multi-scénarios
Version vérifiée / sécurisée :
- Validation des entrées (probabilités, impacts, catégories)
- Gestion robuste des suppressions AgGrid via une colonne 'uid' unique
- Contrôles d'usage (scénarios non sélectionnés, limite N recommandée)
- Bloc try/except autour de la simulation avec messages utilisateur
- Résumés narratifs avec st.markdown
- Commentaires pour traçabilité des améliorations
"""

import uuid
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.express as px

st.set_page_config(page_title="Dashboard Simulation Ultime", layout="wide")
st.title("Tableau de bord multi-scénarios ultime (version vérifiée)")

# =========================
# Paramètres généraux
# =========================
# Limite recommandée pour N ; l'utilisateur peut augmenter si nécessaire mais on affiche un avertissement
N = st.number_input(
    "Nombre de simulations (N)",
    min_value=1000,
    max_value=2_000_000,
    value=100_000,
    step=1000,
    help="Plus N est grand, plus les résultats seront stables — attention à la mémoire."
)
if N > 500_000:
    st.warning("Valeur élevée de N (>500k) : cela peut consommer beaucoup de mémoire et rendre l'interface lente.")

# =========================
# Gestion des scénarios dans session_state
# =========================
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

# Impacts et priorité définis (source unique)
IMPACT_LEVELS = ["Succès total", "Impact léger", "Impact moyen", "Impact```` grave", "Annulé"]
PRIORITY = {"Annyaml typeulé":="issue 4, "Impact grave-tree"
data:
": 3,- tag "Impact: ' moyen":dadamar 2ou/, "SimulationVImpact léger": 2#1'
1, "Succ  title: 'ès totalVér": ification et sécurisation0}
DEFAULT_CATEGORY du code principal de = "Non catég Simulation.V2.pyorisé"

scenario'
  repository:_name = 'dad st.textamarou_input("Nom du/Simulation scénario (V2'
 existant ou nouveau number:)")

# ================= 1========
# Fonction utilitaires
 
# =========================
def state: ensure_uid_column(df: pd 'open.DataFrame) -> pd.DataFrame:
'
     """ url:S'ass 'httpsure que le Data://githubFrame contient.com/dadamar une colonneou/ 'uidSimulationV2/issues/1' unique'
```` pour identifier les lignes."""
    if "uid" not in df.columns:
        df = df.copy()
        df["uid"] = [str(uuid.uuid4()) for _ in range(len
