import uuid
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.express as px

st.set_page_config(page_title="Dashboard Simulation Générique", layout="wide")
st.title("Tableau de bord multi-scénarios générique")

# =========================
# Paramètres généraux
# =========================
N = st.number_input(
    "Nombre de simulations (N)",
    min_value=1000,
    max_value=1_000_000,
    value=100_000,
    step=1000,
    help="Attention : valeurs très élevées peuvent consommer beaucoup de mémoire/CPU."
)
if N > 500_000:
    st.warning("Valeur de N > 500k — la simulation peut être lente.")

# =========================
# Définition dynamique des niveaux d'impact
# =========================
st.sidebar.header("Configuration des niveaux d'impact")
impact_levels_input = st.sidebar.text_area(
    "Définir les niveaux d'impact, du moins grave au plus grave, séparés par des virgules",
    value="Succès total,Impact léger,Impact moyen,Impact grave,Annulé"
)
IMPACT_LEVELS = [x.strip() for x in impact_levels_input.split(",") if x.strip()]
PRIORITY = {lvl: i for i, lvl in enumerate(IMPACT_LEVELS)}

# =========================
# Gestion des scénarios
# =========================
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

scenario_name = st.text_input("Nom du scénario (existant ou nouveau)")

def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    else:
        df = df.copy()
        missing = df["id"].isna() | (df["id"] == "")
        df.loc[missing, "id"] = [str(uuid.uuid4()) for _ in range(missing.sum())]
    return df

# =========================
# Interface des événements
# =========================
if scenario_name:
    if scenario_name not in st.session_state.scenarios:
        st.session_state.scenarios[scenario_name] = pd.DataFrame(
            columns=["id", "name", "prob", "impact", "category"]
        )

    st.subheader(f"Événements du scénario '{scenario_name}'")
    df = ensure_id_column(pd.DataFrame(st.session_state.scenarios[scenario_name]))

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_default_column(editable=True, resizable=True)
    gb.configure_column("id", hide=True)
    gb.configure_column("prob", type=["numericColumn", "numberColumnFilter"], precision=4)
    gb.configure_column("impact", cellEditor="agSelectCellEditor", cellEditorParams={"values": IMPACT_LEVELS})
    grid_options = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True
    )

    updated_df = pd.DataFrame(grid_response["data"])
    updated_df = ensure_id_column(updated_df)
    cols = ["id", "name", "prob", "impact", "category"]
    for c in cols:
        if c not in updated_df.columns:
            updated_df[c] = "" if c != "prob" else 0.0
    updated_df = updated_df[cols]
    st.session_state.scenarios[scenario_name] = updated_df

    if st.button("Ajouter un événement"):
        new_row = pd.DataFrame([{
            "id": str(uuid.uuid4()),
            "name": "",
            "prob": 0.0,
            "impact": IMPACT_LEVELS[1] if len(IMPACT_LEVELS) > 1 else IMPACT_LEVELS[0],
            "category": "Non catégorisé"
        }])
        st.session_state.scenarios[scenario_name] = pd.concat(
            [st.session_state.scenarios[scenario_name], new_row], ignore_index=True
        )
        st.experimental_rerun()

    selected = grid_response.get("selected_rows", [])
    if selected:
        if st.button("Supprimer l'événement(s) sélectionné(s)"):
            ids_to_remove = [r.get("id") for r in selected if r.get("id") is not None]
            if ids_to_remove:
                st.session_state.scenarios[scenario_name] = st.session_state.scenarios[scenario_name][
                    ~st.session_state.scenarios[scenario_name]["id"].isin(ids_to_remove)
                ].reset_index(drop=True)
                st.experimental_rerun()

# =========================
# Simulation multi-scénarios
# =========================
st.header("Simulation multi-scénarios")
selected_scenarios = st.multiselect("Sélectionner les scénarios à simuler", list(st.session_state.scenarios.keys()))

# Récupérer toutes les catégories disponibles
all_categories = set()
for df in st.session_state.scenarios.values():
    if isinstance(df, pd.DataFrame) and "category" in df.columns:
        all_categories.update(df["category"].dropna().astype(str).unique())
all_categories = sorted(list(all_categories))
selected_categories = st.multiselect("Filtrer par catégorie", options=["Toutes"] + all_categories, default=["Toutes"])

if st.button("Lancer la simulation"):
    if not selected_scenarios:
        st.warning("Sélectionnez au moins un scénario avant de lancer la simulation.")
        st.stop()

    rng = np.random.default_rng()
    results = []
    validation_issues = []

    try:
        for scenario in selected_scenarios:
            df = pd.DataFrame(st.session_state.scenarios.get(scenario, pd.DataFrame()))
            if df.empty:
                st.info(f"Scénario '{scenario}' vide — aucune simulation.")
                continue
            df = ensure_id_column(df)

            # Validation des probabilités
            probs = []
            for val in df["prob"].fillna(0):
                try:
                    p = float(val)
                    p = max(0.0, min(1.0, p))
                    probs.append(p)
                except:
                    probs.append(0.0)
            df["prob"] = probs

            # Validation impacts
            df["impact"] = df["impact"].apply(lambda x: x if x in IMPACT_LEVELS else IMPACT_LEVELS[1] if len(IMPACT_LEVELS) > 1 else IMPACT_LEVELS[0])

            # Catégories
            df["category"] = df["category"].fillna("Non catégorisé").astype(str)

            # Filtrage par catégorie
            if selected_categories and "Toutes" not in selected_categories:
                df = df[df["category"].isin(selected_categories)]
                if df.empty:
                    st.info(f"Aucun événement du scénario '{scenario}' ne correspond au filtre.")
                    results.append({"scenario": scenario, "summary": pd.DataFrame(), "category_summary": pd.DataFrame()})
                    continue

            # Simulation vectorisée
            outcomes = np.full(N, IMPACT_LEVELS[0], dtype=object)
            outcome_categories = np.full(N, IMPACT_LEVELS[0], dtype=object)

            for _, ev in df.iterrows():
                p = float(ev["prob"])
                if p <= 0:
                    continue
                mask = rng.random(N) < p
                current_priority = np.vectorize(PRIORITY.get)(outcomes)
                new_priority = PRIORITY.get(ev["impact"], 1)
                mask_update = mask & (new_priority > current_priority)
                outcomes[mask_update] = ev["impact"]
                outcome_categories[mask_update] = ev["category"]

            # Résumés
            summary = pd.Series(outcomes).value_counts().rename_axis("issue").reset_index(name="count")
            summary["probability"] = (summary["count"] / N * 100).round(2)
            summary["scenario"] = scenario

            df_cat = pd.DataFrame({"category": outcome_categories, "impact": outcomes})
            cat_summary = df_cat.groupby("category").size().reset_index(name="count")
            cat_summary["probability"] = (cat_summary["count"] / N * 100).round(2)
            cat_summary["scenario"] = scenario

            results.append({"scenario": scenario, "summary": summary, "category_summary": cat_summary})

        # Affichage
        st.subheader("Comparaison des issues")
        all_summary = pd.concat([r["summary"] for r in results if not r["summary"].empty], ignore_index=True) if any([not r["summary"].empty for r in results]) else pd.DataFrame()
        if not all_summary.empty:
            fig_issue = px.bar(all_summary, x="issue", y="probability", color="scenario", barmode="group",
                               text=all_summary["probability"].astype(str) + "%")
            fig_issue.update_layout(yaxis_title="Probabilité (%)")
            st.plotly_chart(fig_issue)

        st.subheader("Comparaison par catégorie")
        all_cat_summary = pd.concat([r["category_summary"] for r in results if not r["category_summary"].empty], ignore_index=True) if any([not r["category_summary"].empty for r in results]) else pd.DataFrame()
        if not all_cat_summary.empty:
            fig_cat = px.bar(all_cat_summary, x="category", y="probability", color="scenario", barmode="group",
                             text=all_cat_summary["probability"].astype(str) + "%")
            fig_cat.update_layout(yaxis_title="Probabilité (%)")
            st.plotly_chart(fig_cat)

    except Exception as e:
        st.error(f"Erreur lors de la simulation : {e}")
