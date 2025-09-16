import uuid
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.express as px

st.set_page_config(page_title="Dashboard Simulation Ultime", layout="wide")
st.title("Tableau de bord multi-scénarios ultime")

# =========================
# Paramètres généraux
# =========================
# Limite supérieure raisonnable par défaut. On avertit si l'utilisateur choisit une très grande valeur.
N = st.number_input(
    "Nombre de simulations (N)",
    min_value=1000,
    max_value=1_000_000,
    value=100_000,
    step=1000,
    help="Attention : valeurs très élevées peuvent consommer beaucoup de mémoire/CPU."
)
if N > 500_000:
    st.warning("Valeur de N supérieure à 500k — la simulation peut être lente ou consommer beaucoup de mémoire.")

# =========================
# Gestion des scénarios
# =========================
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

scenario_name = st.text_input("Nom du scénario (existant ou nouveau)")

# utilitaires
IMPACT_LEVELS = ["Succès total", "Impact léger", "Impact moyen", "Impact grave", "Annulé"]
PRIORITY = {"Annulé": 4, "Impact grave": 3, "Impact moyen": 2, "Impact léger": 1, "Succès total": 0}

def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """S'assure que le DataFrame a une colonne 'id' unique."""
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    else:
        # Remplir les ids manquants
        df = df.copy()
        missing = df["id"].isna() | (df["id"] == "")
        df.loc[missing, "id"] = [str(uuid.uuid4()) for _ in range(missing.sum())]
    return df

# =========================
# Tableau interactif des événements
# =========================
if scenario_name:
    if scenario_name not in st.session_state.scenarios:
        # initialisation avec colonnes standards + id
        st.session_state.scenarios[scenario_name] = pd.DataFrame(
            columns=["id", "name", "prob", "impact", "category"]
        )

    st.subheader(f"Événements du scénario '{scenario_name}'")
    df = ensure_id_column(pd.DataFrame(st.session_state.scenarios[scenario_name]))

    # Forcer la colonne 'impact' à des valeurs connues lors de la validation,
    # mais on permet l'édition libre dans la grille (corrigé ensuite).
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_default_column(editable=True, resizable=True)
    # masquer l'id dans l'affichage mais le garder pour la gestion
    gb.configure_column("id", header_name="ID interne", hide=True)
    gb.configure_column("prob", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=4)
    grid_options = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True
    )

    # Récupérer les données et s'assurer de la colonne id
    updated_df = pd.DataFrame(grid_response["data"])
    updated_df = ensure_id_column(updated_df)
    # Réordonner colonnes standards si nécessaire
    cols = ["id", "name", "prob", "impact", "category"]
    for c in cols:
        if c not in updated_df.columns:
            updated_df[c] = "" if c != "prob" else 0.0
    updated_df = updated_df[cols]

    st.session_state.scenarios[scenario_name] = updated_df

    # Ajouter ligne vide (avec id unique)
    if st.button("Ajouter un événement"):
        new_row = pd.DataFrame([{
            "id": str(uuid.uuid4()),
            "name": "",
            "prob": 0.0,
            "impact": "Impact léger",
            "category": "Non catégorisé"
        }])
        st.session_state.scenarios[scenario_name] = pd.concat([st.session_state.scenarios[scenario_name], new_row], ignore_index=True)
        st.experimental_rerun()

    # Supprimer ligne(s) sélectionnée(s)
    selected = grid_response.get("selected_rows", [])
    if selected:
        if st.button("Supprimer l'événement(s) sélectionné(s)"):
            # Extraire les ids des lignes sélectionnées (champ 'id')
            ids_to_remove = [r.get("id") for r in selected if r.get("id") is not None]
            if ids_to_remove:
                before_count = len(st.session_state.scenarios[scenario_name])
                st.session_state.scenarios[scenario_name] = st.session_state.scenarios[scenario_name][
                    ~st.session_state.scenarios[scenario_name]["id"].isin(ids_to_remove)
                ].reset_index(drop=True)
                after_count = len(st.session_state.scenarios[scenario_name])
                st.success(f"Supprimé {before_count - after_count} ligne(s).")
                st.experimental_rerun()
            else:
                st.error("Impossible de déterminer les lignes sélectionnées (champ 'id' manquant).")

# =========================
# Simulation multi-scénarios
# =========================
st.header("Simulation multi-scénarios")
selected_scenarios = st.multiselect("Sélectionner les scénarios à simuler", list(st.session_state.scenarios.keys()))

# Construire la liste de catégories existantes en évitant les dataframes vides / colonnes manquantes
all_categories = set()
for df in st.session_state.scenarios.values():
    if isinstance(df, pd.DataFrame) and "category" in df.columns:
        cats = df["category"].dropna().astype(str).unique().tolist()
        all_categories.update(cats)
all_categories = sorted(list(all_categories))
selected_categories = st.multiselect("Filtrer par catégorie", options=["Toutes"] + all_categories, default=["Toutes"])

if st.button("Lancer la simulation"):
    if not selected_scenarios:
        st.warning("Veuillez sélectionner au moins un scénario avant de lancer la simulation.")
        st.stop()

    rng = np.random.default_rng()
    results = []
    validation_issues = []

    try:
        for scenario in selected_scenarios:
            df = pd.DataFrame(st.session_state.scenarios.get(scenario, pd.DataFrame()))
            if df.empty:
                st.info(f"Le scénario '{scenario}' est vide — aucune simulation effectuée pour ce scénario.")
                continue

            df = ensure_id_column(df)

            # Validation / nettoyage des données du scénario avant simulation
            # 1) prob -> float entre 0 et 1
            probs = []
            prob_issues = []
            for i, val in enumerate(df["prob"].fillna(0)):
                try:
                    p = float(val)
                    if p < 0 or p > 1:
                        prob_issues.append((i, val))
                        p = max(0.0, min(1.0, p))
                    probs.append(p)
                except Exception:
                    prob_issues.append((i, val))
                    probs.append(0.0)
            if prob_issues:
                validation_issues.append(f"Scénario '{scenario}': {len(prob_issues)} prob non-numériques ou hors [0,1] (corrigées).")

            df["prob"] = probs

            # 2) impact -> s'assurer que la valeur appartient à IMPACT_LEVELS
            impacts = []
            impact_issues = []
            for val in df["impact"].fillna("Impact léger"):
                if val not in IMPACT_LEVELS:
                    impact_issues.append(val)
                    impacts.append("Impact léger")
                else:
                    impacts.append(val)
            if impact_issues:
                validation_issues.append(f"Scénario '{scenario}': {len(impact_issues)} impact(s) non-standards convertis en 'Impact léger'.")

            df["impact"] = impacts

            # 3) catégorie : remplir les vides
            if "category" not in df.columns:
                df["category"] = "Non catégorisé"
            else:
                df["category"] = df["category"].fillna("Non catégorisé").astype(str)

            # Appliquer le filtre de catégories demandé par l'utilisateur
            if selected_categories and "Toutes" not in selected_categories:
                df = df[df["category"].isin(selected_categories)]
                if df.empty:
                    st.info(f"Aucun événement du scénario '{scenario}' ne correspond au filtre de catégorie sélectionné.")
                    results.append({"scenario": scenario, "summary": pd.DataFrame(), "category_summary": pd.DataFrame()})
                    continue

            # Simulation vectorisée
            outcomes = np.full(N, "Succès total", dtype=object)
            outcome_categories = np.full(N, "Succès total", dtype=object)

            # convert impacts to priority numbers once
            for _, ev in df.iterrows():
                p = float(ev["prob"])
                if p <= 0:
                    continue
                mask = rng.random(N) < p
                current_priority = np.vectorize(PRIORITY.get)(outcomes)
                new_priority = PRIORITY.get(ev["impact"], PRIORITY["Impact léger"])
                mask_update = mask & (new_priority > current_priority)
                if mask_update.any():
                    outcomes[mask_update] = ev["impact"]
                    outcome_categories[mask_update] = ev["category"]

            # Résumés
            if len(outcomes) == 0:
                summary = pd.DataFrame(columns=["issue", "count", "probability", "scenario"])
            else:
                summary = pd.Series(outcomes).value_counts().rename_axis("issue").reset_index(name="count")
                summary["probability"] = (summary["count"] / N * 100).round(2)
                summary["scenario"] = scenario

            df_cat = pd.DataFrame({"category": outcome_categories, "impact": outcomes})
            cat_summary = df_cat.groupby("category").size().reset_index(name="count")
            cat_summary["probability"] = (cat_summary["count"] / N * 100).round(2)
            cat_summary["scenario"] = scenario

            results.append({"scenario": scenario, "summary": summary, "category_summary": cat_summary})

        # Afficher les éventuelles issues de validation
        if validation_issues:
            for msg in validation_issues:
                st.warning(msg)

        # =========================
        # Visualisation
        # =========================
        st.subheader("Comparaison des issues")
        all_summary = pd.concat([r["summary"] for r in results if not r["summary"].empty], ignore_index=True) if any([not r["summary"].empty for r in results]) else pd.DataFrame()
        if not all_summary.empty:
            fig_issue = px.bar(all_summary, x="issue", y="probability", color="scenario", barmode="group",
                               text=all_summary["probability"].astype(str) + "%")
            fig_issue.update_layout(yaxis_title="Probabilité (%)")
            st.plotly_chart(fig_issue)
        else:
            st.info("Aucun résultat d'issue à afficher.")

        st.subheader("Comparaison par catégorie")
        all_cat_summary = pd.concat([r["category_summary"] for r in results if not r["category_summary"].empty], ignore_index=True) if any([not r["category_summary"].empty for r in results]) else pd.DataFrame()
        if not all_cat_summary.empty:
            fig_cat = px.bar(all_cat_summary, x="category", y="probability", color="scenario", barmode="group",
                             text=all_cat_summary["probability"].astype(str) + "%")
            fig_cat.update_layout(yaxis_title="Probabilité (%)")
            st.plotly_chart(fig_cat)
        else:
            st.info("Aucun résultat par catégorie à afficher.")

        # =========================
        # Indicateurs clés
        # =========================
        st.subheader("Indicateurs clés par scénario")
        for r in results:
            scenario = r["scenario"]
            summary = r["summary"]
            if summary.empty:
                st.markdown(f"**{scenario}** : aucun résultat.")
                continue
            total_success = float(summary.loc[summary["issue"] == "Succès total", "probability"].values[0]) if "Succès total" in summary["issue"].values else 0.0
            major_block = float(summary.loc[summary["issue"].isin(["Impact grave", "Annulé"]), "probability"].sum())
            st.markdown(f"**{scenario}** : Succès total = {total_success:.2f}%, Impact majeur ou annulation = {major_block:.2f}%")

        # =========================
        # Résumé narratif
        # =========================
        st.subheader("Résumé narratif")
        for r in results:
            scenario = r["scenario"]
            summary = r["summary"]
            cat_summary = r["category_summary"]
            if summary.empty and cat_summary.empty:
                st.markdown(f"**Scénario '{scenario}'** : aucun résultat.")
                continue
            narrative_lines = [f"**Scénario '{scenario}':**"]
            success_rate = float(summary.loc[summary["issue"] == "Succès total", "probability"].values[0]) if ("Succès total" in summary["issue"].values) else 0.0
            narrative_lines.append(f"- {success_rate:.2f}% des simulations sont un succès total.")
            for idx, row in summary.iterrows():
                if row["issue"] != "Succès total":
                    narrative_lines.append(f"- {row['probability']:.2f}% des simulations ont été affectées par '{row['issue']}'.")
            narrative_lines.append("**Impact par catégorie :**")
            for idx, row in cat_summary.iterrows():
                narrative_lines.append(f"- {row['category']}: {row['probability']:.2f}%")
            st.markdown("\n".join(narrative_lines))

        # Export CSV complet (concaténation propre des summaries)
        csv_pieces = []
        for r in results:
            s = r["summary"].copy()
            if not s.empty:
                s["type"] = "issue"
                csv_pieces.append(s)
            c = r["category_summary"].copy()
            if not c.empty:
                c = c.rename(columns={"category": "issue"})  # pour uniformiser les colonnes lors de la concat
                c["type"] = "category"
                csv_pieces.append(c)
        if csv_pieces:
            csv_all = pd.concat(csv_pieces, ignore_index=True).fillna("")
            st.download_button(
                "Télécharger CSV complet",
                data=csv_all.to_csv(index=False),
                file_name="simulation_dashboard_ultime.csv",
                mime="text/csv"
            )
        else:
            st.info("Aucun CSV à exporter.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la simulation : {e}")
        # En debug local, il peut être utile d'afficher la stack trace :
        # import traceback; st.text(traceback.format_exc())
