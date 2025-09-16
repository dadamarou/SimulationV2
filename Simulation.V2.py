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
N = st.number_input("Nombre de simulations (N)", min_value=1000, max_value=1_000_000, value=100_000, step=1000)

# =========================
# Gestion des scénarios
# =========================
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

scenario_name = st.text_input("Nom du scénario (existant ou nouveau)")

# =========================
# Tableau interactif des événements
# =========================
if scenario_name:
    if scenario_name not in st.session_state.scenarios:
        st.session_state.scenarios[scenario_name] = pd.DataFrame(columns=["name","prob","impact","category"])
    
    st.subheader(f"Événements du scénario '{scenario_name}'")
    df = st.session_state.scenarios[scenario_name]
    
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_default_column(editable=True, resizable=True)
    gb.configure_column("prob", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
    grid_options = gb.build()
    
    grid_response = AgGrid(df, gridOptions=grid_options, update_mode=GridUpdateMode.VALUE_CHANGED, allow_unsafe_jscode=True)
    st.session_state.scenarios[scenario_name] = grid_response["data"]
    
    # Ajouter ligne vide
    if st.button("Ajouter un événement"):
        new_row = pd.DataFrame([{"name":"","prob":0.0,"impact":"Impact léger","category":"Non catégorisé"}])
        st.session_state.scenarios[scenario_name] = pd.concat([st.session_state.scenarios[scenario_name], new_row], ignore_index=True)
    
    # Supprimer ligne sélectionnée
    selected = grid_response["selected_rows"]
    if selected:
        if st.button("Supprimer l'événement sélectionné"):
            indices = [r["_selectedRowNodeInfo"]["nodeId"] for r in selected]
            st.session_state.scenarios[scenario_name] = st.session_state.scenarios[scenario_name].drop(index=indices).reset_index(drop=True)
            st.success("Événement supprimé")

# =========================
# Simulation multi-scénarios
# =========================
st.header("Simulation multi-scénarios")
selected_scenarios = st.multiselect("Sélectionner les scénarios à simuler", list(st.session_state.scenarios.keys()))

all_categories = list(set(cat for df in st.session_state.scenarios.values() for cat in df["category"]))
selected_categories = st.multiselect("Filtrer par catégorie", options=["Toutes"] + all_categories, default=["Toutes"])

impact_levels = ["Succès total", "Impact léger", "Impact moyen", "Impact grave", "Annulé"]
priority = {"Annulé":4, "Impact grave":3, "Impact moyen":2, "Impact léger":1, "Succès total":0}

if st.button("Lancer la simulation"):
    rng = np.random.default_rng()
    results = []
    
    for scenario in selected_scenarios:
        df = st.session_state.scenarios[scenario]
        outcomes = np.full(N, "Succès total", dtype=object)
        outcome_categories = np.full(N, "Succès total", dtype=object)
        
        for _, ev in df.iterrows():
            if selected_categories and "Toutes" not in selected_categories and ev["category"] not in selected_categories:
                continue
            mask = rng.random(N) < float(ev["prob"])
            current_priority = np.vectorize(priority.get)(outcomes)
            new_priority = priority[ev["impact"]]
            mask_update = mask & (new_priority > current_priority)
            outcomes[mask_update] = ev["impact"]
            outcome_categories[mask_update] = ev["category"]
        
        summary = pd.Series(outcomes).value_counts().rename_axis("issue").reset_index(name="count")
        summary["probability"] = (summary["count"]/N*100).round(2)
        summary["scenario"] = scenario
        
        df_cat = pd.DataFrame({"category": outcome_categories, "impact": outcomes})
        cat_summary = df_cat.groupby("category").size().reset_index(name="count")
        cat_summary["probability"] = (cat_summary["count"]/N*100).round(2)
        cat_summary["scenario"] = scenario
        
        results.append({"scenario":scenario, "summary":summary, "category_summary":cat_summary})
    
    # =========================
    # Visualisation
    # =========================
    st.subheader("Comparaison des issues")
    all_summary = pd.concat([r["summary"] for r in results])
    fig_issue = px.bar(all_summary, x="issue", y="probability", color="scenario", barmode="group",
                       text=all_summary["probability"].astype(str)+"%")
    fig_issue.update_layout(yaxis_title="Probabilité (%)")
    st.plotly_chart(fig_issue)
    
    st.subheader("Comparaison par catégorie")
    all_cat_summary = pd.concat([r["category_summary"] for r in results])
    fig_cat = px.bar(all_cat_summary, x="category", y="probability", color="scenario", barmode="group",
                     text=all_cat_summary["probability"].astype(str)+"%")
    fig_cat.update_layout(yaxis_title="Probabilité (%)")
    st.plotly_chart(fig_cat)
    
    # =========================
    # Indicateurs clés
    # =========================
    st.subheader("Indicateurs clés par scénario")
    for r in results:
        scenario = r["scenario"]
        summary = r["summary"]
        total_success = summary.loc[summary["issue"]=="Succès total","probability"].values[0] if "Succès total" in summary["issue"].values else 0
        major_block = summary.loc[summary["issue"].isin(["Impact grave","Annulé"]),"probability"].sum()
        st.markdown(f"**{scenario}** : Succès total = {total_success:.2f}%, Impact majeur ou annulation = {major_block:.2f}%")
    
    # =========================
    # Résumé narratif
    # =========================
    st.subheader("Résumé narratif")
    for r in results:
        scenario = r["scenario"]
        summary = r["summary"]
        cat_summary = r["category_summary"]
        narrative = f"Scénario '{scenario}':\n"
        success_rate = summary.loc[summary["issue"]=="Succès total","probability"].values[0] if "Succès total" in summary["issue"].values else 0
        narrative += f"- {success_rate:.2f}% des simulations sont un succès total.\n"
        for idx, row in summary.iterrows():
            if row["issue"] != "Succès total":
                narrative += f"- {row['probability']:.2f}% des simulations ont été affectées par '{row['issue']}'.\n"
        narrative += "Impact par catégorie :\n"
        for idx, row in cat_summary.iterrows():
            narrative += f"- {row['category']}: {row['probability']:.2f}%\n"
        st.text(narrative)
    
    # Export CSV complet
    csv_all = pd.concat([pd.concat([r["summary"], r["category_summary"]], axis=0) for r in results])
    st.download_button("Télécharger CSV complet", data=csv_all.to_csv(index=False),
                       file_name="simulation_dashboard_ultime.csv", mime="text/csv")
