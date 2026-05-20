"""
app.py — ML Immobilier Dashboard (PRO version)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Immobilier Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.main-title{
    font-size:2rem;
    font-weight:800;
    color:#1a73e8;
}

.card{
    background:#f5f7ff;
    padding:20px;
    border-radius:12px;
    border-left:5px solid #1a73e8;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def get_data(n=1200):

    np.random.seed(42)

    villes = ["Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir"]

    surface = np.random.normal(95, 45, n).clip(20, 500)
    pieces = np.random.randint(1, 8, n)

    prix = (surface * np.random.normal(9500, 2500, n)).clip(80_000, 8_000_000)

    df = pd.DataFrame({
        "ville": np.random.choice(villes, n),
        "prix": prix.astype(int),
        "surface": surface.round(1),
        "chambres": np.random.randint(1, 6, n),
        "salles_bain": np.random.randint(1, 4, n),
        "etage": np.random.randint(0, 12, n),
        "annee": np.random.randint(1970, 2024, n),
        "prix_m2": (prix / surface).round(0),
        "pieces": pieces
    })

    return df


df = get_data()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Exploration", "Prédiction", "Classification"]
)


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
if page == "Dashboard":

    st.markdown('<p class="main-title">Dashboard Immobilier</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total biens", len(df))
    c2.metric("Prix médian", f"{df['prix'].median():,.0f} MAD")
    c3.metric("Surface moyenne", f"{df['surface'].mean():.0f} m²")
    c4.metric("Prix/m²", f"{df['prix_m2'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="prix",
            nbins=40,
            title="Distribution des prix",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city = df.groupby("ville")["prix"].median().reset_index()

        fig = px.bar(
            city,
            x="ville",
            y="prix",
            color="prix",
            title="Prix médian par ville",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            df,
            x="surface",
            y="prix",
            color="ville",
            title="Surface vs Prix",
            opacity=0.6,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        city_m2 = df.groupby("ville")["prix_m2"].mean().sort_values(ascending=False).reset_index()

        fig = px.bar(
            city_m2,
            x="ville",
            y="prix_m2",
            color="prix_m2",
            title="Prix/m² par ville",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# EXPLORATION
# ─────────────────────────────────────────────
elif page == "Exploration":

    st.markdown('<p class="main-title">Exploration des données</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Data", "Stats", "Corrélation"])

    with tab1:
        st.dataframe(df.head(200), use_container_width=True)

    with tab2:
        st.dataframe(df.describe().T, use_container_width=True)

    with tab3:
        corr = df.select_dtypes(include=np.number).corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PRÉDICTION (IMPROVED UI + SAME LOGIC)
# ─────────────────────────────────────────────
elif page == "Prédiction":

    st.markdown('<p class="main-title">Prédiction Prix</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])

    with c1:

        surface_i = st.slider("Surface", 20, 500, 100)
        nb_pieces_i = st.slider("Pièces", 1, 10, 4)
        chambres_i = st.slider("Chambres", 1, 8, 2)
        sdb_i = st.slider("Salles Bain", 1, 5, 1)
        etage_i = st.slider("Étage", 0, 20, 2)
        annee_i = st.slider("Année", 1950, 2024, 2000)

        ville_i = st.selectbox(
            "Ville",
            sorted(df["ville"].unique())
        )

        st.markdown("---")

        btn = st.button(
            "🔮 Estimer le prix",
            use_container_width=True
        )

    with c2:

        st.markdown("### 📊 Résultat")

        if btn:

            prix_m2 = df[df["ville"] == ville_i]["prix_m2"].median()
            pred = prix_m2 * surface_i

            st.markdown(f"""
            <div class="card">
                <h2 style="color:#1a73e8;">{pred:,.0f} MAD</h2>
                <p>Estimation basée sur le marché de <b>{ville_i}</b></p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.info("Clique sur Estimer pour afficher la prédiction")


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────
elif page == "Classification":

    st.markdown('<p class="main-title">Classification des prix</p>', unsafe_allow_html=True)

    bins = [0, 100_000, 250_000, 500_000, 1_000_000, float("inf")]
    labels = ["<100k", "100k-250k", "250k-500k", "500k-1M", ">1M"]

    df2 = df.copy()
    df2["categorie"] = pd.cut(df2["prix"], bins=bins, labels=labels)

    fig = px.histogram(
        df2,
        x="categorie",
        color="ville",
        barmode="group",
        title="Répartition des prix par catégorie",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)