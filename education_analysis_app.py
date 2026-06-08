"""
🇲🇾 Malaysian School Dropout & Completion Rate Analysis
Streamlit App — COS40007 Theme 3: Smart Government
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇲🇾 MY Education Analysis",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#1D9E75", "#534AB7", "#D85A30", "#BA7517", "#185FA5",
           "#993556", "#3B6D11", "#A32D2D"]

STATES_16 = [
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
    "Pahang", "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak",
    "Selangor", "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"
]

T_PRI = "completion_primary"
T_LOW = "completion_secondary_lower"
T_UPP = "completion_secondary_upper"

# ── Session-state initialisation ───────────────────────────────────────────────
for key in ["master", "latest", "cluster_configs", "cv_results", "imp_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "📂 1 · Data Loading",
        "🧹 2 · Cleaning & Master",
        "📊 3 · EDA",
        "🔵 4 · K-Means Clustering",
        "🌿 5 · Hierarchical Clustering",
        "⏱ 6 · DTW Time-Series Clustering",
        "🤖 7 · Supervised Models",
        "📋 8 · Summary",
    ],
    key="nav_page",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data directory**")
DATA_DIR = st.sidebar.text_input("Path to raw CSVs", value="./raw_data/")

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🇲🇾 Malaysian School Dropout & Completion Rate Analysis")
    st.markdown("""
> **Research Question:**  
> *Does economic pressure (poverty, household income, family size) drive school dropout /  
> lower completion rates across Malaysian states, and are there distinct state groupings  
> with different vulnerability profiles?*

**Pipeline (Approach E — Unsupervised → Supervised)**

| Step | Module | Description |
|------|--------|-------------|
| 1 | Data Loading | Load 9 raw DOSM CSV files |
| 2 | Cleaning & Master | Preprocess, feature-engineer, merge into master panel |
| 3 | EDA | Distributions, trends, correlations |
| 4 | K-Means | Configurable state clustering by socio-economic profile |
| 5 | Hierarchical | Ward-linkage dendrogram clustering |
| 6 | DTW | Time-series shape clustering |
| 7 | Supervised | Ridge / Random Forest / Gradient Boosting with LOO-CV |
| 8 | Summary | Evaluation metrics & policy interpretation |

**Use the sidebar to navigate and run each module in order.**  
Upload your data directory path in the sidebar before starting.
""")
    st.info("👈 Set your **Data directory** in the sidebar, then go to **1 · Data Loading**.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 1 · Data Loading":
    st.title("📂 Data Loading")
    st.markdown("Load the nine raw DOSM CSV files from your data directory.")

    FILE_MAP = {
        "completion":  "completion_school_state.csv",
        "poverty":     "hh_poverty_state.csv",
        "income":      "hh_income_state.csv",
        "households":  "hh_lq_state.csv",
        "population":  "population_state.csv",
        "fertility":   "fertility_state.csv",
        "teachers":    "teachers_district.csv",
        "schools":     "schools_district.csv",
        "enrolment":   "enrolment_school_district.csv",
    }

    st.subheader("File mapping")
    cols = st.columns(2)
    for i, (k, v) in enumerate(FILE_MAP.items()):
        cols[i % 2].text(f"  {k:15s} ← {v}")

    if st.button("▶ Load all files", type="primary"):
        import os
        loaded = {}
        errors = []
        prog = st.progress(0)
        status = st.empty()

        for idx, (name, fname) in enumerate(FILE_MAP.items()):
            path = os.path.join(DATA_DIR, fname)
            try:
                df = pd.read_csv(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df["year"] = df["date"].dt.year
                loaded[name] = df
                status.success(f"✅ {name}: {df.shape[0]:,} rows")
            except Exception as e:
                errors.append(f"❌ {name}: {e}")
                status.error(f"❌ {name}: {e}")
            prog.progress((idx + 1) / len(FILE_MAP))

        st.session_state["raw"] = loaded

        if errors:
            st.warning(f"{len(errors)} file(s) failed to load.")
        else:
            st.success("All 9 files loaded successfully!")

        st.subheader("Dataset overview")
        rows = []
        for name, df in loaded.items():
            rows.append({
                "Dataset": name,
                "Rows": df.shape[0],
                "Cols": df.shape[1],
                "Columns": ", ".join(df.columns[:6].tolist()) + ("…" if df.shape[1] > 6 else ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif "raw" in st.session_state and st.session_state["raw"]:
        st.info("Data already loaded. Proceed to **2 · Cleaning & Master**.")
        for name, df in st.session_state["raw"].items():
            st.write(f"**{name}**: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLEANING & MASTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧹 2 · Cleaning & Master":
    st.title("🧹 Cleaning & Master Panel")

    if "raw" not in st.session_state or not st.session_state["raw"]:
        st.warning("⚠️ Load data first (Module 1).")
        st.stop()

    raw = st.session_state["raw"]

    with st.expander("⚙️ Preprocessing options", expanded=True):
        min_year = st.number_input("Min year", value=2016, step=1)
        max_year = st.number_input("Max year", value=2022, step=1)
        drop_missing = st.checkbox("Drop rows with any missing value (after interpolation)", value=True)
        save_path = st.text_input("Save master CSV (optional)", value="./master.csv")

    if st.button("▶ Build master panel", type="primary"):
        with st.spinner("Processing…"):
            try:
                # ── Completion ─────────────────────────────────────────────
                df_completion = raw["completion"]
                comp = df_completion[
                    (df_completion["state"].isin(STATES_16)) &
                    (df_completion["sex"] == "both")
                ][["state", "year", "stage", "completion"]].copy()
                comp = comp[(comp["year"] >= min_year) & (comp["year"] <= max_year)]
                comp_wide = comp.pivot_table(
                    index=["state", "year"], columns="stage", values="completion"
                ).reset_index()
                comp_wide = comp_wide.rename(columns={
                    "primary": T_PRI,
                    "secondary_lower": T_LOW,
                    "secondary_upper": T_UPP,
                })

                # ── Enrolment ──────────────────────────────────────────────
                df_enrolment = raw["enrolment"]
                ENROL_COL = next(
                    (c for c in df_enrolment.columns
                     if c not in ["state","district","stage","sex","date","year","type"]
                     and df_enrolment[c].dtype in ["int64","float64"]), None)
                enrol_st = df_enrolment[
                    (df_enrolment.get("district", pd.Series(["All Districts"]*len(df_enrolment))) == "All Districts") &
                    (df_enrolment["state"].isin(STATES_16)) &
                    (df_enrolment["stage"].isin(["primary","secondary"]))
                ].copy()
                if "sex" in enrol_st.columns:
                    enrol_st = enrol_st[enrol_st["sex"].isin(["both","Both","overall","Overall"])]
                if ENROL_COL and "stage" in enrol_st.columns:
                    enrol_st = enrol_st.pivot_table(
                        index=["state","year"], columns="stage", values=ENROL_COL
                    ).reset_index()
                    enrol_st.columns = ["state","year","enrolment_primary","enrolment_secondary"]

                # ── Schools ────────────────────────────────────────────────
                df_schools = raw["schools"]
                SCHOOL_COL = next(
                    (c for c in df_schools.columns
                     if c not in ["state","district","stage","sex","date","year","type"]
                     and df_schools[c].dtype in ["int64","float64"]), None)
                schools_st = df_schools[
                    (df_schools.get("district", pd.Series(["All Districts"]*len(df_schools))) == "All Districts") &
                    (df_schools["state"].isin(STATES_16)) &
                    (df_schools["stage"].isin(["primary","secondary"]))
                ].copy()
                if SCHOOL_COL and "stage" in schools_st.columns:
                    schools_st = schools_st.pivot_table(
                        index=["state","year"], columns="stage", values=SCHOOL_COL
                    ).reset_index()
                    schools_st.columns = ["state","year","schools_primary","schools_secondary"]

                # ── Teachers ───────────────────────────────────────────────
                df_teachers = raw["teachers"]
                TEACH_COL = next(
                    (c for c in df_teachers.columns
                     if c not in ["state","district","stage","sex","date","year","type"]
                     and df_teachers[c].dtype in ["int64","float64"]), None)
                teachers_st = df_teachers[
                    (df_teachers.get("district", pd.Series(["All Districts"]*len(df_teachers))) == "All Districts") &
                    (df_teachers["state"].isin(STATES_16)) &
                    (df_teachers["stage"].isin(["primary","secondary"]))
                ].copy()
                if "sex" in teachers_st.columns:
                    teachers_st = teachers_st[teachers_st["sex"].isin(["both","Both","overall","Overall"])]
                if TEACH_COL and "stage" in teachers_st.columns:
                    teachers_st = teachers_st.pivot_table(
                        index=["state","year"], columns="stage", values=TEACH_COL
                    ).reset_index()
                    teachers_st.columns = ["state","year","teachers_primary","teachers_secondary"]

                # ── Poverty ────────────────────────────────────────────────
                poverty = raw["poverty"][raw["poverty"]["state"].isin(STATES_16)].copy()
                pov_cols = ["state","year"] + [c for c in ["poverty_absolute","poverty_hardcore","poverty_relative"] if c in poverty.columns]
                poverty = poverty[pov_cols]

                # ── Income ─────────────────────────────────────────────────
                income = raw["income"][raw["income"]["state"].isin(STATES_16)].copy()
                median_col = next((c for c in income.columns if "median" in c.lower()), None)
                mean_col   = next((c for c in income.columns if "mean"   in c.lower()), None)
                keep = ["state","year"] + [c for c in [median_col, mean_col] if c]
                income = income[keep].copy()
                if median_col: income.rename(columns={median_col: "income_median"}, inplace=True)
                if mean_col:   income.rename(columns={mean_col: "income_mean"}, inplace=True)

                # ── Household size ─────────────────────────────────────────
                df_pop = raw["population"]
                pop_total = df_pop[
                    (df_pop["state"].isin(STATES_16)) &
                    (df_pop["sex"] == "both") &
                    (df_pop["age"] == "overall") &
                    (df_pop.get("ethnicity", pd.Series(["overall"]*len(df_pop))) == "overall")
                ][["state","year","population"]].copy()
                hh = raw["households"][raw["households"]["state"].isin(STATES_16)][["state","year","households"]].copy()
                hh_pop = pd.merge(hh, pop_total, on=["state","year"], how="inner")
                hh_pop["avg_hh_size"] = (hh_pop["population"] * 1000) / hh_pop["households"]

                # ── Fertility ──────────────────────────────────────────────
                df_fert = raw["fertility"][raw["fertility"]["state"].isin(STATES_16)].copy()
                if "age_group" in df_fert.columns and "fertility_rate" in df_fert.columns:
                    fertility_pivot = df_fert.pivot_table(
                        index=["state","year"], columns="age_group", values="fertility_rate"
                    ).reset_index()
                    fertility_pivot.columns = ["state","year"] + [
                        f"tfr_{col}" for col in fertility_pivot.columns if col not in ["state","year"]
                    ]
                else:
                    fertility_pivot = df_fert[["state","year"]].drop_duplicates()

                # ── Merge ──────────────────────────────────────────────────
                master = comp_wide.copy()
                for df_m in [teachers_st, schools_st, enrol_st, poverty, income,
                              hh_pop[["state","year","avg_hh_size"]], fertility_pivot]:
                    if df_m is not None and len(df_m) > 0:
                        master = master.merge(df_m, on=["state","year"], how="left")

                master = master.sort_values(["state","year"]).reset_index(drop=True)

                # ── Interpolate sparse columns ─────────────────────────────
                SPARSE = [c for c in master.columns
                          if c not in ["state","year"] and master[c].isna().sum() > 0]
                master[SPARSE] = master.groupby("state")[SPARSE].transform(
                    lambda s: s.interpolate(method="linear", limit_direction="both")
                )

                # ── Derived features ───────────────────────────────────────
                if "enrolment_primary" in master and "teachers_primary" in master:
                    master["str_primary"] = master["enrolment_primary"] / master["teachers_primary"]
                if "enrolment_secondary" in master and "teachers_secondary" in master:
                    master["str_secondary"] = master["enrolment_secondary"] / master["teachers_secondary"]
                if "schools_primary" in master and "enrolment_primary" in master:
                    master["schools_per_1k_primary"] = master["schools_primary"] / (master["enrolment_primary"].replace(0, np.nan) / 1000)
                if "schools_secondary" in master and "enrolment_secondary" in master:
                    master["schools_per_1k_secondary"] = master["schools_secondary"] / (master["enrolment_secondary"].replace(0, np.nan) / 1000)
                if "str_primary" in master and "schools_per_1k_primary" in master:
                    master["education_pressure_primary"] = master["str_primary"] / (master["schools_per_1k_primary"] + 1e-6)
                if "str_secondary" in master and "schools_per_1k_secondary" in master:
                    master["education_pressure_secondary"] = master["str_secondary"] / (master["schools_per_1k_secondary"] + 1e-6)
                if "income_median" in master and "avg_hh_size" in master:
                    master["urban_pressure_proxy"] = master["income_median"] / (master["avg_hh_size"] + 1e-6)

                # ── Drop missing ───────────────────────────────────────────
                if drop_missing:
                    before = len(master)
                    missing_rows = master.loc[master.isnull().any(axis=1), ["state","year"]].drop_duplicates()
                    master = master.merge(missing_rows, on=["state","year"], how="left", indicator=True)
                    master = master[master["_merge"] == "left_only"].drop(columns=["_merge"])
                    master = master.sort_values(["state","year"]).reset_index(drop=True)
                    st.info(f"Dropped {before - len(master)} rows with missing values.")

                st.session_state["master"] = master

                # ── Save ───────────────────────────────────────────────────
                if save_path:
                    try:
                        master.to_csv(save_path, index=False)
                        st.success(f"Saved to {save_path}")
                    except Exception as e:
                        st.warning(f"Could not save: {e}")

            except Exception as e:
                st.error(f"Error building master: {e}")
                st.exception(e)
                st.stop()

        st.success(f"✅ Master panel ready: {master.shape[0]} rows × {master.shape[1]} cols")
        st.write("**Years:**", sorted(master["year"].unique()))
        st.write("**States:**", sorted(master["state"].unique()))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample rows")
            st.dataframe(master.head(10), use_container_width=True)
        with col2:
            st.subheader("Missing values")
            miss = master.isnull().sum().sort_values(ascending=False)
            miss = miss[miss > 0]
            if len(miss):
                st.dataframe(miss.rename("Missing").reset_index(), use_container_width=True)
            else:
                st.success("No missing values!")

    elif st.session_state["master"] is not None:
        master = st.session_state["master"]
        st.info(f"Master already built: {master.shape[0]} rows × {master.shape[1]} cols")
        st.dataframe(master.head(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 3 · EDA":
    st.title("📊 Exploratory Data Analysis")

    master = st.session_state.get("master")
    if master is None:
        st.warning("⚠️ Build master panel first (Module 2).")
        st.stop()

    tabs = st.tabs(["Completion by State", "Trends Over Time", "Correlation Heatmap", "Scatter Plots"])

    # ── Tab 1: Bar chart ──────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Mean Completion Rate by State")
        state_avg = master.groupby("state").agg(
            pri=(T_PRI, "mean"), low=(T_LOW, "mean"), upp=(T_UPP, "mean"),
            poverty=("poverty_absolute", "mean") if "poverty_absolute" in master.columns else ("year","count"),
        ).reset_index()

        stage_sel = st.selectbox("Stage", ["Primary", "Lower Secondary", "Upper Secondary"])
        col_map = {"Primary": "pri", "Lower Secondary": "low", "Upper Secondary": "upp"}
        col = col_map[stage_sel]

        if col in state_avg.columns:
            sdf = state_avg.sort_values(col)
            median_val = sdf[col].median()
            sdf["color"] = sdf[col].apply(lambda v: "Below median" if v < median_val else "Above median")
            fig = px.bar(
                sdf, x=col, y="state", orientation="h",
                color="color",
                color_discrete_map={"Below median": "#D85A30", "Above median": "#1D9E75"},
                title=f"{stage_sel} Completion Rate by State",
                labels={col: "Mean Completion Rate (%)", "state": "State"},
                template="plotly_white", height=550,
            )
            fig.add_vline(x=median_val, line_dash="dash", line_color="gray",
                          annotation_text=f"Median {median_val:.1f}%")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("5 lowest states")
            show_cols = [c for c in ["state", col, "poverty"] if c in state_avg.columns]
            st.dataframe(state_avg.sort_values(col)[show_cols].head(5).round(2), use_container_width=True)
        else:
            st.warning("Completion column not found in master.")

    # ── Tab 2: Trend lines ────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Completion Rate Trends Over Time")
        stage_sel2 = st.selectbox("Stage ", ["Primary", "Lower Secondary", "Upper Secondary"], key="trend_stage")
        target_map = {"Primary": T_PRI, "Lower Secondary": T_LOW, "Upper Secondary": T_UPP}
        t = target_map[stage_sel2]
        states_sel = st.multiselect("States", sorted(master["state"].unique()), default=sorted(master["state"].unique()))
        plot_df = master[master["state"].isin(states_sel)].sort_values("year")
        if t in plot_df.columns:
            fig2 = px.line(plot_df, x="year", y=t, color="state",
                           title=f"{stage_sel2} Completion Rate by State",
                           labels={t: "Completion Rate (%)", "year": "Year"},
                           template="plotly_white", markers=True, height=480)
            fig2.add_hline(y=100, line_dash="dash", line_color="lightgray", annotation_text="100% benchmark")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Column not found.")

    # ── Tab 3: Correlation heatmap ────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Correlation Heatmap")
        available_cols = [c for c in [
            T_PRI, T_LOW, T_UPP,
            "poverty_absolute", "poverty_relative", "income_median", "avg_hh_size",
            "tfr_tfr", "str_primary", "str_secondary",
            "schools_per_1k_primary", "schools_per_1k_secondary",
            "education_pressure_primary", "education_pressure_secondary", "urban_pressure_proxy"
        ] if c in master.columns and master[c].notna().sum() > 20]

        sel_corr = st.multiselect("Features to include", available_cols, default=available_cols[:10])
        if len(sel_corr) >= 2:
            corr = master[sel_corr].corr()
            import plotly.figure_factory as ff
            fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                             zmin=-1, zmax=1, title="Correlation Matrix", height=600)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Select at least 2 features.")

    # ── Tab 4: Scatter (1 variable → 3 stages side by side) ──────────────────────
    with tabs[3]:
        st.subheader("Completion Rate vs Economic Pressure Indicators")
        from scipy import stats as scipy_stats
        import matplotlib.pyplot as plt
        import io

        PRESSURE_VARS_EDA = [c for c in [
            'poverty_absolute', 'poverty_relative', 'avg_hh_size',
            'income_median', 'tfr_tfr',
            'education_pressure_primary', 'education_pressure_secondary',
            'urban_pressure_proxy'
        ] if c in master.columns and master[c].notna().sum() > 15]

        TARGETS_EDA = [
            (T_PRI, 'Primary Completion'),
            (T_LOW, 'Lower Secondary Completion'),
            (T_UPP, 'Upper Secondary Completion'),
        ]

        if not PRESSURE_VARS_EDA:
            st.warning("No pressure variables found in master.")
        else:
            selected_var = st.selectbox(
                "Select predictor variable", 
                PRESSURE_VARS_EDA,
                key="scatter_var_sel"
            )

            col_idx = PRESSURE_VARS_EDA.index(selected_var)
            dot_color = PALETTE[col_idx % len(PALETTE)]

            cols = st.columns(3)

            for col_widget, (target, title) in zip(cols, TARGETS_EDA):
                sub = master[[selected_var, target, 'state']].dropna()

                fig, ax = plt.subplots(figsize=(4.5, 4))

                ax.scatter(sub[selected_var], sub[target],
                        color=dot_color, alpha=0.5, s=30)

                if len(sub) > 2:
                    sl, ic, r, p, _ = scipy_stats.linregress(sub[selected_var], sub[target])
                    xr = np.linspace(sub[selected_var].min(), sub[selected_var].max(), 50)
                    ax.plot(xr, sl * xr + ic, 'r--', lw=1.3, alpha=0.7)
                    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                    r_label = f'r={r:.2f} {sig}'
                else:
                    r_label = 'n/a'

                ax.set_title(r_label, fontsize=9)
                ax.set_xlabel(selected_var, fontsize=8)
                ax.set_ylabel('Completion (%)', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)
                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)

                col_widget.subheader(title)
                col_widget.image(buf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 4 · K-Means Clustering":
    st.title("🔵 K-Means State Clustering")

    master = st.session_state.get("master")
    if master is None:
        st.warning("⚠️ Build master panel first (Module 2).")
        st.stop()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from kneed import KneeLocator

    latest = master[master["state"].isin(STATES_16)].sort_values("year").groupby("state").last().reset_index()

    st.subheader("⚙️ Clustering Configuration")

    # Feature selector
    all_feat_cols = [c for c in latest.columns
                     if c not in ["state","year","date"] and latest[c].dtype in ["float64","int64"]
                     and latest[c].notna().sum() >= len(latest) * 0.6]

    st.markdown("**Select features for each clustering stage:**")
    col1, col2, col3 = st.columns(3)

    default_pri = [c for c in ["poverty_absolute","income_median","str_primary","tfr_tfr","schools_per_1k_primary", T_PRI] if c in all_feat_cols]
    default_low = [c for c in ["poverty_absolute","income_median","str_primary","tfr_tfr", T_LOW] if c in all_feat_cols]
    default_upp = [c for c in ["poverty_absolute","income_median","tfr_tfr", T_UPP] if c in all_feat_cols]

    feats_pri = col1.multiselect("Primary stage", all_feat_cols, default=default_pri, key="km_feats_pri")
    feats_low = col2.multiselect("Lower Secondary", all_feat_cols, default=default_low, key="km_feats_low")
    feats_upp = col3.multiselect("Upper Secondary", all_feat_cols, default=default_upp, key="km_feats_upp")

    st.markdown("**K range & options:**")
    c1, c2, c3 = st.columns(3)
    k_min = c1.number_input("Min k", value=2, min_value=2, step=1)
    k_max = c2.number_input("Max k", value=7, min_value=3, step=1)
    n_init = c3.number_input("n_init", value=10, min_value=1, step=5)

    if st.button("▶ Run K-Means for all 3 stages", type="primary"):
        if not feats_pri or not feats_low or not feats_upp:
            st.error("Select at least 1 feature per stage.")
            st.stop()

        CLUSTER_CONFIGS = []
        cluster_configs_store = []

        for label, feats, col_prefix in [
            ("Primary",         feats_pri, "cluster_primary"),
            ("Lower Secondary", feats_low, "cluster_lower"),
            ("Upper Secondary", feats_upp, "cluster_upper"),
        ]:
            st.markdown(f"---\n### {label}")
            X = latest[feats].fillna(latest[feats].median()).values
            sc = StandardScaler()
            X_sc = sc.fit_transform(X)

            K_RANGE = range(int(k_min), min(int(k_max) + 1, len(latest)))
            inertias, silhouettes, db_scores = [], [], []
            for k in K_RANGE:
                km = KMeans(n_clusters=k, random_state=42, n_init=int(n_init))
                lbl = km.fit_predict(X_sc)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X_sc, lbl))
                db_scores.append(davies_bouldin_score(X_sc, lbl))

            optimal_k = list(K_RANGE)[np.argmax(silhouettes)]
            kl = KneeLocator(list(K_RANGE), inertias, curve="convex", direction="decreasing")
            k_elbow = kl.elbow or optimal_k

            # Plot elbow/silhouette/DB
            fig_ev = make_subplots(1, 3, subplot_titles=["Elbow (Inertia)", "Silhouette ↑", "Davies-Bouldin ↓"])
            ks = list(K_RANGE)
            fig_ev.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia", line_color="#185FA5"), 1, 1)
            fig_ev.add_vline(x=k_elbow, line_dash="dash", line_color="red", row=1, col=1)
            fig_ev.add_trace(go.Scatter(x=ks, y=silhouettes, mode="lines+markers", name="Silhouette", line_color="#1D9E75"), 1, 2)
            fig_ev.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=2)
            fig_ev.add_trace(go.Scatter(x=ks, y=db_scores, mode="lines+markers", name="DB", line_color="#D85A30"), 1, 3)
            fig_ev.add_vline(x=optimal_k, line_dash="dash", line_color="blue", row=1, col=3)
            fig_ev.update_layout(height=320, showlegend=False, template="plotly_white",
                                  title=f"{label} — Elbow k={k_elbow} | Best silhouette k={optimal_k} ({max(silhouettes):.3f})")
            st.plotly_chart(fig_ev, use_container_width=True)

            # Allow user override of k
            k_use = st.number_input(f"Use k for {label}", value=optimal_k, min_value=2, max_value=int(k_max), key=f"k_use_{col_prefix}")
            km_final = KMeans(n_clusters=int(k_use), random_state=42, n_init=int(n_init))
            latest[col_prefix] = km_final.fit_predict(X_sc)

            # PCA scatter
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_sc)
            var_ex = pca.explained_variance_ratio_
            pca_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1],
                                   "Cluster": latest[col_prefix].astype(str),
                                   "State": latest["state"]})
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", text="State",
                                 title=f"{label} PCA — {var_ex[0]*100:.1f}% + {var_ex[1]*100:.1f}% variance",
                                 template="plotly_white", height=400)
            fig_pca.update_traces(textposition="top center", marker_size=10)
            st.plotly_chart(fig_pca, use_container_width=True)

            # Cluster profiles
            st.write("**Cluster profiles (mean):**")
            profile = latest.groupby(col_prefix)[feats].mean().round(2)
            st.dataframe(profile, use_container_width=True)

            st.write("**States per cluster:**")
            for c_id in sorted(latest[col_prefix].unique()):
                states = latest[latest[col_prefix] == c_id]["state"].tolist()
                st.write(f"  Cluster {c_id}: {', '.join(states)}")

            sil = silhouette_score(X_sc, latest[col_prefix])
            db  = davies_bouldin_score(X_sc, latest[col_prefix])
            st.metric(f"Silhouette (k={k_use})", f"{sil:.3f}", delta=None)
            st.metric(f"Davies-Bouldin (k={k_use})", f"{db:.3f}", delta=None)

            CLUSTER_CONFIGS.append((label, feats, col_prefix, int(k_use), X_sc))

        st.session_state["latest"]         = latest
        st.session_state["cluster_configs"] = CLUSTER_CONFIGS
        st.success("✅ K-Means clustering complete. Proceed to Hierarchical or DTW.")

    elif st.session_state.get("latest") is not None:
        st.info("K-Means already run. Results stored in session. Re-run to update.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — HIERARCHICAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 5 · Hierarchical Clustering":
    st.title("🌿 Hierarchical Clustering")

    latest = st.session_state.get("latest")
    cluster_configs = st.session_state.get("cluster_configs")
    if latest is None or cluster_configs is None:
        st.warning("⚠️ Run K-Means first (Module 4).")
        st.stop()

    from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score
    import matplotlib.pyplot as plt
    import io

    linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
    cut_ratio = st.slider("Dendrogram cut height (% of max)", 40, 90, 70)

    if st.button("▶ Run Hierarchical Clustering", type="primary"):
        for label, feats, cluster_col, k, X_sc in cluster_configs:
            st.markdown(f"---\n### {label}")

            Z = linkage(X_sc, method=linkage_method)
            c_stat, _ = cophenet(Z, pdist(X_sc))
            st.metric("Cophenetic Correlation", f"{c_stat:.4f}",
                      help=">0.75 = good fit")

            # Dendrogram using matplotlib → display as image
            fig_d, ax_d = plt.subplots(figsize=(14, 5))
            cut = (cut_ratio / 100) * max(Z[:, 2])
            dendrogram(Z, labels=latest["state"].values, leaf_rotation=40,
                       leaf_font_size=9, color_threshold=cut, ax=ax_d)
            ax_d.set_title(f"Hierarchical Clustering — {label} ({linkage_method} linkage)", fontsize=12)
            ax_d.set_xlabel("State"); ax_d.set_ylabel("Distance")
            ax_d.axhline(y=cut, color="red", linestyle="--", label=f"{cut_ratio}% cut")
            ax_d.legend(fontsize=8)
            plt.tight_layout()
            buf = io.BytesIO()
            fig_d.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig_d)

            # Assign HC clusters
            hc_labels = fcluster(Z, t=k, criterion="maxclust") - 1
            hc_col = f"hc_{cluster_col}"
            latest[hc_col] = hc_labels

            st.write(f"**HC Clusters (k={k}, {linkage_method}):**")
            for c_id in sorted(latest[hc_col].unique()):
                states = latest[latest[hc_col] == c_id]["state"].tolist()
                st.write(f"  HC Cluster {c_id}: {', '.join(states)}")

            # Comparison with K-Means
            ari = adjusted_rand_score(latest[cluster_col], latest[hc_col])
            col1, col2 = st.columns(2)
            col1.metric("ARI vs K-Means", f"{ari:.4f}", help="1.0 = perfect agreement")
            if ari > 0.7:
                col2.success("Strong agreement — cluster structure is stable.")
            elif ari > 0.4:
                col2.warning("Moderate agreement — some states differ.")
            else:
                col2.error("Low agreement — methods disagree.")

        st.session_state["latest"] = latest
        st.success("✅ Hierarchical clustering complete.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DTW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏱ 6 · DTW Time-Series Clustering":
    st.title("⏱ DTW Time-Series Clustering")

    master = st.session_state.get("master")
    latest = st.session_state.get("latest")
    cluster_configs = st.session_state.get("cluster_configs")

    if master is None:
        st.warning("⚠️ Build master panel first (Module 2).")
        st.stop()

    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.utils import to_time_series_dataset
        from tslearn.metrics import cdist_dtw
        from tslearn.preprocessing import TimeSeriesScalerMinMax
        from sklearn.metrics import silhouette_score
    except ImportError:
        st.error("tslearn not installed. Run: `pip install tslearn`")
        st.stop()

    ts_map = {T_PRI: "Primary", T_LOW: "Lower Secondary", T_UPP: "Upper Secondary"}
    ts_options = [c for c in [T_PRI, T_LOW, T_UPP] if c in master.columns]

    st.subheader("⚙️ Configuration")
    col1, col2, col3 = st.columns(3)
    ts_target = col1.selectbox("Time-series target", ts_options, format_func=lambda x: ts_map.get(x, x))
    k_min_dtw = col2.number_input("Min k", value=2, min_value=2)
    k_max_dtw = col3.number_input("Max k", value=6, min_value=3)

    if st.button("▶ Run DTW Clustering", type="primary"):
        YEARS_TS = sorted(master["year"].unique())
        states_actual = master["state"].unique()

        ts_list, state_order = [], []
        for state in states_actual:
            state_df = (
                master[master["state"] == state]
                .groupby("year")[ts_target].mean()
                .reindex(YEARS_TS)
            )
            series = state_df.values
            if not np.isnan(series).all():
                series = pd.Series(series).interpolate(limit_direction="both").values
                ts_list.append(series)
                state_order.append(state)

        X_ts = to_time_series_dataset(ts_list)
        scaler_ts = TimeSeriesScalerMinMax()
        X_ts_scaled = scaler_ts.fit_transform(X_ts)

        with st.spinner("Computing DTW distance matrix…"):
            dtw_dist_matrix = cdist_dtw(X_ts_scaled)

        dtw_sil_scores = {}
        DTW_K_RANGE = range(int(k_min_dtw), min(int(k_max_dtw) + 1, len(ts_list) - 1))

        prog = st.progress(0)
        for i, k_dtw in enumerate(DTW_K_RANGE):
            km_dtw = TimeSeriesKMeans(n_clusters=k_dtw, metric="dtw",
                                      max_iter=10, random_state=42, n_jobs=-1)
            dtw_lbl = km_dtw.fit_predict(X_ts_scaled)
            sil = silhouette_score(dtw_dist_matrix, dtw_lbl, metric="precomputed")
            dtw_sil_scores[k_dtw] = sil
            prog.progress((i + 1) / len(DTW_K_RANGE))

        # Silhouette vs k
        fig_sil = px.line(x=list(DTW_K_RANGE), y=list(dtw_sil_scores.values()),
                          markers=True, labels={"x": "k", "y": "Silhouette"},
                          title="DTW Silhouette Score vs k", template="plotly_white")
        st.plotly_chart(fig_sil, use_container_width=True)

        OPTIMAL_K_DTW = max(dtw_sil_scores, key=dtw_sil_scores.get)
        st.info(f"Best k = **{OPTIMAL_K_DTW}** (Silhouette = {dtw_sil_scores[OPTIMAL_K_DTW]:.4f})")

        k_dtw_use = st.number_input("Use k", value=OPTIMAL_K_DTW, min_value=2, max_value=int(k_max_dtw))

        km_dtw_final = TimeSeriesKMeans(n_clusters=int(k_dtw_use), metric="dtw",
                                        max_iter=20, random_state=42, n_jobs=-1)
        dtw_labels_final = km_dtw_final.fit_predict(X_ts_scaled)

        # Plot cluster time series
        fig_ts = make_subplots(1, int(k_dtw_use),
                               subplot_titles=[f"DTW Cluster {c}" for c in range(int(k_dtw_use))],
                               shared_yaxes=True)
        for c_id in range(int(k_dtw_use)):
            member_states = [state_order[i] for i, l in enumerate(dtw_labels_final) if l == c_id]
            centroid = km_dtw_final.cluster_centers_[c_id].flatten()
            for state in member_states:
                idx = state_order.index(state)
                fig_ts.add_trace(go.Scatter(
                    x=YEARS_TS, y=X_ts_scaled[idx].flatten(),
                    mode="lines", name=state,
                    line=dict(color=PALETTE[c_id % len(PALETTE)], width=1.2),
                    opacity=0.5, legendgroup=f"c{c_id}", showlegend=(state == member_states[0])
                ), row=1, col=c_id + 1)
            fig_ts.add_trace(go.Scatter(
                x=YEARS_TS, y=centroid, mode="lines",
                line=dict(color="#FFFFFF", width=2.5, dash="dash"),
                name=f"Centroid {c_id}", legendgroup=f"c{c_id}"
            ), row=1, col=c_id + 1)
        fig_ts.update_layout(height=420, template="plotly_white",
                             title=f"DTW Clusters — {ts_map.get(ts_target, ts_target)}")
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Cluster assignments")
        for c_id in range(int(k_dtw_use)):
            states_in = [state_order[i] for i, l in enumerate(dtw_labels_final) if l == c_id]
            st.write(f"**DTW Cluster {c_id}:** {', '.join(states_in)}")

        # Save to latest
        if latest is not None:
            dtw_lookup = {state_order[i]: int(dtw_labels_final[i]) for i in range(len(state_order))}
            dtw_col = f"dtw_{ts_target}"
            latest[dtw_col] = latest["state"].map(dtw_lookup)
            st.session_state["latest"] = latest

        st.success("✅ DTW clustering complete.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SUPERVISED MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 7 · Supervised Models":
    st.title("🤖 Supervised Learning")

    master = st.session_state.get("master")
    if master is None:
        st.warning("⚠️ Build master panel first (Module 2).")
        st.stop()

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.inspection import permutation_importance

    st.subheader("⚙️ Model Configuration")

    target_opts = [c for c in [T_PRI, T_LOW, T_UPP] if c in master.columns]
    target_labels = {T_PRI: "Primary", T_LOW: "Lower Secondary", T_UPP: "Upper Secondary"}
    TARGET_ML = st.selectbox("Target variable", target_opts, index=len(target_opts)-1,
                              format_func=lambda x: target_labels.get(x, x))

    all_feats = [c for c in master.columns
                 if c not in ["state","year","date"] + list(target_opts)
                 and master[c].dtype in ["float64","int64"]
                 and master[c].notna().sum() > master.shape[0] * 0.4]
    default_feats = [c for c in ["poverty_absolute","poverty_relative","avg_hh_size",
                                   "tfr_tfr","income_median","str_primary","str_secondary",
                                   "schools_per_1k_primary"] if c in all_feats]
    PRED_FEATS = st.multiselect("Predictor features", all_feats, default=default_feats)

    st.markdown("**Model hyperparameters:**")
    c1, c2, c3, c4 = st.columns(4)
    ridge_alpha = c1.number_input("Ridge α", value=1.0, min_value=0.01)
    rf_trees    = c2.number_input("RF trees", value=300, min_value=50, step=50)
    rf_depth    = c3.number_input("RF max depth", value=4, min_value=1)
    gb_lr       = c4.number_input("GBM learning rate", value=0.05, min_value=0.01, step=0.01)

    models_to_run = st.multiselect(
        "Models", ["Ridge Regression", "Random Forest", "Gradient Boosting"],
        default=["Ridge Regression", "Random Forest", "Gradient Boosting"]
    )

    if st.button("▶ Run LOO-CV", type="primary"):
        if not PRED_FEATS:
            st.error("Select at least 1 feature.")
            st.stop()

        ml_df = master[PRED_FEATS + [TARGET_ML]].dropna()
        X_ml  = ml_df[PRED_FEATS].values
        y_ml  = ml_df[TARGET_ML].values
        X_std = StandardScaler().fit_transform(X_ml)

        st.info(f"Samples: {len(y_ml)} | Features: {len(PRED_FEATS)} | "
                f"Target range: {y_ml.min():.1f}% – {y_ml.max():.1f}%")

        MODELS = {}
        if "Ridge Regression" in models_to_run:
            MODELS["Ridge Regression"] = (Ridge(alpha=ridge_alpha), X_std)
        if "Random Forest" in models_to_run:
            MODELS["Random Forest"] = (RandomForestRegressor(
                n_estimators=int(rf_trees), max_depth=int(rf_depth), random_state=42), X_ml)
        if "Gradient Boosting" in models_to_run:
            MODELS["Gradient Boosting"] = (GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=gb_lr, random_state=42), X_ml)

        loo = LeaveOneOut()
        cv_res = {}
        prog = st.progress(0)
        total_steps = len(MODELS) * len(y_ml)
        step = 0

        for name, (model, X_in) in MODELS.items():
            preds, truths = [], []
            for tr, te in loo.split(X_in):
                model.fit(X_in[tr], y_ml[tr])
                preds.append(model.predict(X_in[te])[0])
                truths.append(y_ml[te][0])
                step += 1
                prog.progress(step / total_steps)

            p, t = np.array(preds), np.array(truths)
            cv_res[name] = {
                "RMSE": np.sqrt(mean_squared_error(t, p)),
                "MAE":  mean_absolute_error(t, p),
                "R2":   r2_score(t, p),
                "preds": p, "truths": t,
            }

        # Metrics table
        st.subheader("LOO-CV Results")
        BEST = max(cv_res, key=lambda k: cv_res[k]["R2"])
        rows = []
        for n, r in cv_res.items():
            rows.append({"Model": n, "RMSE": round(r["RMSE"],3), "MAE": round(r["MAE"],3), "R²": round(r["R2"],3), "Best": "✅" if n==BEST else ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Chart
        names = list(cv_res.keys())
        fig_m = make_subplots(1, 3, subplot_titles=["RMSE ↓", "MAE ↓", "R² ↑"])
        fig_m.add_trace(go.Bar(x=names, y=[cv_res[n]["RMSE"] for n in names],
                               marker_color=PALETTE[:len(names)]), 1, 1)
        fig_m.add_trace(go.Bar(x=names, y=[cv_res[n]["MAE"] for n in names],
                               marker_color=PALETTE[:len(names)], showlegend=False), 1, 2)
        fig_m.add_trace(go.Bar(x=names, y=[cv_res[n]["R2"] for n in names],
                               marker_color=PALETTE[:len(names)], showlegend=False), 1, 3)
        fig_m.update_layout(height=380, template="plotly_white",
                            showlegend=False, title="Model Comparison (LOO-CV)")
        st.plotly_chart(fig_m, use_container_width=True)

        # Predicted vs Actual
        bp, bt = cv_res[BEST]["preds"], cv_res[BEST]["truths"]
        fig_pva = px.scatter(x=bt, y=bp, labels={"x": "Actual (%)", "y": "Predicted (%)"},
                             title=f"Predicted vs Actual — {BEST}", template="plotly_white")
        lim = [min(bt.min(), bp.min()) - 1, max(bt.max(), bp.max()) + 1]
        fig_pva.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                          line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_pva, use_container_width=True)

        # Feature importance
        st.subheader("Feature Importance (Permutation, best model)")
        best_model, X_best = MODELS[BEST]
        best_model.fit(X_best, y_ml)
        perm = permutation_importance(best_model, X_best, y_ml, n_repeats=20, random_state=42)
        imp_df = pd.DataFrame({
            "feature": PRED_FEATS,
            "importance": perm.importances_mean,
            "std": perm.importances_std,
        }).sort_values("importance", ascending=False)

        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                         error_x="std", title=f"Feature Importance — {BEST}",
                         template="plotly_white", color="importance",
                         color_continuous_scale="RdYlGn")
        fig_imp.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.session_state["cv_results"] = cv_res
        st.session_state["imp_df"]     = imp_df
        st.session_state["TARGET_ML"]  = TARGET_ML
        st.success(f"✅ Best model: **{BEST}** | R² = {cv_res[BEST]['R2']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 8 · Summary":
    st.title("📋 Summary & Policy Interpretation")

    master       = st.session_state.get("master")
    latest       = st.session_state.get("latest")
    cluster_configs = st.session_state.get("cluster_configs")
    cv_res       = st.session_state.get("cv_results")
    imp_df       = st.session_state.get("imp_df")

    if master is None:
        st.warning("Run all modules first.")
        st.stop()

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    st.subheader("📐 Data Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("States", master["state"].nunique())
    c2.metric("Years", f"{master['year'].min()} – {master['year'].max()}")
    c3.metric("Observations", len(master))

    if cluster_configs and latest is not None:
        st.markdown("---")
        st.subheader("🔵 Clustering Evaluation")
        for label, feats, cluster_col, k, X_sc in cluster_configs:
            if cluster_col in latest.columns:
                labels_arr = latest[cluster_col].values
                sil = silhouette_score(X_sc, labels_arr)
                db  = davies_bouldin_score(X_sc, labels_arr)
                km_eval = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_eval.fit(X_sc)
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown(f"**{label} (k={k})**")
                col2.metric("Silhouette", f"{sil:.3f}")
                col3.metric("Davies-Bouldin", f"{db:.3f}")
                col4.metric("Inertia", f"{km_eval.inertia_:.1f}")

                st.write("States per cluster:")
                for c_id in sorted(latest[cluster_col].unique()):
                    states = latest[latest[cluster_col] == c_id]["state"].tolist()
                    st.write(f"  Cluster {c_id}: {', '.join(states)}")
                st.markdown("")

    if cv_res:
        st.markdown("---")
        st.subheader("🤖 Supervised Model Evaluation")
        TARGET_ML = st.session_state.get("TARGET_ML", T_UPP)
        BEST = max(cv_res, key=lambda k: cv_res[k]["R2"])
        rows = []
        for n, r in cv_res.items():
            rows.append({"Model": n, "RMSE": round(r["RMSE"],3), "MAE": round(r["MAE"],3), "R²": round(r["R2"],3)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.success(f"**Best model: {BEST}** — R² = {cv_res[BEST]['R2']:.3f}, RMSE = {cv_res[BEST]['RMSE']:.3f}")

    if imp_df is not None:
        st.markdown("---")
        st.subheader("🔑 Top Predictors")
        st.dataframe(imp_df.head(8).round(4), use_container_width=True)

    st.markdown("---")
    st.subheader("📝 Policy Interpretation Notes")
    st.text_area("Add your interpretation here", height=220, placeholder=
"""Example:
• Cluster 0 (Sabah, Kelantan, Terengganu) — highest poverty + largest HH size → lowest upper secondary completion
• Cluster 1 (KL, Selangor, Penang) — high income → near-100% completion

Key correlations:
• Poverty rate: r = ___ with primary dropout
• Household size: r = ___ — larger families = more economic pressure
• Median income: r = ___ (negative — higher income protects)

Policy recommendations:
1. Conditional cash transfers for Cluster 0 states
2. School feeding + transport subsidies for high-HH-size states
3. Upper secondary retention is most urgent (primary ≈100%)
""")
