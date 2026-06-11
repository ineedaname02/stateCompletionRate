"""
🌎 Education Analysis Dashboard
Malaysia (State-level) + Brazil (National) — COS40007 Theme 3: Smart Government
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🌎 Education Analysis Dashboard",
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

BZ_GDP      = "WB | GDP per capita (current US$)"
BZ_GINI     = "WB | Gini index"
BZ_FERT     = "WB | Fertility rate (total births per woman)"
BZ_ADOFERT  = "WB | Adolescent fertility rate (births per 1,000 women ages 15-19)"
BZ_POP      = "WB | Population, total"
BZ_POV215   = "WB | Poverty headcount ratio at $2.15/day (2017 PPP) (%)"
BZ_POVLM    = "WB | Poverty headcount ratio at lower-middle-income poverty line (%)"
BZ_POVUM    = "WB | Poverty headcount ratio at upper-middle-income poverty line (%)"
BZ_CR_LOW   = "UNESCO | Completion rate for lower secondary education (%)"
BZ_CR_PRI   = "UNESCO | Completion rate for primary education (%)"
BZ_CR_UPP   = "UNESCO | Completion rate for upper secondary education (%)"
BZ_ENRL_LOW = "UNESCO | Enrolment in lower secondary education (number)"
BZ_ENRL_PRI = "UNESCO | Enrolment in primary education (number)"
BZ_ENRL_UPP = "UNESCO | Enrolment in upper secondary education (number)"
BZ_TCH_LOW  = "UNESCO | Teachers in lower secondary education (number)"
BZ_TCH_PRI  = "UNESCO | Teachers in primary education (number)"
BZ_TCH_UPP  = "UNESCO | Teachers in upper secondary education (number)"

BZ_COMPLETION_COLS = [BZ_CR_PRI, BZ_CR_LOW, BZ_CR_UPP]
BZ_ECON_COLS       = [BZ_GDP, BZ_GINI, BZ_POV215, BZ_POVLM, BZ_POVUM,
                      BZ_FERT, BZ_ADOFERT]

for key in ["master", "latest", "cluster_configs", "cv_results", "imp_df",
            "bz_df", "bz_cv_results", "bz_imp_df", "bz_cluster_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.sidebar.title("📋 Navigation")
country = st.sidebar.radio("🌍 Country", ["🇲🇾 Malaysia", "🇧🇷 Brazil"], key="country_sel")

if country == "🇲🇾 Malaysia":
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Home", "📂 1 · Data Loading", "🧹 2 · Cleaning & Master",
         "📊 3 · EDA", "🔵 4 · K-Means Clustering", "🌿 5 · Hierarchical Clustering",
         "⏱ 6 · DTW Time-Series Clustering", "🤖 7 · Supervised Models", "📋 8 · Summary"],
        key="nav_page_my",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data directory**")
    DATA_DIR = st.sidebar.text_input("Path to raw CSVs", value="./raw_data/")
else:
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 BZ Home", "📂 BZ-1 · Load CSV", "📊 BZ-2 · EDA",
         "🔵 BZ-3 · K-Means (Time Periods)", "🌿 BZ-4 · Hierarchical",
         "⏱ BZ-5 · DTW", "🤖 BZ-6 · Supervised Models", "📋 BZ-7 · Summary"],
        key="nav_page_bz",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Brazil CSV path**")
    BZ_CSV_PATH = st.sidebar.text_input("Path to brazil_final_combined.csv",
                                         value="./brazil_final_combined.csv")

# ══════════════════════════════════════════════════════════════════════════════
# MALAYSIA PAGES
# ══════════════════════════════════════════════════════════════════════════════

if country == "🇲🇾 Malaysia":

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
| 7 | Supervised | Unified Forecasting — ML + ARIMA + Holt-Winters (expanding window) |
| 8 | Summary | Evaluation metrics & policy interpretation |
""")
        st.info("👈 Set your **Data directory** in the sidebar, then go to **1 · Data Loading**.")

    elif page == "📂 1 · Data Loading":
        st.title("📂 Data Loading")
        FILE_MAP = {
            "completion": "completion_school_state.csv",
            "poverty":    "hh_poverty_state.csv",
            "income":     "hh_income_state.csv",
            "households": "hh_lq_state.csv",
            "population": "population_state.csv",
            "fertility":  "fertility_state.csv",
            "teachers":   "teachers_district.csv",
            "schools":    "schools_district.csv",
            "enrolment":  "enrolment_school_district.csv",
        }
        st.subheader("File mapping")
        cols = st.columns(2)
        for i, (k, v) in enumerate(FILE_MAP.items()):
            cols[i % 2].text(f"  {k:15s} ← {v}")

        if st.button("▶ Load all files", type="primary"):
            import os
            loaded, errors = {}, []
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
            rows = [{"Dataset": n, "Rows": d.shape[0], "Cols": d.shape[1],
                     "Columns": ", ".join(d.columns[:6].tolist()) + ("…" if d.shape[1] > 6 else "")}
                    for n, d in loaded.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        elif "raw" in st.session_state and st.session_state["raw"]:
            st.info("Data already loaded. Proceed to **2 · Cleaning & Master**.")
            for name, df in st.session_state["raw"].items():
                st.write(f"**{name}**: {df.shape[0]:,} rows × {df.shape[1]} cols")

    elif page == "🧹 2 · Cleaning & Master":
        st.title("🧹 Cleaning & Master Panel")
        if "raw" not in st.session_state or not st.session_state["raw"]:
            st.warning("⚠️ Load data first (Module 1).")
            st.stop()
        raw = st.session_state["raw"]
        with st.expander("⚙️ Preprocessing options", expanded=True):
            min_year = st.number_input("Min year", value=2016, step=1)
            max_year = st.number_input("Max year", value=2022, step=1)
            drop_missing = st.checkbox("Drop rows with missing district-level values", value=True)
            save_path = st.text_input("Save master CSV (optional)", value="./master.csv")

        if st.button("▶ Build master panel", type="primary"):
            with st.spinner("Processing…"):
                try:
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
                        "primary": T_PRI, "secondary_lower": T_LOW, "secondary_upper": T_UPP,
                    })

                    df_enrolment = raw["enrolment"]
                    ENROL_COL = next(
                        (c for c in df_enrolment.columns
                         if c not in ["state","district","stage","sex","date","year","type"]
                         and df_enrolment[c].dtype in ["int64","float64"]), None)
                    if "district" in df_enrolment.columns:
                        enrol_st = df_enrolment[df_enrolment["district"] == "All Districts"].copy()
                    else:
                        enrol_st = df_enrolment.copy()
                    enrol_st = enrol_st[
                        (enrol_st["state"].isin(STATES_16)) &
                        (enrol_st["stage"].isin(["primary", "secondary"]))
                    ].copy()
                    if "sex" in enrol_st.columns:
                        enrol_st = enrol_st[enrol_st["sex"].isin(["both","Both","overall","Overall"])]
                    if ENROL_COL and "stage" in enrol_st.columns:
                        enrol_st = enrol_st.pivot_table(
                            index=["state","year"], columns="stage", values=ENROL_COL
                        ).reset_index()
                        enrol_st.columns = ["state","year","enrolment_primary","enrolment_secondary"]

                    df_schools = raw["schools"]
                    SCHOOL_COL = next(
                        (c for c in df_schools.columns
                         if c not in ["state","district","stage","sex","date","year","type"]
                         and df_schools[c].dtype in ["int64","float64"]), None)
                    if "district" in df_schools.columns:
                        schools_st = df_schools[df_schools["district"] == "All Districts"].copy()
                    else:
                        schools_st = df_schools.copy()
                    schools_st = schools_st[
                        (schools_st["state"].isin(STATES_16)) &
                        (schools_st["stage"].isin(["primary","secondary"]))
                    ].copy()
                    if SCHOOL_COL and "stage" in schools_st.columns:
                        schools_st = schools_st.pivot_table(
                            index=["state","year"], columns="stage", values=SCHOOL_COL
                        ).reset_index()
                        schools_st.columns = ["state","year","schools_primary","schools_secondary"]

                    df_teachers = raw["teachers"]
                    TEACH_COL = next(
                        (c for c in df_teachers.columns
                         if c not in ["state","district","stage","sex","date","year","type"]
                         and df_teachers[c].dtype in ["int64","float64"]), None)
                    if "district" in df_teachers.columns:
                        teachers_st = df_teachers[df_teachers["district"] == "All Districts"].copy()
                    else:
                        teachers_st = df_teachers.copy()
                    teachers_st = teachers_st[
                        (teachers_st["state"].isin(STATES_16)) &
                        (teachers_st["stage"].isin(["primary","secondary"]))
                    ].copy()
                    if "sex" in teachers_st.columns:
                        teachers_st = teachers_st[teachers_st["sex"].isin(["both","Both","overall","Overall"])]
                    if TEACH_COL and "stage" in teachers_st.columns:
                        teachers_st = teachers_st.pivot_table(
                            index=["state","year"], columns="stage", values=TEACH_COL
                        ).reset_index()
                        teachers_st.columns = ["state","year","teachers_primary","teachers_secondary"]

                    poverty = raw["poverty"][raw["poverty"]["state"].isin(STATES_16)].copy()
                    pov_cols = ["state","year"] + [c for c in ["poverty_absolute","poverty_hardcore","poverty_relative"] if c in poverty.columns]
                    poverty = poverty[pov_cols]

                    income = raw["income"][raw["income"]["state"].isin(STATES_16)].copy()
                    median_col = next((c for c in income.columns if "median" in c.lower()), None)
                    mean_col   = next((c for c in income.columns if "mean"   in c.lower()), None)
                    keep = ["state","year"] + [c for c in [median_col, mean_col] if c]
                    income = income[keep].copy()
                    if median_col: income.rename(columns={median_col: "income_median"}, inplace=True)
                    if mean_col:   income.rename(columns={mean_col: "income_mean"}, inplace=True)

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

                    master = comp_wide.copy()
                    for df_m in [teachers_st, schools_st, enrol_st, poverty, income,
                                  hh_pop[["state","year","avg_hh_size"]], fertility_pivot]:
                        if df_m is not None and len(df_m) > 0:
                            master = master.merge(df_m, on=["state","year"], how="left")
                    master = master.sort_values(["state","year"]).reset_index(drop=True)

                    # ── Drop missing district-level rows FIRST ─────────────
                    DISTRICT_COLS = [c for c in [
                        "teachers_primary", "teachers_secondary",
                        "schools_primary", "schools_secondary",
                        "enrolment_primary", "enrolment_secondary"
                    ] if c in master.columns]

                    if drop_missing and DISTRICT_COLS:
                        before = len(master)
                        missing_rows = master.loc[
                            master[DISTRICT_COLS].isnull().any(axis=1), ["state", "year"]
                        ].drop_duplicates()
                        master = master.merge(missing_rows, on=["state","year"], how="left", indicator=True)
                        master = master[master["_merge"] == "left_only"].drop(columns=["_merge"])
                        master = master.sort_values(["state","year"]).reset_index(drop=True)
                        st.info(f"Dropped {before - len(master)} rows with missing district-level values.")

                    # ── Interpolate sparse socioeconomic columns ────────────
                    SPARSE = [c for c in master.columns
                              if c not in ["state","year"] and master[c].isna().sum() > 0]
                    master[SPARSE] = master.groupby("state")[SPARSE].transform(
                        lambda s: s.interpolate(method="linear", limit_direction="both")
                    )

                    # ── Derived features ───────────────────────────────────
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

                    st.session_state["master"] = master
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

    elif page == "📊 3 · EDA":
        st.title("📊 Exploratory Data Analysis")
        master = st.session_state.get("master")
        if master is None:
            st.warning("⚠️ Build master panel first (Module 2).")
            st.stop()

        tabs = st.tabs(["Completion by State", "Trends Over Time", "Correlation Heatmap", "Scatter Plots"])

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
                fig = px.bar(sdf, x=col, y="state", orientation="h", color="color",
                             color_discrete_map={"Below median": "#D85A30", "Above median": "#1D9E75"},
                             title=f"{stage_sel} Completion Rate by State",
                             labels={col: "Mean Completion Rate (%)", "state": "State"},
                             template="plotly_white", height=550)
                fig.add_vline(x=median_val, line_dash="dash", line_color="gray",
                              annotation_text=f"Median {median_val:.1f}%")
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("5 lowest states")
                show_cols = [c for c in ["state", col, "poverty"] if c in state_avg.columns]
                st.dataframe(state_avg.sort_values(col)[show_cols].head(5).round(2), use_container_width=True)

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
                fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                                 zmin=-1, zmax=1, title="Correlation Matrix", height=600)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Select at least 2 features.")

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
                selected_var = st.selectbox("Select predictor variable", PRESSURE_VARS_EDA, key="scatter_var_sel")
                col_idx = PRESSURE_VARS_EDA.index(selected_var)
                dot_color = PALETTE[col_idx % len(PALETTE)]
                cols = st.columns(3)
                for col_widget, (target, title) in zip(cols, TARGETS_EDA):
                    sub = master[[selected_var, target, 'state']].dropna()
                    fig, ax = plt.subplots(figsize=(4.5, 4))
                    ax.scatter(sub[selected_var], sub[target], color=dot_color, alpha=0.5, s=30)
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
                fig_ev = make_subplots(1, 3, subplot_titles=["Elbow (Inertia)", "Silhouette ↑", "Davies-Bouldin ↓"])
                ks = list(K_RANGE)
                fig_ev.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", line_color="#185FA5"), 1, 1)
                fig_ev.add_vline(x=k_elbow, line_dash="dash", line_color="red", row=1, col=1)
                fig_ev.add_trace(go.Scatter(x=ks, y=silhouettes, mode="lines+markers", line_color="#1D9E75"), 1, 2)
                fig_ev.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=2)
                fig_ev.add_trace(go.Scatter(x=ks, y=db_scores, mode="lines+markers", line_color="#D85A30"), 1, 3)
                fig_ev.update_layout(height=320, showlegend=False, template="plotly_white",
                                      title=f"{label} — Elbow k={k_elbow} | Best silhouette k={optimal_k} ({max(silhouettes):.3f})")
                st.plotly_chart(fig_ev, use_container_width=True)
                k_use = st.number_input(f"Use k for {label}", value=optimal_k, min_value=2, max_value=int(k_max), key=f"k_use_{col_prefix}")
                km_final = KMeans(n_clusters=int(k_use), random_state=42, n_init=int(n_init))
                latest[col_prefix] = km_final.fit_predict(X_sc)
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_sc)
                var_ex = pca.explained_variance_ratio_
                pca_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1],
                                       "Cluster": latest[col_prefix].astype(str), "State": latest["state"]})
                fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", text="State",
                                     title=f"{label} PCA — {var_ex[0]*100:.1f}% + {var_ex[1]*100:.1f}% variance",
                                     template="plotly_white", height=400)
                fig_pca.update_traces(textposition="top center", marker_size=10)
                st.plotly_chart(fig_pca, use_container_width=True)
                st.write("**Cluster profiles (mean):**")
                st.dataframe(latest.groupby(col_prefix)[feats].mean().round(2), use_container_width=True)
                st.write("**States per cluster:**")
                for c_id in sorted(latest[col_prefix].unique()):
                    states = latest[latest[col_prefix] == c_id]["state"].tolist()
                    st.write(f"  Cluster {c_id}: {', '.join(states)}")
                sil = silhouette_score(X_sc, latest[col_prefix])
                db  = davies_bouldin_score(X_sc, latest[col_prefix])
                st.metric(f"Silhouette (k={k_use})", f"{sil:.3f}")
                st.metric(f"Davies-Bouldin (k={k_use})", f"{db:.3f}")
                CLUSTER_CONFIGS.append((label, feats, col_prefix, int(k_use), X_sc))
            st.session_state["latest"] = latest
            st.session_state["cluster_configs"] = CLUSTER_CONFIGS
            st.success("✅ K-Means clustering complete.")
        elif st.session_state.get("latest") is not None:
            st.info("K-Means already run. Re-run to update.")

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
                st.metric("Cophenetic Correlation", f"{c_stat:.4f}")
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
                hc_labels = fcluster(Z, t=k, criterion="maxclust") - 1
                hc_col = f"hc_{cluster_col}"
                latest[hc_col] = hc_labels
                st.write(f"**HC Clusters (k={k}, {linkage_method}):**")
                for c_id in sorted(latest[hc_col].unique()):
                    states = latest[latest[hc_col] == c_id]["state"].tolist()
                    st.write(f"  HC Cluster {c_id}: {', '.join(states)}")
                ari = adjusted_rand_score(latest[cluster_col], latest[hc_col])
                col1, col2 = st.columns(2)
                col1.metric("ARI vs K-Means", f"{ari:.4f}")
                if ari > 0.7:
                    col2.success("Strong agreement — cluster structure is stable.")
                elif ari > 0.4:
                    col2.warning("Moderate agreement — some states differ.")
                else:
                    col2.error("Low agreement — methods disagree.")
            st.session_state["latest"] = latest
            st.success("✅ Hierarchical clustering complete.")

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
        col1, col2, col3 = st.columns(3)
        ts_target = col1.selectbox("Time-series target", ts_options, format_func=lambda x: ts_map.get(x, x))
        k_min_dtw = col2.number_input("Min k", value=2, min_value=2)
        k_max_dtw = col3.number_input("Max k", value=6, min_value=3)

        if st.button("▶ Run DTW Clustering", type="primary"):
            YEARS_TS = sorted(master["year"].unique())
            states_actual = master["state"].unique()
            ts_list, state_order = [], []
            for state in states_actual:
                state_df = (master[master["state"] == state]
                            .groupby("year")[ts_target].mean().reindex(YEARS_TS))
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
                km_dtw = TimeSeriesKMeans(n_clusters=k_dtw, metric="dtw", max_iter=10, random_state=42, n_jobs=-1)
                dtw_lbl = km_dtw.fit_predict(X_ts_scaled)
                sil = silhouette_score(dtw_dist_matrix, dtw_lbl, metric="precomputed")
                dtw_sil_scores[k_dtw] = sil
                prog.progress((i + 1) / len(DTW_K_RANGE))
            fig_sil = px.line(x=list(DTW_K_RANGE), y=list(dtw_sil_scores.values()),
                              markers=True, labels={"x": "k", "y": "Silhouette"},
                              title="DTW Silhouette Score vs k", template="plotly_white")
            st.plotly_chart(fig_sil, use_container_width=True)
            OPTIMAL_K_DTW = max(dtw_sil_scores, key=dtw_sil_scores.get)
            st.info(f"Best k = **{OPTIMAL_K_DTW}** (Silhouette = {dtw_sil_scores[OPTIMAL_K_DTW]:.4f})")
            k_dtw_use = st.number_input("Use k", value=OPTIMAL_K_DTW, min_value=2, max_value=int(k_max_dtw))
            km_dtw_final = TimeSeriesKMeans(n_clusters=int(k_dtw_use), metric="dtw", max_iter=20, random_state=42, n_jobs=-1)
            dtw_labels_final = km_dtw_final.fit_predict(X_ts_scaled)
            fig_ts = make_subplots(1, int(k_dtw_use),
                                   subplot_titles=[f"DTW Cluster {c}" for c in range(int(k_dtw_use))],
                                   shared_yaxes=True)
            for c_id in range(int(k_dtw_use)):
                member_states = [state_order[i] for i, l in enumerate(dtw_labels_final) if l == c_id]
                centroid = km_dtw_final.cluster_centers_[c_id].flatten()
                for state in member_states:
                    idx = state_order.index(state)
                    fig_ts.add_trace(go.Scatter(
                        x=YEARS_TS, y=X_ts_scaled[idx].flatten(), mode="lines", name=state,
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
            if latest is not None:
                dtw_lookup = {state_order[i]: int(dtw_labels_final[i]) for i in range(len(state_order))}
                latest[f"dtw_{ts_target}"] = latest["state"].map(dtw_lookup)
                st.session_state["latest"] = latest
            st.success("✅ DTW clustering complete.")

    # ── PAGE 7 — MALAYSIA UNIFIED FORECASTING ────────────────────────────────
    elif page == "🤖 7 · Supervised Models":
        st.title("🤖 Unified Forecasting — Malaysia")
        st.markdown("""
> **Method:** Expanding window, 1-step ahead per state per target.  
> **Models:** ML (Ridge / Random Forest / Gradient Boosting) + ARIMA (auto) + Holt-Winters  
> Results reported as RMSE, MAE, R² aggregated across all states and test years.
""")
        master = st.session_state.get("master")
        if master is None:
            st.warning("⚠️ Build master panel first (Module 2).")
            st.stop()

        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        TARGETS_MY = {
            "Primary":         T_PRI,
            "Lower_Secondary": T_LOW,
            "Upper_Secondary": T_UPP,
        }
        EXOG_FEATS_MY = [f for f in [
            'poverty_absolute', 'poverty_relative', 'avg_hh_size',
            'tfr_tfr', 'income_median', 'str_primary', 'str_secondary',
            'schools_per_1k_primary', 'schools_per_1k_secondary',
            'education_pressure_primary', 'education_pressure_secondary', 'urban_pressure_proxy'
        ] if f in master.columns]

        st.subheader("⚙️ Configuration")
        c1, c2 = st.columns(2)
        min_train = c1.number_input("Min training rows per state", value=3, min_value=2)
        sel_targets = c2.multiselect(
            "Targets to forecast",
            list(TARGETS_MY.keys()),
            default=list(TARGETS_MY.keys())
        )
        models_to_run = st.multiselect(
            "Models",
            ["ML_Ridge", "ML_RandomForest", "ML_GradientBoosting", "ARIMA", "HoltWinters"],
            default=["ML_Ridge", "ML_RandomForest", "ML_GradientBoosting", "ARIMA", "HoltWinters"]
        )

        if st.button("▶ Run Unified Forecasting", type="primary"):
            if not sel_targets:
                st.error("Select at least 1 target.")
                st.stop()

            def run_forecasting_my(target_name, target_col, master_df, exog_feats, min_tr):
                results = []
                feats = [f for f in exog_feats if f in master_df.columns]
                MIN_TRAIN = int(min_tr)

                for state in master_df["state"].unique():
                    state_df = master_df[master_df["state"] == state].sort_values("year")
                    ts = state_df.set_index("year")[target_col].dropna()
                    if len(ts) < MIN_TRAIN + 1:
                        continue

                    for i in range(len(ts) - 1):
                        train_end_year = ts.index[i]
                        test_year      = ts.index[i + 1]
                        train_df = state_df[state_df["year"] <= train_end_year]
                        test_df  = state_df[state_df["year"] == test_year]
                        if len(test_df) == 0:
                            continue
                        y_true = test_df[target_col].values[0]
                        if pd.isna(y_true):
                            continue

                        # ML models
                        train_ml = train_df[feats + [target_col]].dropna()
                        if len(train_ml) >= MIN_TRAIN and "ML_Ridge" in models_to_run or \
                           len(train_ml) >= MIN_TRAIN and "ML_RandomForest" in models_to_run or \
                           len(train_ml) >= MIN_TRAIN and "ML_GradientBoosting" in models_to_run:
                            test_ml = test_df[feats].dropna()
                            if len(test_ml) > 0 and len(train_ml) >= MIN_TRAIN:
                                scaler = StandardScaler()
                                X_tr = scaler.fit_transform(train_ml[feats])
                                X_te = scaler.transform(test_ml[feats])
                                y_tr = train_ml[target_col]
                                ml_models = {}
                                if "ML_Ridge" in models_to_run:
                                    ml_models["ML_Ridge"] = Ridge(alpha=1.0)
                                if "ML_RandomForest" in models_to_run:
                                    ml_models["ML_RandomForest"] = RandomForestRegressor(
                                        n_estimators=100, max_depth=3, random_state=42)
                                if "ML_GradientBoosting" in models_to_run:
                                    ml_models["ML_GradientBoosting"] = GradientBoostingRegressor(
                                        n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
                                for mname, model in ml_models.items():
                                    try:
                                        model.fit(X_tr, y_tr)
                                        results.append({
                                            "state": state, "test_year": test_year,
                                            "target": target_name, "model": mname,
                                            "true": y_true, "pred": model.predict(X_te)[0]
                                        })
                                    except:
                                        pass

                        # ARIMA + Holt-Winters
                        ts_train = ts[ts.index <= train_end_year]
                        if len(ts_train) >= 3:
                            if "ARIMA" in models_to_run:
                                try:
                                    from pmdarima import auto_arima
                                    am = auto_arima(ts_train, seasonal=False, stepwise=True,
                                                    suppress_warnings=True, error_action='ignore',
                                                    max_p=2, max_q=2, max_d=1)
                                    results.append({
                                        "state": state, "test_year": test_year,
                                        "target": target_name, "model": "ARIMA",
                                        "true": y_true, "pred": am.predict(n_periods=1)[0]
                                    })
                                except:
                                    pass
                            if "HoltWinters" in models_to_run:
                                try:
                                    hw = ExponentialSmoothing(ts_train, trend='add', seasonal=None).fit()
                                    results.append({
                                        "state": state, "test_year": test_year,
                                        "target": target_name, "model": "HoltWinters",
                                        "true": y_true, "pred": hw.forecast(1).iloc[-1]
                                    })
                                except:
                                    pass
                return pd.DataFrame(results)

            all_results = []
            prog = st.progress(0)
            for idx, target_name in enumerate(sel_targets):
                target_col = TARGETS_MY[target_name]
                st.write(f"Running **{target_name}**…")
                res_df = run_forecasting_my(target_name, target_col, master,
                                            EXOG_FEATS_MY, min_train)
                if len(res_df) == 0:
                    st.warning(f"No valid forecasts for {target_name}.")
                else:
                    all_results.append(res_df)
                prog.progress((idx + 1) / len(sel_targets))

            if not all_results:
                st.error("No results generated across any target.")
                st.stop()

            full_results = pd.concat(all_results, ignore_index=True)

            st.subheader("📊 Results — RMSE")
            pivot_rmse = full_results.groupby(["target","model"]).apply(
                lambda g: pd.Series({"RMSE": np.sqrt(mean_squared_error(g["true"], g["pred"]))})
            ).reset_index().pivot(index="target", columns="model", values="RMSE")
            st.dataframe(pivot_rmse.round(3), use_container_width=True)

            st.subheader("📊 Results — MAE")
            pivot_mae = full_results.groupby(["target","model"]).apply(
                lambda g: pd.Series({"MAE": mean_absolute_error(g["true"], g["pred"])})
            ).reset_index().pivot(index="target", columns="model", values="MAE")
            st.dataframe(pivot_mae.round(3), use_container_width=True)

            st.subheader("📊 Results — R²")
            pivot_r2 = full_results.groupby(["target","model"]).apply(
                lambda g: pd.Series({"R2": r2_score(g["true"], g["pred"])})
            ).reset_index().pivot(index="target", columns="model", values="R2")
            st.dataframe(pivot_r2.round(3), use_container_width=True)

            # Best per target by R² and MAE
            comparison = full_results.groupby(["target","model"]).apply(
                lambda g: pd.Series({
                    "RMSE": np.sqrt(mean_squared_error(g["true"], g["pred"])),
                    "MAE":  mean_absolute_error(g["true"], g["pred"]),
                    "R2":   r2_score(g["true"], g["pred"])
                })
            ).reset_index()

            st.subheader("🏆 Best model per target")
            col1, col2 = st.columns(2)
            best_r2  = comparison.loc[comparison.groupby("target")["R2"].idxmax(),  ["target","model","R2"]]
            best_mae = comparison.loc[comparison.groupby("target")["MAE"].idxmin(), ["target","model","MAE"]]
            col1.write("**By highest R²:**")
            col1.dataframe(best_r2.round(4), use_container_width=True)
            col2.write("**By lowest MAE:**")
            col2.dataframe(best_mae.round(4), use_container_width=True)

            # Bar chart comparison
            fig_bar = px.bar(
                comparison.melt(id_vars=["target","model"], value_vars=["RMSE","MAE"]),
                x="model", y="value", color="target", barmode="group",
                facet_col="variable", template="plotly_white", height=420,
                title="Model Comparison — RMSE & MAE by Target",
                color_discrete_sequence=PALETTE
            )
            fig_bar.update_xaxes(tickangle=30)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.session_state["cv_results"] = {
                "full_results": full_results,
                "comparison": comparison,
                "pivot_rmse": pivot_rmse,
                "pivot_mae": pivot_mae,
                "pivot_r2": pivot_r2,
            }
            st.success("✅ Unified forecasting complete.")

        elif st.session_state.get("cv_results") is not None:
            st.info("Forecasting already run. Re-run to update.")
            cv = st.session_state["cv_results"]
            if isinstance(cv, dict) and "pivot_r2" in cv:
                st.subheader("R² (cached)")
                st.dataframe(cv["pivot_r2"].round(3), use_container_width=True)

    elif page == "📋 8 · Summary":
        st.title("📋 Summary & Policy Interpretation")
        master          = st.session_state.get("master")
        latest          = st.session_state.get("latest")
        cluster_configs = st.session_state.get("cluster_configs")
        cv_store        = st.session_state.get("cv_results")
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

        if cv_store is not None and isinstance(cv_store, dict) and "comparison" in cv_store:
            st.markdown("---")
            st.subheader("🤖 Forecasting Evaluation")
            comp = cv_store["comparison"]
            st.dataframe(comp.round(4), use_container_width=True)
            best_r2 = comp.loc[comp.groupby("target")["R2"].idxmax(), ["target","model","R2","MAE","RMSE"]]
            st.success("Best models per target (highest R²):")
            st.dataframe(best_r2.round(4), use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Policy Interpretation Notes")
        st.text_area("Add your interpretation here", height=220, placeholder=
"""Example:
• Cluster 0 (Sabah, Kelantan, Terengganu) — highest poverty + largest HH size → lowest upper secondary completion
• Cluster 1 (KL, Selangor, Penang) — high income → near-100% completion
""")


# ══════════════════════════════════════════════════════════════════════════════
# BRAZIL PAGES
# ══════════════════════════════════════════════════════════════════════════════

else:

    if page == "🏠 BZ Home":
        st.title("🇧🇷 Brazil Education & Socioeconomic Analysis")
        st.markdown("""
> **Research Question:**  
> *Do economic pressures (poverty, inequality, GDP per capita, fertility) explain  
> variation in school completion rates across time in Brazil, and can we identify  
> distinct developmental eras or regime shifts in the data?*

**Pipeline (mirrors Malaysia approach — national time-series instead of state cross-section)**

| Step | Module | Description |
|------|--------|-------------|
| BZ-1 | Load CSV | Load `brazil_final_combined.csv` |
| BZ-2 | EDA | Trends, correlations, scatter plots |
| BZ-3 | K-Means | Cluster years into developmental eras |
| BZ-4 | Hierarchical | Dendrogram over years |
| BZ-5 | DTW | Time-series shape clustering per indicator group |
| BZ-6 | Supervised | Unified Forecasting — ML + ARIMA + Holt-Winters (expanding window over years) |
| BZ-7 | Summary | Metrics & policy interpretation |
""")
        st.info("👈 Set path to **brazil_final_combined.csv** in the sidebar, then go to **BZ-1 · Load CSV**.")

    # ── BZ-1 — LOAD + DROP MISSING ────────────────────────────────────────────
    elif page == "📂 BZ-1 · Load CSV":
        st.title("📂 BZ-1 · Load Brazil Data")

        with st.expander("⚙️ Loading options", expanded=True):
            drop_bz_missing = st.checkbox("Drop years with any missing value after loading", value=True)
            interp_bz = st.checkbox("Interpolate remaining sparse columns before dropping", value=False)
            st.caption("Tip: try interpolate first, then drop — this fills sparse WB columns before checking for gaps.")

        # Define a path for the master CSV (adjust as needed)
        MASTER_BZ_PATH = "brazil_master_2000plus.csv"

        if st.button("▶ Load brazil_final_combined.csv", type="primary"):
            try:
                df = pd.read_csv(BZ_CSV_PATH, index_col="Year")
                df.index = df.index.astype(int)
                df.index.name = "Year"

                # Step 1: Drop years before 2000
                original_len = len(df)
                df = df[df.index >= 2000]
                st.info(f"Dropped {original_len - len(df)} year(s) before 2000. Kept {len(df)} rows (≥2000).")

                # Step 2: Interpolate missing values (for years ≥2000)
                cols_with_nan = [c for c in df.columns if df[c].isna().any()]
                if cols_with_nan:
                    df[cols_with_nan] = df[cols_with_nan].interpolate(
                        method="linear", limit_direction="both"
                    )
                    st.info(f"Interpolated {len(cols_with_nan)} column(s).")

                # Optional: Drop rows that still have any missing after interpolation
                # (uncomment if you want a completely clean dataset)
                # before_drop = len(df)
                # df = df.dropna()
                # st.info(f"Dropped {before_drop - len(df)} row(s) with remaining missing values.")

                # Step 3: Check final missing
                remaining_missing = df.isnull().sum()
                remaining_missing = remaining_missing[remaining_missing > 0]
                if len(remaining_missing) > 0:
                    st.warning("⚠️ Still have missing values after cleaning:")
                    st.dataframe(remaining_missing.rename("NaN count").reset_index())
                else:
                    st.success("✅ No missing values remaining.")

                # ─────────────────────────────────────────────
                # Save to master CSV (overwrites if exists)
                df.to_csv(MASTER_BZ_PATH)
                st.success(f"💾 Saved cleaned data to `{MASTER_BZ_PATH}`")
                # ─────────────────────────────────────────────

                st.session_state["bz_df"] = df
                st.dataframe(df.head(10).round(4), use_container_width=True)

            except Exception as e:
                st.error(f"Could not load file: {e}")

        elif st.session_state["bz_df"] is not None:
            df = st.session_state["bz_df"]
            st.info(f"Already loaded: {df.shape[0]} years × {df.shape[1]} cols. Proceed to BZ-2.")
            st.dataframe(df.head(), use_container_width=True)

    elif page == "📊 BZ-2 · EDA":
        st.title("📊 BZ-2 · Exploratory Data Analysis — Brazil")
        df = st.session_state.get("bz_df")
        if df is None:
            st.warning("⚠️ Load data first (BZ-1).")
            st.stop()

        tabs = st.tabs(["Completion Trends", "Economic Trends", "Correlation Heatmap",
                         "Scatter: Completion vs Economics", "Enrolment & Teachers"])

        with tabs[0]:
            st.subheader("Completion Rate Trends Over Time")
            cr_cols = [c for c in BZ_COMPLETION_COLS if c in df.columns]
            if cr_cols:
                melt = df[cr_cols].reset_index().melt(id_vars="Year", var_name="Stage", value_name="Value")
                melt["Stage"] = melt["Stage"].str.replace("UNESCO | Completion rate for ", "", regex=False).str.replace(" (%)", "", regex=False)
                fig = px.line(melt, x="Year", y="Value", color="Stage",
                              title="Brazil: Completion Rates", template="plotly_white", markers=True, height=420)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df[cr_cols].describe().round(4), use_container_width=True)
            else:
                st.warning("No completion columns found.")

        with tabs[1]:
            st.subheader("Economic Indicators Over Time")
            econ_avail = [c for c in BZ_ECON_COLS if c in df.columns]
            sel_econ = st.multiselect("Select indicators", econ_avail, default=econ_avail[:4])
            if sel_econ:
                melt2 = df[sel_econ].reset_index().melt(id_vars="Year", var_name="Indicator", value_name="Value")
                melt2["Indicator"] = melt2["Indicator"].str.replace("WB | ", "", regex=False)
                fig2 = px.line(melt2, x="Year", y="Value", color="Indicator",
                               title="Brazil: Economic Indicators", template="plotly_white", markers=True, height=460)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[2]:
            st.subheader("Correlation Heatmap")
            num_cols = [c for c in df.columns if df[c].notna().sum() > 10]
            sel_corr = st.multiselect("Columns", num_cols, default=num_cols[:12])
            if len(sel_corr) >= 2:
                corr = df[sel_corr].corr()
                short_labels = [c.replace("WB | ", "WB:").replace("UNESCO | ", "UN:") for c in sel_corr]
                corr.index = short_labels
                corr.columns = short_labels
                fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                                 zmin=-1, zmax=1, title="Correlation Matrix", height=600)
                st.plotly_chart(fig3, use_container_width=True)

        with tabs[3]:
            st.subheader("Completion vs Economic Predictors")
            from scipy import stats as scipy_stats
            import matplotlib.pyplot as plt
            import io
            econ_avail2 = [c for c in BZ_ECON_COLS if c in df.columns]
            cr_avail    = [c for c in BZ_COMPLETION_COLS if c in df.columns]
            if not econ_avail2 or not cr_avail:
                st.warning("Not enough columns.")
            else:
                sel_x = st.selectbox("X-axis (economic variable)", econ_avail2,
                                     format_func=lambda c: c.replace("WB | ", ""))
                cols_w = st.columns(len(cr_avail))
                for col_w, cr_col in zip(cols_w, cr_avail):
                    sub = df[[sel_x, cr_col]].dropna()
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    ax.scatter(sub[sel_x], sub[cr_col], color=PALETTE[0], alpha=0.6, s=30)
                    if len(sub) > 2:
                        sl, ic, r, p, _ = scipy_stats.linregress(sub[sel_x], sub[cr_col])
                        xr = np.linspace(sub[sel_x].min(), sub[sel_x].max(), 50)
                        ax.plot(xr, sl * xr + ic, 'r--', lw=1.4, alpha=0.8)
                        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                        ax.set_title(f"r={r:.2f} {sig}", fontsize=9)
                    ax.set_xlabel(sel_x.replace("WB | ",""), fontsize=7)
                    ax.set_ylabel(cr_col.replace("UNESCO | ",""), fontsize=7)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.25)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig)
                    col_w.subheader(cr_col.replace("UNESCO | Completion rate for ","").replace(" (%)",""))
                    col_w.image(buf, use_container_width=True)

        with tabs[4]:
            st.subheader("Enrolment & Teacher Trends")
            enrl_cols = [c for c in [BZ_ENRL_PRI, BZ_ENRL_LOW, BZ_ENRL_UPP] if c in df.columns]
            tch_cols  = [c for c in [BZ_TCH_PRI, BZ_TCH_LOW, BZ_TCH_UPP] if c in df.columns]
            if enrl_cols:
                melt_e = df[enrl_cols].reset_index().melt(id_vars="Year", var_name="Stage", value_name="Value")
                melt_e["Stage"] = melt_e["Stage"].str.replace("UNESCO | Enrolment in ", "", regex=False).str.replace(" (number)", "", regex=False)
                fig_e = px.line(melt_e, x="Year", y="Value", color="Stage",
                                title="Enrolment by Stage", template="plotly_white", markers=True)
                st.plotly_chart(fig_e, use_container_width=True)
            if tch_cols:
                melt_t = df[tch_cols].reset_index().melt(id_vars="Year", var_name="Stage", value_name="Value")
                melt_t["Stage"] = melt_t["Stage"].str.replace("UNESCO | Teachers in ", "", regex=False).str.replace(" (number)", "", regex=False)
                fig_t = px.line(melt_t, x="Year", y="Value", color="Stage",
                                title="Teachers by Stage", template="plotly_white", markers=True)
                st.plotly_chart(fig_t, use_container_width=True)

    elif page == "🔵 BZ-3 · K-Means (Time Periods)":
        st.title("🔵 BZ-3 · K-Means — Clustering Years into Developmental Eras")
        df = st.session_state.get("bz_df")
        if df is None:
            st.warning("⚠️ Load data first (BZ-1).")
            st.stop()

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        from kneed import KneeLocator

        all_cols = [c for c in df.columns if df[c].notna().sum() > len(df) * 0.5]
        st.subheader("⚙️ Feature selection")
        default_feats = [c for c in BZ_ECON_COLS + BZ_COMPLETION_COLS if c in all_cols]
        sel_feats = st.multiselect("Features", all_cols, default=default_feats)
        c1, c2, c3 = st.columns(3)
        k_min = c1.number_input("Min k", value=2, min_value=2)
        k_max = c2.number_input("Max k", value=7, min_value=3)
        n_init = c3.number_input("n_init", value=20, min_value=5, step=5)

        if st.button("▶ Run K-Means on years", type="primary"):
            if not sel_feats:
                st.error("Select at least 1 feature.")
                st.stop()
            sub = df[sel_feats].dropna()
            years_used = sub.index.tolist()
            X = sub.values
            sc = StandardScaler()
            X_sc = sc.fit_transform(X)
            K_RANGE = range(int(k_min), min(int(k_max) + 1, len(sub)))
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
            fig_ev = make_subplots(1, 3, subplot_titles=["Elbow (Inertia)", "Silhouette ↑", "Davies-Bouldin ↓"])
            ks = list(K_RANGE)
            fig_ev.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", line_color="#185FA5"), 1, 1)
            fig_ev.add_vline(x=k_elbow, line_dash="dash", line_color="red", row=1, col=1)
            fig_ev.add_trace(go.Scatter(x=ks, y=silhouettes, mode="lines+markers", line_color="#1D9E75"), 1, 2)
            fig_ev.add_vline(x=optimal_k, line_dash="dash", line_color="red", row=1, col=2)
            fig_ev.add_trace(go.Scatter(x=ks, y=db_scores, mode="lines+markers", line_color="#D85A30"), 1, 3)
            fig_ev.update_layout(height=300, showlegend=False, template="plotly_white",
                                  title=f"Elbow k={k_elbow} | Best silhouette k={optimal_k} ({max(silhouettes):.3f})")
            st.plotly_chart(fig_ev, use_container_width=True)
            k_use = st.number_input("Use k", value=optimal_k, min_value=2, max_value=int(k_max), key="bz_k_use")
            km_final = KMeans(n_clusters=int(k_use), random_state=42, n_init=int(n_init))
            labels = km_final.fit_predict(X_sc)
            era_df = pd.DataFrame({"Year": years_used, "Era": labels.astype(str)})
            fig_era = px.scatter(era_df, x="Year", y=[0]*len(era_df), color="Era",
                                 symbol="Era", size=[12]*len(era_df),
                                 title=f"Brazil Developmental Eras (k={k_use})",
                                 template="plotly_white", height=280,
                                 color_discrete_sequence=PALETTE)
            fig_era.update_yaxes(visible=False)
            st.plotly_chart(fig_era, use_container_width=True)
            for c_id in sorted(set(labels)):
                yrs = [years_used[i] for i, l in enumerate(labels) if l == c_id]
                st.write(f"**Era {c_id}:** {min(yrs)}–{max(yrs)}  ({', '.join(map(str, yrs))})")
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_sc)
            var_ex = pca.explained_variance_ratio_
            pca_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1],
                                   "Era": labels.astype(str), "Year": years_used})
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Era", text="Year",
                                 title=f"PCA — {var_ex[0]*100:.1f}% + {var_ex[1]*100:.1f}% variance",
                                 template="plotly_white", height=420,
                                 color_discrete_sequence=PALETTE)
            fig_pca.update_traces(textposition="top center", marker_size=9)
            st.plotly_chart(fig_pca, use_container_width=True)
            era_df2 = pd.DataFrame(X_sc, columns=sel_feats, index=years_used)
            era_df2["Era"] = labels
            st.write("**Era profiles (mean of scaled features):**")
            st.dataframe(era_df2.groupby("Era")[sel_feats].mean().round(3), use_container_width=True)
            sil = silhouette_score(X_sc, labels)
            db  = davies_bouldin_score(X_sc, labels)
            col1, col2 = st.columns(2)
            col1.metric("Silhouette", f"{sil:.3f}")
            col2.metric("Davies-Bouldin", f"{db:.3f}")
            st.session_state["bz_cluster_result"] = {
                "labels": labels, "years": years_used,
                "X_sc": X_sc, "feats": sel_feats, "k": int(k_use),
                "sil": sil, "db": db
            }
            st.success("✅ K-Means complete.")
        elif st.session_state.get("bz_cluster_result") is not None:
            st.info("K-Means already run. Re-run to update.")

    elif page == "🌿 BZ-4 · Hierarchical":
        st.title("🌿 BZ-4 · Hierarchical Clustering — Brazil Years")
        cr = st.session_state.get("bz_cluster_result")
        if cr is None:
            st.warning("⚠️ Run K-Means first (BZ-3).")
            st.stop()
        from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
        from scipy.spatial.distance import pdist
        from sklearn.metrics import adjusted_rand_score
        import matplotlib.pyplot as plt
        import io

        X_sc      = cr["X_sc"]
        years     = cr["years"]
        km_labels = cr["labels"]
        k         = cr["k"]
        linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
        cut_ratio      = st.slider("Cut height (% of max)", 40, 90, 70)

        if st.button("▶ Run Hierarchical Clustering", type="primary"):
            Z = linkage(X_sc, method=linkage_method)
            c_stat, _ = cophenet(Z, pdist(X_sc))
            st.metric("Cophenetic Correlation", f"{c_stat:.4f}")
            fig_d, ax_d = plt.subplots(figsize=(max(14, len(years)//2), 5))
            cut = (cut_ratio / 100) * max(Z[:, 2])
            dendrogram(Z, labels=[str(y) for y in years], leaf_rotation=60,
                       leaf_font_size=8, color_threshold=cut, ax=ax_d)
            ax_d.set_title(f"Brazil Hierarchical Clustering — Years ({linkage_method} linkage)", fontsize=12)
            ax_d.set_xlabel("Year"); ax_d.set_ylabel("Distance")
            ax_d.axhline(y=cut, color="red", linestyle="--", label=f"{cut_ratio}% cut")
            ax_d.legend(fontsize=8)
            plt.tight_layout()
            buf = io.BytesIO()
            fig_d.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig_d)
            hc_labels = fcluster(Z, t=k, criterion="maxclust") - 1
            for c_id in sorted(set(hc_labels)):
                yrs = [years[i] for i, l in enumerate(hc_labels) if l == c_id]
                st.write(f"**HC Cluster {c_id}:** {', '.join(map(str, yrs))}")
            ari = adjusted_rand_score(km_labels, hc_labels)
            col1, col2 = st.columns(2)
            col1.metric("ARI vs K-Means", f"{ari:.4f}")
            if ari > 0.7:
                col2.success("Strong agreement — era structure is stable.")
            elif ari > 0.4:
                col2.warning("Moderate agreement.")
            else:
                col2.error("Low agreement — methods disagree.")
            st.success("✅ Hierarchical clustering complete.")

    elif page == "⏱ BZ-5 · DTW":
        st.title("⏱ BZ-5 · DTW Time-Series Clustering — Brazil Indicators")
        df = st.session_state.get("bz_df")
        if df is None:
            st.warning("⚠️ Load data first (BZ-1).")
            st.stop()
        try:
            from tslearn.clustering import TimeSeriesKMeans
            from tslearn.utils import to_time_series_dataset
            from tslearn.metrics import cdist_dtw
            from tslearn.preprocessing import TimeSeriesScalerMinMax
            from sklearn.metrics import silhouette_score
        except ImportError:
            st.error("tslearn not installed.")
            st.stop()

        avail_cols = [c for c in df.columns if df[c].notna().sum() > len(df) * 0.4]
        sel_cols = st.multiselect("Select indicator time series to cluster",
                                  avail_cols, default=avail_cols[:min(12, len(avail_cols))])
        c1, c2 = st.columns(2)
        k_min_dtw = c1.number_input("Min k", value=2, min_value=2)
        k_max_dtw = c2.number_input("Max k", value=5, min_value=3)

        if st.button("▶ Run DTW on indicators", type="primary"):
            if len(sel_cols) < 3:
                st.error("Select at least 3 indicator columns.")
                st.stop()
            ts_list = [df[col].interpolate(limit_direction="both").values for col in sel_cols]
            X_ts = to_time_series_dataset(ts_list)
            scaler_ts = TimeSeriesScalerMinMax()
            X_ts_scaled = scaler_ts.fit_transform(X_ts)
            with st.spinner("Computing DTW distance matrix…"):
                dtw_dist = cdist_dtw(X_ts_scaled)
            sil_scores = {}
            DTW_K_RANGE = range(int(k_min_dtw), min(int(k_max_dtw) + 1, len(sel_cols) - 1))
            prog = st.progress(0)
            for i, k_d in enumerate(DTW_K_RANGE):
                km = TimeSeriesKMeans(n_clusters=k_d, metric="dtw", max_iter=10, random_state=42, n_jobs=-1)
                lbl = km.fit_predict(X_ts_scaled)
                sil_scores[k_d] = silhouette_score(dtw_dist, lbl, metric="precomputed")
                prog.progress((i + 1) / len(DTW_K_RANGE))
            fig_sil = px.line(x=list(DTW_K_RANGE), y=list(sil_scores.values()), markers=True,
                              labels={"x": "k", "y": "Silhouette"},
                              title="DTW Silhouette vs k", template="plotly_white")
            st.plotly_chart(fig_sil, use_container_width=True)
            best_k = max(sil_scores, key=sil_scores.get)
            st.info(f"Best k = **{best_k}** (Silhouette = {sil_scores[best_k]:.4f})")
            k_use = st.number_input("Use k", value=best_k, min_value=2, max_value=int(k_max_dtw))
            km_final = TimeSeriesKMeans(n_clusters=int(k_use), metric="dtw", max_iter=20, random_state=42, n_jobs=-1)
            final_labels = km_final.fit_predict(X_ts_scaled)
            years_ax = df.index.tolist()
            fig_ts = make_subplots(1, int(k_use),
                                   subplot_titles=[f"Cluster {c}" for c in range(int(k_use))],
                                   shared_yaxes=True)
            for c_id in range(int(k_use)):
                members = [sel_cols[i] for i, l in enumerate(final_labels) if l == c_id]
                centroid = km_final.cluster_centers_[c_id].flatten()
                short_names = [m.replace("WB | ","WB:").replace("UNESCO | ","UN:") for m in members]
                for short, member in zip(short_names, members):
                    idx = sel_cols.index(member)
                    fig_ts.add_trace(go.Scatter(
                        x=years_ax, y=X_ts_scaled[idx].flatten(), mode="lines",
                        name=short, line=dict(color=PALETTE[c_id % len(PALETTE)], width=1.5),
                        opacity=0.6, legendgroup=f"c{c_id}"
                    ), row=1, col=c_id + 1)
                fig_ts.add_trace(go.Scatter(
                    x=years_ax, y=centroid, mode="lines",
                    line=dict(color="white", width=2.5, dash="dash"),
                    name=f"Centroid {c_id}", legendgroup=f"c{c_id}"
                ), row=1, col=c_id + 1)
                st.write(f"**Cluster {c_id}:** {', '.join(short_names)}")
            fig_ts.update_layout(height=420, template="plotly_white", title="DTW Clusters — Brazil Indicators")
            st.plotly_chart(fig_ts, use_container_width=True)
            st.success("✅ DTW clustering complete.")

        # ── BZ-6 — UNIFIED FORECASTING ────────────────────────────────────────────
    elif page == "🤖 BZ-6 · Supervised Models":
        st.title("🤖 BZ-6 · Unified Forecasting — Brazil")
        st.markdown("""
        > **Method:** Two evaluation modes:  
        > 1. **Expanding window** – 1‑step ahead over all years (backtest).  
        > 2. **Hold‑out forecast** – leave out last N years, train once, forecast N years ahead, compare to actual.  
        > **Models:** ML (Ridge / RF / GBM) + ARIMA (auto) + Holt‑Winters
        """)
        df = st.session_state.get("bz_df")
        if df is None:
            st.warning("⚠️ Load data first (BZ-1).")
            st.stop()

        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np
        import pandas as pd

        # Colour palette (fallback if not defined globally)
        if 'PALETTE' not in dir():
            PALETTE = px.colors.qualitative.Set2

        target_opts = [c for c in BZ_COMPLETION_COLS if c in df.columns]
        sel_targets_bz = st.multiselect(
            "Targets to forecast",
            target_opts,
            default=target_opts,
            format_func=lambda c: c.replace("UNESCO | Completion rate for ","").replace(" (%)","")
        )

        all_feat_candidates = [c for c in df.columns
                            if c not in target_opts and df[c].notna().sum() > len(df) * 0.4]
        default_x = [c for c in BZ_ECON_COLS if c in all_feat_candidates]
        PRED_FEATS_BZ = st.multiselect("Predictor features (exogenous)", all_feat_candidates, default=default_x)

        c1, c2 = st.columns(2)
        min_train_bz = c1.number_input("Min training years", value=5, min_value=3)

        # ===== Mode selection =====
        forecast_mode = st.radio(
            "Evaluation mode",
            ["Expanding window CV (backtest on all years)", "Hold‑out forecast (leave out last N years)"],
            index=0,
            help="Expanding window: test each year one‑by‑one. Hold‑out: train once, forecast last N years."
        )

        if forecast_mode == "Hold‑out forecast (leave out last N years)":
            holdout_years = st.number_input("Number of years to leave out for testing", min_value=1, max_value=10, value=3, step=1)
            st.info(f"Will train on all data except the last {holdout_years} years, then forecast those {holdout_years} years.")
            # For hold‑out, only univariate models are used (no future exogenous features)
            available_models = ["ARIMA", "HoltWinters"]
            models_bz = st.multiselect(
                "Models (univariate only for hold‑out)",
                available_models,
                default=available_models,
                help="ML models require future exogenous values not available here."
            )
        else:
            models_bz = st.multiselect(
                "Models",
                ["ML_Ridge", "ML_RandomForest", "ML_GradientBoosting", "ARIMA", "HoltWinters"],
                default=["ML_Ridge", "ML_RandomForest", "ML_GradientBoosting", "ARIMA", "HoltWinters"]
            )

        if st.button("▶ Run Unified Forecasting", type="primary"):
            if not sel_targets_bz:
                st.error("Select at least 1 target.")
                st.stop()

            df_sorted = df.sort_index()

            # ------------------------------------------------------------------
            #  HOLD‑OUT FORECAST MODE
            # ------------------------------------------------------------------
            if forecast_mode == "Hold‑out forecast (leave out last N years)":
                if len(df_sorted) <= holdout_years:
                    st.error(f"Not enough data: need more than {holdout_years} years.")
                    st.stop()

                train_df = df_sorted.iloc[:-holdout_years]
                test_df = df_sorted.iloc[-holdout_years:]
                test_years = test_df.index.tolist()

                results_holdout = []

                for target_col in sel_targets_bz:
                    ts_train = train_df[target_col].dropna()
                    ts_test = test_df[target_col]

                    # ARIMA
                    if "ARIMA" in models_bz:
                        try:
                            from pmdarima import auto_arima
                            am = auto_arima(ts_train, seasonal=False, stepwise=True,
                                            suppress_warnings=True, error_action='ignore',
                                            max_p=2, max_q=2, max_d=1)
                            forecast = am.predict(n_periods=holdout_years)
                            for year, pred, true in zip(test_years, forecast, ts_test):
                                if not pd.isna(true):
                                    results_holdout.append({
                                        "target": target_col,
                                        "model": "ARIMA",
                                        "test_year": year,
                                        "true": true,
                                        "pred": pred
                                    })
                        except Exception as e:
                            st.warning(f"ARIMA failed for {target_col}: {e}")

                    # Holt‑Winters
                    if "HoltWinters" in models_bz:
                        try:
                            hw = ExponentialSmoothing(ts_train, trend='add', seasonal=None).fit()
                            forecast = hw.forecast(steps=holdout_years)
                            for year, pred, true in zip(test_years, forecast, ts_test):
                                if not pd.isna(true):
                                    results_holdout.append({
                                        "target": target_col,
                                        "model": "HoltWinters",
                                        "test_year": year,
                                        "true": true,
                                        "pred": pred
                                    })
                        except Exception as e:
                            st.warning(f"Holt‑Winters failed for {target_col}: {e}")

                if not results_holdout:
                    st.error("No valid hold‑out forecasts generated.")
                    st.stop()

                full_bz = pd.DataFrame(results_holdout)
                full_bz["target_short"] = full_bz["target"].str.replace(
                    "UNESCO | Completion rate for ", "", regex=False).str.replace(" (%)", "", regex=False)

                # Metrics on hold-out period
                st.subheader(f"📊 Hold‑out forecast error (last {holdout_years} years)")
                metrics_hold = []
                for (target, model), group in full_bz.groupby(["target_short", "model"]):
                    rmse = np.sqrt(mean_squared_error(group["true"], group["pred"]))
                    mae = mean_absolute_error(group["true"], group["pred"])
                    metrics_hold.append({"target": target, "model": model, "RMSE": rmse, "MAE": mae})
                df_metrics = pd.DataFrame(metrics_hold)
                st.dataframe(df_metrics.round(4), use_container_width=True)

                # Plot for each target
                for target_col in sel_targets_bz:
                    label = target_col.replace("UNESCO | Completion rate for ", "").replace(" (%)", "")
                    sub = full_bz[full_bz["target"] == target_col].sort_values("test_year")
                    if sub.empty:
                        continue
                    fig = go.Figure()
                    # Full historical series (train + test)
                    full_series = df_sorted[target_col].dropna()
                    fig.add_trace(go.Scatter(
                        x=full_series.index, y=full_series.values,
                        mode="lines", name="Historical", line=dict(color="gray", width=1)))
                    # Actual hold-out points
                    fig.add_trace(go.Scatter(
                        x=sub["test_year"], y=sub["true"],
                        mode="markers", name="Actual (hold‑out)", marker=dict(color="blue", size=8)))
                    # Forecasts per model
                    for model in sub["model"].unique():
                        model_sub = sub[sub["model"] == model]
                        fig.add_trace(go.Scatter(
                            x=model_sub["test_year"], y=model_sub["pred"],
                            mode="lines+markers", name=f"Forecast ({model})",
                            line=dict(dash="dot")))
                    fig.update_layout(
                        title=f"{label} – Hold‑out forecast vs actual",
                        xaxis_title="Year", yaxis_title="Completion rate (%)",
                        template="plotly_white", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.success("Hold‑out forecast complete.")
                st.session_state["bz_holdout"] = full_bz

            # ------------------------------------------------------------------
            #  ORIGINAL EXPANDING WINDOW CV MODE
            # ------------------------------------------------------------------
            else:
                def run_forecasting_bz(target_col, feat_cols, df_ts, min_tr, model_names):
                    results = []
                    MIN_TRAIN = int(min_tr)
                    ts = df_ts[target_col].dropna()

                    for i in range(MIN_TRAIN, len(ts)):
                        train_years = ts.index[:i]
                        test_year   = ts.index[i]
                        y_true = ts.iloc[i]
                        if pd.isna(y_true):
                            continue

                        train_df = df_ts.loc[train_years]
                        test_row  = df_ts.loc[[test_year]]

                        # ML models
                        feats_avail = [f for f in feat_cols if f in df_ts.columns]
                        train_ml = train_df[feats_avail + [target_col]].dropna()
                        if len(train_ml) >= MIN_TRAIN:
                            test_ml = test_row[feats_avail].dropna()
                            if len(test_ml) > 0:
                                scaler = StandardScaler()
                                X_tr = scaler.fit_transform(train_ml[feats_avail])
                                X_te = scaler.transform(test_ml[feats_avail])
                                y_tr = train_ml[target_col]
                                ml_map = {}
                                if "ML_Ridge" in model_names:
                                    ml_map["ML_Ridge"] = Ridge(alpha=1.0)
                                if "ML_RandomForest" in model_names:
                                    ml_map["ML_RandomForest"] = RandomForestRegressor(
                                        n_estimators=100, max_depth=3, random_state=42)
                                if "ML_GradientBoosting" in model_names:
                                    ml_map["ML_GradientBoosting"] = GradientBoostingRegressor(
                                        n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
                                for mname, model in ml_map.items():
                                    try:
                                        model.fit(X_tr, y_tr)
                                        results.append({
                                            "test_year": test_year, "target": target_col,
                                            "model": mname, "true": y_true,
                                            "pred": model.predict(X_te)[0]
                                        })
                                    except:
                                        pass

                        # ARIMA + Holt-Winters on raw target series
                        ts_train = ts.iloc[:i]
                        if len(ts_train) >= 3:
                            if "ARIMA" in model_names:
                                try:
                                    from pmdarima import auto_arima
                                    am = auto_arima(ts_train, seasonal=False, stepwise=True,
                                                    suppress_warnings=True, error_action='ignore',
                                                    max_p=2, max_q=2, max_d=1)
                                    results.append({
                                        "test_year": test_year, "target": target_col,
                                        "model": "ARIMA", "true": y_true,
                                        "pred": am.predict(n_periods=1)[0]
                                    })
                                except:
                                    pass
                            if "HoltWinters" in model_names:
                                try:
                                    hw = ExponentialSmoothing(ts_train, trend='add', seasonal=None).fit()
                                    results.append({
                                        "test_year": test_year, "target": target_col,
                                        "model": "HoltWinters", "true": y_true,
                                        "pred": hw.forecast(1).iloc[-1]
                                    })
                                except:
                                    pass
                    return pd.DataFrame(results)

                all_results_bz = []
                prog = st.progress(0)
                for idx, target_col in enumerate(sel_targets_bz):
                    label = target_col.replace("UNESCO | Completion rate for ","").replace(" (%)","")
                    st.write(f"Running **{label}**…")
                    res = run_forecasting_bz(target_col, PRED_FEATS_BZ, df_sorted,
                                            min_train_bz, models_bz)
                    if len(res) == 0:
                        st.warning(f"No valid forecasts for {label}.")
                    else:
                        all_results_bz.append(res)
                    prog.progress((idx + 1) / len(sel_targets_bz))

                if not all_results_bz:
                    st.error("No results generated. Try reducing Min training years.")
                    st.stop()

                full_bz = pd.concat(all_results_bz, ignore_index=True)
                full_bz["target_short"] = full_bz["target"].str.replace(
                    "UNESCO | Completion rate for ","", regex=False).str.replace(" (%)","", regex=False)

                st.subheader("📊 Results — RMSE")
                pivot_rmse_bz = full_bz.groupby(["target_short","model"]).apply(
                    lambda g: pd.Series({"RMSE": np.sqrt(mean_squared_error(g["true"], g["pred"]))})
                ).reset_index().pivot(index="target_short", columns="model", values="RMSE")
                st.dataframe(pivot_rmse_bz.round(4), use_container_width=True)

                st.subheader("📊 Results — MAE")
                pivot_mae_bz = full_bz.groupby(["target_short","model"]).apply(
                    lambda g: pd.Series({"MAE": mean_absolute_error(g["true"], g["pred"])})
                ).reset_index().pivot(index="target_short", columns="model", values="MAE")
                st.dataframe(pivot_mae_bz.round(4), use_container_width=True)

                st.subheader("📊 Results — R²")
                pivot_r2_bz = full_bz.groupby(["target_short","model"]).apply(
                    lambda g: pd.Series({"R2": r2_score(g["true"], g["pred"])})
                ).reset_index().pivot(index="target_short", columns="model", values="R2")
                st.dataframe(pivot_r2_bz.round(4), use_container_width=True)

                comparison_bz = full_bz.groupby(["target_short","model"]).apply(
                    lambda g: pd.Series({
                        "RMSE": np.sqrt(mean_squared_error(g["true"], g["pred"])),
                        "MAE":  mean_absolute_error(g["true"], g["pred"]),
                        "R2":   r2_score(g["true"], g["pred"])
                    })
                ).reset_index()

                st.subheader("🏆 Best model per target")
                col1, col2 = st.columns(2)
                best_r2_bz  = comparison_bz.loc[comparison_bz.groupby("target_short")["R2"].idxmax(),
                                                ["target_short","model","R2"]]
                best_mae_bz = comparison_bz.loc[comparison_bz.groupby("target_short")["MAE"].idxmin(),
                                                ["target_short","model","MAE"]]
                col1.write("**By highest R²:**")
                col1.dataframe(best_r2_bz.round(4), use_container_width=True)
                col2.write("**By lowest MAE:**")
                col2.dataframe(best_mae_bz.round(4), use_container_width=True)

                # Actual vs predicted line chart for best model
                st.subheader("📈 Predicted vs Actual over time")
                for target_col in sel_targets_bz:
                    label = target_col.replace("UNESCO | Completion rate for ","").replace(" (%)","")
                    sub_t = full_bz[full_bz["target"] == target_col]
                    if sub_t.empty:
                        continue
                    best_m = sub_t.groupby("model").apply(
                        lambda g: r2_score(g["true"], g["pred"])
                    ).idxmax()
                    sub_best = sub_t[sub_t["model"] == best_m].sort_values("test_year")
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(
                        x=sub_best["test_year"], y=sub_best["true"],
                        mode="lines+markers", name="Actual", line_color=PALETTE[0]))
                    fig_line.add_trace(go.Scatter(
                        x=sub_best["test_year"], y=sub_best["pred"],
                        mode="lines+markers", name=f"Predicted ({best_m})",
                        line=dict(color=PALETTE[2], dash="dash")))
                    fig_line.update_layout(
                        title=f"{label} — Actual vs Predicted ({best_m})",
                        template="plotly_white", height=350,
                        xaxis_title="Year", yaxis_title="Completion Rate"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

                # Bar comparison
                fig_bar_bz = px.bar(
                    comparison_bz.melt(id_vars=["target_short","model"], value_vars=["RMSE","MAE"]),
                    x="model", y="value", color="target_short", barmode="group",
                    facet_col="variable", template="plotly_white", height=420,
                    title="Model Comparison — RMSE & MAE by Target",
                    color_discrete_sequence=PALETTE
                )
                fig_bar_bz.update_xaxes(tickangle=30)
                st.plotly_chart(fig_bar_bz, use_container_width=True)

                st.session_state["bz_cv_results"] = {
                    "full_results": full_bz,
                    "comparison": comparison_bz,
                    "pivot_rmse": pivot_rmse_bz,
                    "pivot_mae": pivot_mae_bz,
                    "pivot_r2": pivot_r2_bz,
                }
                st.success("✅ Unified forecasting complete.")

        elif st.session_state.get("bz_cv_results") is not None:
            st.info("Forecasting already run. Re-run to update.")
            cv = st.session_state["bz_cv_results"]
            if isinstance(cv, dict) and "pivot_r2" in cv:
                st.subheader("R² (cached)")
                st.dataframe(cv["pivot_r2"].round(4), use_container_width=True)

    elif page == "📋 BZ-7 · Summary":
        st.title("📋 BZ-7 · Summary — Brazil")
        df     = st.session_state.get("bz_df")
        cr     = st.session_state.get("bz_cluster_result")
        cv_res = st.session_state.get("bz_cv_results")
        if df is None:
            st.warning("Run all modules first.")
            st.stop()

        st.subheader("📐 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Years covered", f"{df.index.min()}–{df.index.max()}")
        c2.metric("Observations (years)", len(df))
        c3.metric("Indicator columns", df.shape[1])

        if cr is not None:
            st.markdown("---")
            st.subheader("🔵 Developmental Eras (K-Means)")
            col1, col2, col3 = st.columns(3)
            col1.metric("k (eras)", cr["k"])
            col2.metric("Silhouette", f"{cr['sil']:.3f}")
            col3.metric("Davies-Bouldin", f"{cr['db']:.3f}")
            for c_id in sorted(set(cr["labels"])):
                yrs = [cr["years"][i] for i, l in enumerate(cr["labels"]) if l == c_id]
                st.write(f"  Era {c_id}: {min(yrs)}–{max(yrs)} → {', '.join(map(str, yrs))}")

        if cv_res is not None and isinstance(cv_res, dict) and "comparison" in cv_res:
            st.markdown("---")
            st.subheader("🤖 Forecasting Evaluation")
            comp = cv_res["comparison"]
            st.dataframe(comp.round(4), use_container_width=True)
            best_r2 = comp.loc[comp.groupby("target_short")["R2"].idxmax(),
                               ["target_short","model","R2","MAE","RMSE"]]
            st.success("Best models per target (highest R²):")
            st.dataframe(best_r2.round(4), use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Policy Interpretation Notes")
        st.text_area("Add your interpretation here", height=240, placeholder=
"""Example:
Developmental Eras:
• Era 0 (1970–1989): high poverty, low GDP per capita, low completion — pre-reform period
• Era 1 (1990–2004): declining poverty + Bolsa Família era, rising primary completion
• Era 2 (2005–2025): GDP growth + Gini decline → near-universal primary, upper secondary gap remains

Key correlations:
• GDP per capita: r = ___  with upper secondary completion
• Poverty ($2.15): r = ___ — strongest negative predictor
• Gini index: r = ___ — inequality persists even as mean income rises
""")