"""
Master Causal Machine Learning Pipeline: Hormone Therapy in Breast Cancer
Dataset: METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)
Method: Double Machine Learning via Causal Forest (EconML CausalForestDML)

Description:
  This production-grade script encapsulates the complete causal machine learning
  workflow:
    1. Loads clinical data and isolates the METABRIC cohort.
    2. Derives Tumor Stage from the Nottingham Prognostic Index (NPI).
    3. Selects target causal variables and creates the clean analytical cohort.
    4. Appropriately encodes Effect Modifiers (X) and Confounders (W).
    5. Trains CausalForestDML with Random Forest nuisance estimators.
    6. Computes patient-level Conditional Average Treatment Effects (CATE).
    7. Validates Overall ATE and subgroup effect heterogeneity.
    8. Appends 'Individual_CATE_Months' and persists results to 'metabric_causal_results.csv'.

Author: Causal AI Machine Learning Engineer
Date: 2026-10-06
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Suppress deprecation and algorithmic convergence warnings for clean console output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure UTF-8 output encoding for terminal on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        pass

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML


def npi_to_stage(npi: float) -> str:
    """
    Clinically maps Nottingham Prognostic Index (NPI) to Tumor Stage equivalents:
      - NPI <= 3.4       -> Stage_I   (Good prognosis)
      - 3.4 < NPI <= 5.4  -> Stage_II  (Moderate prognosis)
      - NPI > 5.4        -> Stage_III (Poor prognosis)
    """
    if pd.isna(npi):
        return np.nan
    if npi <= 3.4:
        return "Stage_I"
    elif npi <= 5.4:
        return "Stage_II"
    else:
        return "Stage_III"


def run_master_causal_pipeline(
    input_csv: str = "brca_clinical_data.csv",
    output_csv: str = "metabric_causal_results.csv",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Executes the end-to-end Master Causal Pipeline and saves results to disk.
    """
    print("=" * 85)
    print("      MASTER CAUSAL PIPELINE: HORMONE THERAPY IN BREAST CANCER (METABRIC)")
    print("      DOUBLE MACHINE LEARNING & CAUSAL FOREST HETEROGENEITY ANALYSIS")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # Step 1: Load clinical data, filter METABRIC, and derive Tumor Stage
    # -------------------------------------------------------------------------
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input clinical dataset not found at '{input_csv}'")

    print(f"\n[Step 1] Loading clinical data from '{input_csv}'...")
    raw_df = pd.read_csv(input_csv, low_memory=False)
    print(f"         - Raw dataset shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

    metabric_df = raw_df[raw_df["studyId"] == "brca_metabric"].copy()
    print(f"         - Filtered cohort (studyId == 'brca_metabric'): N = {len(metabric_df):,} patients")

    # Clinically derive Tumor Stage from NPI (as AJCC stage is unpopulated in METABRIC)
    print("         - Mapping Nottingham Prognostic Index (NPI) -> derived 'TUMOR_STAGE'...")
    metabric_df["TUMOR_STAGE"] = metabric_df["NPI"].apply(npi_to_stage)
    stage_dist = metabric_df["TUMOR_STAGE"].value_counts(dropna=False).to_dict()
    print(f"           Stage distribution: {stage_dist}")

    # -------------------------------------------------------------------------
    # Step 2: Select target columns & drop NaNs to construct the clean cohort
    # -------------------------------------------------------------------------
    target_columns = [
        "HORMONE_THERAPY",
        "OS_MONTHS",
        "AGE_AT_DIAGNOSIS",
        "ER_IHC",
        "TUMOR_STAGE",
        "CLAUDIN_SUBTYPE",
        "INFERRED_MENOPAUSAL_STATE",
    ]
    print(f"\n[Step 2] Selecting target columns: {target_columns}")
    clean_df = metabric_df[target_columns].dropna().copy()
    print(f"         - Complete-case clean analytical cohort: N = {len(clean_df):,} patients")

    # Binary transformations
    # Treatment: HORMONE_THERAPY (YES -> 1, NO -> 0)
    # Confounder: ER_IHC (Positive -> 1, Negative -> 0)
    clean_df["HORMONE_THERAPY"] = clean_df["HORMONE_THERAPY"].map({"YES": 1, "NO": 0}).astype(int)
    clean_df["ER_IHC"] = clean_df["ER_IHC"].map({"Positive": 1, "Positve": 1, "Negative": 0}).astype(int)

    # -------------------------------------------------------------------------
    # Step 3: One-hot encode Effect Modifiers (X) and Confounders (W)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Encoding features for Causal Double Machine Learning:")

    # Target outcome (Y) and treatment (T)
    Y = clean_df["OS_MONTHS"].values.astype(float)
    T = clean_df["HORMONE_THERAPY"].values.astype(int)

    # Effect Modifiers (X): CLAUDIN_SUBTYPE and INFERRED_MENOPAUSAL_STATE
    # All categories retained (drop_first=False) for optimal causal tree partitioning
    effect_modifier_cols = ["CLAUDIN_SUBTYPE", "INFERRED_MENOPAUSAL_STATE"]
    X_df = pd.get_dummies(clean_df[effect_modifier_cols], drop_first=False, dtype=float)
    X = X_df.values
    x_features = list(X_df.columns)
    print(f"         - Effect Modifiers (X) [{len(x_features)} features]: {x_features}")

    # Confounders (W): Age, ER_IHC, and Tumor Stage
    # Tumor stage one-hot encoded with drop_first=True to avoid collinearity
    W_stage_df = pd.get_dummies(clean_df[["TUMOR_STAGE"]], drop_first=True, dtype=float)
    W_num_df = clean_df[["AGE_AT_DIAGNOSIS", "ER_IHC"]].reset_index(drop=True)
    W_df = pd.concat([W_num_df, W_stage_df.reset_index(drop=True)], axis=1)
    W = W_df.values
    w_features = list(W_df.columns)
    print(f"         - Confounders (W) [{len(w_features)} features]: {w_features}")

    # -------------------------------------------------------------------------
    # Step 4: Initialize and train CausalForestDML
    # -------------------------------------------------------------------------
    print("\n[Step 4] Training CausalForestDML (EconML) with Random Forest nuisance estimators...")
    print("         - Model Y: RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)")
    print("         - Model T: RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)")
    print("         - Random State: 42 (Exact Reproducibility)")

    causal_forest = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=random_state),
        model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=random_state),
        discrete_treatment=True,
        random_state=random_state,
    )

    causal_forest.fit(Y, T, X=X, W=W)
    print("         [OK] CausalForestDML fitting complete!")

    # -------------------------------------------------------------------------
    # Step 5: Calculate patient-level CATE
    # -------------------------------------------------------------------------
    print("\n[Step 5] Calculating patient-level CATE (Conditional Average Treatment Effect)...")
    cate_values = causal_forest.effect(X)

    # -------------------------------------------------------------------------
    # Step 6: Print Overall ATE and Subgroup Analysis
    # -------------------------------------------------------------------------
    overall_ate = float(causal_forest.ate(X))
    ci_lower, ci_upper = causal_forest.ate_interval(X)

    print("\n" + "=" * 85)
    print("                             CAUSAL ESTIMATION RESULTS")
    print("=" * 85)
    print(f"Overall Population ATE (Causal Forest) : {overall_ate:+.2f} months")
    print(f"95% Confidence Interval                 : [{ci_lower:+.2f}, {ci_upper:+.2f}] months")
    print("-" * 85)

    # Temporary evaluation series for subgroup summaries
    clean_df["temp_cate"] = cate_values

    print("\n" + "=" * 85)
    print("SUBGROUP ANALYSIS BY MOLECULAR SUBTYPE (CLAUDIN_SUBTYPE)")
    print("=" * 85)
    subtype_summary = (
        clean_df.groupby("CLAUDIN_SUBTYPE")["temp_cate"]
        .agg(
            Patient_Count="count",
            Mean_CATE="mean",
            Median_CATE="median",
            Std_CATE="std",
        )
        .sort_values(by="Mean_CATE", ascending=False)
    )
    print(f"{'Molecular Subtype':<18} | {'N':>6} | {'Mean CATE (mo)':>16} | {'Median CATE (mo)':>18} | {'Std (mo)':>10}")
    print("-" * 78)
    for subtype, row in subtype_summary.iterrows():
        print(f"{subtype:<18} | {int(row['Patient_Count']):>6} | {row['Mean_CATE']:>+16.2f} | {row['Median_CATE']:>+18.2f} | {row['Std_CATE']:>10.2f}")

    print("\n" + "=" * 85)
    print("SUBGROUP ANALYSIS BY SUBTYPE & INFERRED_MENOPAUSAL_STATE")
    print("=" * 85)
    interaction_summary = (
        clean_df.groupby(["CLAUDIN_SUBTYPE", "INFERRED_MENOPAUSAL_STATE"])["temp_cate"]
        .agg(
            Patient_Count="count",
            Mean_CATE="mean",
            Min_CATE="min",
            Max_CATE="max",
        )
        .sort_values(by="Mean_CATE", ascending=False)
    )
    print(f"{'Molecular Subtype':<15} | {'Menopausal Status':<18} | {'N':>5} | {'Mean CATE (mo)':>16} | {'Range [Min, Max] (mo)':>23}")
    print("-" * 85)
    for (subtype, status), row in interaction_summary.iterrows():
        range_str = f"[{row['Min_CATE']:+.1f}, {row['Max_CATE']:+.1f}]"
        print(f"{subtype:<15} | {status:<18} | {int(row['Patient_Count']):>5} | {row['Mean_CATE']:>+16.2f} | {range_str:>23}")

    best_subgroup = interaction_summary.iloc[0]
    best_name = interaction_summary.index[0]
    print("\n" + "-" * 85)
    print(f"Top Responding Subgroup : {best_name[0]} / {best_name[1]}-menopausal (N = {int(best_subgroup['Patient_Count'])})")
    print(f"Mean Survival Gain      : {best_subgroup['Mean_CATE']:+.2f} months")
    print("-" * 85)

    # -------------------------------------------------------------------------
    # Step 7: Append Individual_CATE_Months and save to CSV
    # -------------------------------------------------------------------------
    print(f"\n[Step 7] Appending 'Individual_CATE_Months' and persisting to '{output_csv}'...")
    clean_df.drop(columns=["temp_cate"], inplace=True)
    clean_df["Individual_CATE_Months"] = cate_values

    # Save to disk without index
    clean_df.to_csv(output_csv, index=False)
    print(f"         [OK] File saved successfully!")
    print(f"         - Destination: {os.path.abspath(output_csv)}")
    print(f"         - Shape: {clean_df.shape[0]:,} rows x {clean_df.shape[1]} columns")
    print(f"         - Final Columns: {list(clean_df.columns)}")

    # Sanity checks
    assert "Individual_CATE_Months" in clean_df.columns, "CATE column missing!"
    assert len(clean_df) == 1937, f"Expected 1937 rows, got {len(clean_df)}"
    assert not clean_df["Individual_CATE_Months"].isna().any(), "NaN values found in CATE!"

    print("\n" + "=" * 85)
    print("MASTER CAUSAL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 85)

    return clean_df


if __name__ == "__main__":
    run_master_causal_pipeline(
        input_csv="brca_clinical_data.csv",
        output_csv="metabric_causal_results.csv",
        random_state=42,
    )
