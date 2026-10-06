"""
Causal Meta-Analysis of Hormone Therapy in Breast Cancer
Step 2: Causal Effect Estimation on METABRIC Cohort using DoWhy

Author: Causal AI Machine Learning Engineer
Date: 2026-10-06
"""

import os
import sys
import warnings

# Suppress library deprecation and future warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure UTF-8 output encoding for terminal on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        pass

# Ensure compatibility between DoWhy and NetworkX 3.x+
import networkx as nx
import networkx.algorithms
if not hasattr(nx.algorithms, "d_separated"):
    nx.algorithms.d_separated = nx.is_d_separator

import pandas as pd
from dowhy import CausalModel


def run_causal_estimation(data_path: str = "brca_clinical_data.csv"):
    """
    Executes the causal inference pipeline on the METABRIC cohort:
      1. Loads data and filters to METABRIC cohort
      2. Selects necessary columns and cleans missing values
      3. Encodes variables and renames columns
      4. Builds Causal DAG with DoWhy
      5. Identifies backdoor causal estimand
      6. Estimates ATE using Linear Regression
      7. Reports exact survival months causally attributed to Hormone Therapy
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("=" * 80)
    print("CAUSAL META-ANALYSIS: HORMONE THERAPY IN BREAST CANCER")
    print("STEP 2: CAUSAL EFFECT ESTIMATION (METABRIC COHORT)")
    print("=" * 80)

    # ---------------------------------------------------------
    # Step 1: Load and filter dataset
    # ---------------------------------------------------------
    print(f"\n[1] Loading dataset from '{data_path}'...")
    raw_df = pd.read_csv(data_path, low_memory=False)
    print(f"    - Full dataset dimensions: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

    metabric_df = raw_df[raw_df["studyId"] == "brca_metabric"].copy()
    print(f"    - Filtered to 'brca_metabric': {metabric_df.shape[0]} rows")

    # ---------------------------------------------------------
    # Step 2: Select necessary columns
    # ---------------------------------------------------------
    target_columns = ["HORMONE_THERAPY", "OS_MONTHS", "AGE_AT_DIAGNOSIS", "ER_IHC"]
    print(f"\n[2] Selecting target columns: {target_columns}")
    subset_df = metabric_df[target_columns].copy()

    # ---------------------------------------------------------
    # Step 3: Drop missing values (NaN)
    # ---------------------------------------------------------
    missing_before = subset_df.isna().sum().to_dict()
    print(f"    - Missing values before filtering: {missing_before}")
    clean_df = subset_df.dropna().copy()
    dropped_count = len(subset_df) - len(clean_df)
    print(f"[3] Dropped {dropped_count} rows with missing values.")
    print(f"    - Clean analytical cohort size: {len(clean_df)} patients")

    # ---------------------------------------------------------
    # Step 4: Encode categorical variables and rename columns
    # ---------------------------------------------------------
    print("\n[4] Encoding variables and renaming features:")
    # Map Treatment: YES -> 1, NO -> 0
    clean_df["HORMONE_THERAPY"] = clean_df["HORMONE_THERAPY"].map({"YES": 1, "NO": 0}).astype(int)
    print(f"    - 'HORMONE_THERAPY' mapped (YES -> 1, NO -> 0):")
    print(f"      Distribution: {clean_df['HORMONE_THERAPY'].value_counts().to_dict()}")

    # Map ER_IHC: Positive / Positve -> 1, Negative -> 0
    clean_df["ER_IHC"] = clean_df["ER_IHC"].map({"Positive": 1, "Positve": 1, "Negative": 0}).astype(int)
    print(f"    - 'ER_IHC' mapped (Positive/Positve -> 1, Negative -> 0):")
    print(f"      Distribution: {clean_df['ER_IHC'].value_counts().to_dict()}")

    # Rename columns
    clean_df = clean_df.rename(
        columns={
            "AGE_AT_DIAGNOSIS": "Age",
            "OS_MONTHS": "Survival_Months",
        }
    )
    print("    - Renamed 'AGE_AT_DIAGNOSIS' -> 'Age'")
    print("    - Renamed 'OS_MONTHS' -> 'Survival_Months'")

    print("\n    Analytical Dataset Preview (First 5 Rows):")
    print(clean_df.head(5).to_string(index=False))

    # ---------------------------------------------------------
    # Step 5 & 6: Initialize DoWhy CausalModel (DAG Specification)
    # ---------------------------------------------------------
    treatment_var = "HORMONE_THERAPY"
    outcome_var = "Survival_Months"
    common_causes = ["Age", "ER_IHC"]

    print("\n" + "=" * 80)
    print("[5 & 6] Initializing Causal Directed Acyclic Graph (DAG) with DoWhy:")
    print(f"    - Treatment (T)     : '{treatment_var}'")
    print(f"    - Outcome (Y)       : '{outcome_var}'")
    print(f"    - Confounders (W)   : {common_causes}")
    print("=" * 80)

    model = CausalModel(
        data=clean_df,
        treatment=treatment_var,
        outcome=outcome_var,
        common_causes=common_causes,
    )

    # ---------------------------------------------------------
    # Step 7: Identify the causal effect
    # ---------------------------------------------------------
    print("\n[7] Identifying Causal Effect...")
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("\n" + "-" * 50)
    print("IDENTIFIED ESTIMAND SUMMARY:")
    print("-" * 50)
    print(identified_estimand)

    # ---------------------------------------------------------
    # Step 8: Estimate Average Treatment Effect (ATE)
    # ---------------------------------------------------------
    print("\n[8] Estimating Causal Effect via Linear Regression (backdoor.linear_regression)...")
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=None,
    )

    ate_value = float(estimate.value)

    # ---------------------------------------------------------
    # Step 9: Print clean, readable summary of Estimated Causal Effect
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL ESTIMATION RESULTS: AVERAGE TREATMENT EFFECT (ATE)")
    print("=" * 80)
    print(f"Treatment Variable         : {treatment_var} (1 = Yes, 0 = No)")
    print(f"Outcome Variable           : {outcome_var} (Months)")
    print(f"Adjustment Set (Backdoor)  : {', '.join(common_causes)}")
    print(f"Analytical Cohort Sample (N): {len(clean_df)}")
    print("-" * 80)
    print(f"Estimated Causal Effect (ATE): {ate_value:+.4f} months")
    print(f"Rounded Causal Effect (ATE)  : {ate_value:+.2f} months")
    print("-" * 80)

    print("\nClinical & Causal Interpretation:")
    if ate_value > 0:
        print(f"  Hormone therapy is causally estimated to increase overall survival by")
        print(f"  {abs(ate_value):.2f} months compared to no hormone therapy, after adjusting")
        print(f"  for patient Age and ER Status.")
    else:
        print(f"  Under the linear backdoor adjustment model adjusting for Age and ER status,")
        print(f"  the estimated causal effect is {ate_value:.2f} months.")
        print(f"  (Patients receiving hormone therapy survive an estimated {abs(ate_value):.2f} months")
        print(f"  fewer in this observational linear specification. This highlights critical clinical")
        print(f"  observational nuances: treatment selection bias, disease severity/stage confounding,")
        print(f"  and survival censoring dynamics to be further evaluated with advanced causal estimators).")
    print("=" * 80)

    return model, identified_estimand, estimate


if __name__ == "__main__":
    run_causal_estimation("brca_clinical_data.csv")
