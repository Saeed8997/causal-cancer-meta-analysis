"""
Causal Meta-Analysis of Hormone Therapy in Breast Cancer
Step 1: Causal Data Setup and Variable Identification

Author: Causal AI Machine Learning Engineer
Date: 2026-10-06
"""

import os
import pandas as pd


def load_and_setup_causal_data(data_path: str = "brca_clinical_data.csv") -> pd.DataFrame:
    """
    Loads the clinical breast cancer dataset, identifies causal variables according
    to the causal framework, filters the dataset, and outputs an audit of missingness
    and data types.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("=" * 80)
    print("CAUSAL META-ANALYSIS: HORMONE THERAPY IN BREAST CANCER")
    print("STEP 1: DATA SETUP & CAUSAL VARIABLE IDENTIFICATION")
    print("=" * 80)
    print(f"\n[+] Loading clinical data from: {data_path} ...")
    raw_df = pd.read_csv(data_path, low_memory=False)
    print(f"[OK] Loaded dataset with shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns.\n")

    # Mapping variables based on Causal Inference Graph (DAG):
    # Treatment (T), Outcome (Y), Confounders (W/X)
    causal_mapping = [
        {
            "Causal Role": "Treatment (T)",
            "Semantic Name": "Hormone Therapy",
            "Column": "HORMONE_THERAPY",
            "Description": "Endocrine therapy indicator (YES/NO)",
        },
        {
            "Causal Role": "Outcome (Y_time)",
            "Semantic Name": "Overall Survival Time",
            "Column": "OS_MONTHS",
            "Description": "Survival duration in months",
        },
        {
            "Causal Role": "Outcome (Y_event)",
            "Semantic Name": "Overall Survival Status",
            "Column": "OS_STATUS",
            "Description": "Survival event status (0:LIVING, 1:DECEASED)",
        },
        {
            "Causal Role": "Confounder (W)",
            "Semantic Name": "Age (TCGA)",
            "Column": "AGE",
            "Description": "Patient age at diagnosis in TCGA cohort",
        },
        {
            "Causal Role": "Confounder (W)",
            "Semantic Name": "Age (METABRIC)",
            "Column": "AGE_AT_DIAGNOSIS",
            "Description": "Patient age at diagnosis in METABRIC cohort",
        },
        {
            "Causal Role": "Confounder (W)",
            "Semantic Name": "Tumor Stage",
            "Column": "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "Description": "Pathologic tumor stage (AJCC staging)",
        },
        {
            "Causal Role": "Confounder (W)",
            "Semantic Name": "ER Status",
            "Column": "ER_IHC",
            "Description": "Estrogen receptor status by IHC (Positive/Negative)",
        },
        {
            "Causal Role": "Confounder/Context (S)",
            "Semantic Name": "Study / Cohort Name",
            "Column": "studyId",
            "Description": "Study identifier (brca_metabric vs brca_tcga_pan_can_atlas_2018)",
        },
    ]

    # Verify which columns exist in the dataset
    existing_vars = [item for item in causal_mapping if item["Column"] in raw_df.columns]
    target_columns = [item["Column"] for item in existing_vars]

    print("[+] Isolating Causal Variables:")
    for item in existing_vars:
        print(f"  - [{item['Causal Role']}] {item['Semantic Name']}: column '{item['Column']}'")
    print()

    # Filter dataset to isolate causal columns
    causal_df = raw_df[target_columns].copy()

    # Generate summary of data types and missing values
    summary_records = []
    total_rows = len(causal_df)

    for item in existing_vars:
        col = item["Column"]
        missing_count = causal_df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        non_null_count = total_rows - missing_count
        dtype_str = str(causal_df[col].dtype)

        summary_records.append({
            "Causal Role": item["Causal Role"],
            "Variable": item["Semantic Name"],
            "Column": col,
            "Data Type": dtype_str,
            "Non-Null": non_null_count,
            "Missing": missing_count,
            "Missing (%)": f"{missing_pct:.2f}%",
        })

    summary_df = pd.DataFrame(summary_records)

    print("=" * 80)
    print("SUMMARY: CAUSAL VARIABLES, DATA TYPES & MISSING VALUES (OVERALL)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()

    # Cohort breakdown (TCGA vs METABRIC)
    if "studyId" in causal_df.columns:
        print("=" * 80)
        print("COHORT BREAKDOWN (TCGA vs METABRIC)")
        print("=" * 80)
        cohorts = causal_df["studyId"].dropna().unique()
        for cohort in cohorts:
            cohort_sub = causal_df[causal_df["studyId"] == cohort]
            print(f"\nCohort: {cohort} (n = {len(cohort_sub)})")
            cohort_records = []
            for item in existing_vars:
                col = item["Column"]
                miss = cohort_sub[col].isna().sum()
                pct = (miss / len(cohort_sub)) * 100
                cohort_records.append({
                    "Variable": item["Semantic Name"],
                    "Column": col,
                    "Non-Null": len(cohort_sub) - miss,
                    "Missing": miss,
                    "Missing (%)": f"{pct:.2f}%"
                })
            print(pd.DataFrame(cohort_records).to_string(index=False))

    print("\n" + "=" * 80)
    print("CAUSAL DATASET PREVIEW (First 5 Rows)")
    print("=" * 80)
    print(causal_df.head(5).to_string())
    print("=" * 80)

    return causal_df


if __name__ == "__main__":
    load_and_setup_causal_data("brca_clinical_data.csv")
