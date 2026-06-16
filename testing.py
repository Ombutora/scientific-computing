# =============================================================
# ICS 2207 SCIENTIFIC COMPUTING | WEEK 8
# Data Analysis with Pandas - Abusive Language Dataset
# READ-ONLY SECTIONS ONLY (no modifications to dataset)
# =============================================================

import pandas as pd
import numpy as np

# =============================================================
# SLIDE 03 | What is Pandas?
# =============================================================
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version:  {np.__version__}")


# =============================================================
# SLIDE 04 | Series vs DataFrame
# =============================================================
sev = pd.Series(['mild', 'moderate', 'severe'])
print(sev)


# =============================================================
# SLIDE 05 | Loading the Community Dataset
# =============================================================
df = pd.read_excel(r'C:\\Users\\Chris\\Desktop\\scientific computing\\SCT211-0347-2024_Sheng_AbusiveLanguage_dataset.xlsx', 
                   sheet_name='ALD Collection')
print("Shape:", df.shape)
print(df.head(3))


# =============================================================
# SLIDE 06 | Inspecting the Dataset Structure
# =============================================================
df.info()


# =============================================================
# SLIDE 07 | Selecting Data: Columns & Rows
# =============================================================
subset = df[['Entry ID', 'Language', 'Severity']]
print(subset.head())


# =============================================================
# SLIDE 08 | Filtering Data with Boolean Indexing
# =============================================================
severe_df = df[df['Severity'] == 'Severe']
print(severe_df[['Entry ID', 'Original Expression', 'Severity']].head())


# =============================================================
# SLIDE 09 | Handling Missing Data (detection only)
# =============================================================
print(df.isnull().sum())


# =============================================================
# SLIDE 12 | Statistical Summaries
# =============================================================
print(df['Severity'].value_counts())


# =============================================================
# SLIDE 13 | Groupby: Split-Apply-Combine
# =============================================================
category_stats = df.groupby('Primary Category').size()
print(category_stats.sort_values(ascending=False))


# =============================================================
# SLIDE 17 | Pivot Tables
# =============================================================
pivot = df.pivot_table(
    values='Entry ID',
    index='Language',
    columns='Severity',
    aggfunc='count',
    fill_value=0,
    margins=True
)
print(pivot)


# =============================================================
# SLIDE 18 | Automated Data Quality Report
# =============================================================
def dataset_quality_report(df):
    """Generate a quality report for the community dataset."""
    total = len(df.dropna(subset=['Entry ID']))

    # 1. Completeness Check
    missing = df.isnull().sum().sum()

    # 2. Ethics check: Anon? must be 'Yes'
    ethics_ok = (df['Anon?'].str.strip() == 'Yes').sum()

    # 3. Validity: Check severity values
    VALID_SEV = ['Mild', 'Moderate', 'Severe', 'Medium']
    invalid_sev = (~df['Severity'].isin(VALID_SEV)).sum()

    # Final Verdict
    passed = (ethics_ok == total and invalid_sev == 0 and total >= 50)

    print(f"Total entries   : {total}")
    print(f"Missing values  : {missing}")
    print(f"Ethics OK       : {ethics_ok}/{total}")
    print(f"Invalid severity: {invalid_sev}")
    print(f"READY FOR AI/ML : {passed}")

dataset_quality_report(df)