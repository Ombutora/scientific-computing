import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ICS 2207 Scientific Computing | Week 8\n",
    "## Data Analysis with Pandas - Abusive Language Dataset\n",
    "\n",
    "Welcome! In this notebook, we'll dive Python's **Pandas** library to perform data analysis. We'll be working with an 'Abusive Language Dataset'. The goal here is to interactively load, inspect, clean, and visualize the data while keeping the original file completely untouched."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Set a nice plotting style for our graphs\n",
    "plt.style.use('ggplot')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. What is Pandas? (Slide 03)\n",
    "Pandas is an open-source library built on top of NumPy. It provides fast, flexible, and expressive data structures designed to make working with \"relational\" or \"labeled\" data both easy and intuitive.\n",
    "\n",
    "**What this does:** It simply checks what versions we're working with.\n",
    "**Expected Output:** You will see the versions of Pandas (e.g., `2.2.3`) and NumPy (e.g., `2.3.0`) currently installed on your computer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Pandas version: {pd.__version__}\")\n",
    "print(f\"NumPy version:  {np.__version__}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Series vs DataFrame (Slide 04)\n",
    "The two primary structures in Pandas are the `Series` (1-dimensional, like a column in Excel) and the `DataFrame` (2-dimensional, like the whole table).\n",
    "\n",
    "**What this does:** Here, we create a simple `Series` to represent potential severity levels.\n",
    "**Expected Output:** It will print a simple 3-item list containing `mild`, `moderate`, and `severe` with their index numbers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sev = pd.Series(['mild', 'moderate', 'severe'])\n",
    "print(\"Our Pandas Series:\")\n",
    "print(sev)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Loading the Community Dataset (Slide 05)\n",
    "Time to get our hands dirty! We are loading the dataset from the provided Excel file. \n",
    "\n",
    "**What this does:** It loads the data into a Pandas DataFrame. `.shape` tells us the exact size as (Rows, Columns). `.head()` prints the first few rows so we can quickly see what the data looks like.\n",
    "**Expected Output:** It will print out a shape like `Dataset Shape: 60 rows, 14 columns`, followed by a nicely formatted visual table of the first 3 rows in your dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "file_path = r'C:\\Users\\Chris\\Desktop\\scientific computing\\SCT211-0347-2024_Sheng_AbusiveLanguage_dataset.xlsx'\n",
    "# Note: I am loading it to df (DataFrame)\n",
    "df = pd.read_excel(file_path, sheet_name='ALD Collection')\n",
    "\n",
    "print(f\"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\")\n",
    "print(\"\\nFirst 3 rows of the dataset:\")\n",
    "display(df.head(3)) # Using display() which formats beautifully in Jupyter!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Inspecting the Dataset Structure (Slide 06)\n",
    "`df.info()` is an incredibly helpful method.\n",
    "\n",
    "**What this does:** It gives us a technical summary of the data: which columns exist, how many are 'non-null' (not empty!), and what the data type is.\n",
    "**Expected Output:** A list of all your columns highlighting that you have 60 total entries, and showing any columns containing text ('object')."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Selecting Data: Columns & Rows (Slide 07)\n",
    "Often, we don't need all the columns. \n",
    "\n",
    "**What this does:** We can slice our DataFrame by providing a list of the exact column names we want, returning a smaller, more focused DataFrame.\n",
    "**Expected Output:** It will print a fresh mini-table showing only the `Entry ID`, `Language`, and `Severity`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Selecting specific columns\n",
    "subset = df[['Entry ID', 'Language', 'Severity']]\n",
    "\n",
    "print(\"Showing the newly created subset:\")\n",
    "display(subset.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Filtering Data with Boolean Indexing (Slide 08)\n",
    "What if we only want to see rows where the language is particularly toxic? \n",
    "\n",
    "**What this does:** Pandas checks every row. If `df['Severity'] == 'Severe'` is True, it keeps the row; if False, it drops it.\n",
    "**Expected Output:** It will show an even smaller table of only the precise rows that have been flagged with a `Severe` toxicity rating."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Applying the boolean mask filter\n",
    "severe_df = df[df['Severity'] == 'Severe']\n",
    "\n",
    "print(f\"Total 'Severe' entries found: {len(severe_df)}\")\n",
    "print(\"\\nFirst 5 severe entries:\")\n",
    "display(severe_df[['Entry ID', 'Original Expression', 'Severity']].head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7. Handling Missing Data (Slide 09)\n",
    "Real datasets are rarely perfect. We need to find out where information is missing. \n",
    "\n",
    "**What this does:** By combining `.isnull()` with `.sum()`, we can get a quick count of missing values per column.\n",
    "**Expected Output:** It scans the entire 60-row worksheet and lists out any columns (along with the exact count) where data missing or blank."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_data = df.isnull().sum()\n",
    "print(\"Total missing values per column:\\n\")\n",
    "print(missing_data[missing_data > 0]) # Filters to only show columns that actually have missing data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8. Statistical Summaries & Visualization (Slide 12)\n",
    "Let's summarize a categorical column. `.value_counts()` totals up how many times each specific label appears in a column.\n",
    "\n",
    "**What this does:** It graphs those exact counts side-by-side using `matplotlib`.\n",
    "**Expected Output:** It will print raw numbers showing how many rows are `Mild` versus `Severe`. Beneath it, a colorful **Bar Chart** will pop up visually displaying those exact counts side-by-side."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "severity_counts = df['Severity'].value_counts()\n",
    "print(\"Severity Count Distribution:\")\n",
    "print(severity_counts)\n",
    "\n",
    "# Visualization: Bar Chart\n",
    "plt.figure(figsize=(8, 5))\n",
    "severity_counts.plot(kind='bar', color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'])\n",
    "plt.title('Distribution of Toxicity Severity', fontweight='bold')\n",
    "plt.xlabel('Severity Level')\n",
    "plt.ylabel('Count')\n",
    "plt.xticks(rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9. Groupby: Split-Apply-Combine (Slide 13)\n",
    "The `.groupby()` method splits the data into groups based on some criteria, applies a function (like `.size()` to get counts), and combines the results.\n",
    "\n",
    "**What this does:** Let's see the most common primary categories targeted in this text and graph the top 5.\n",
    "**Expected Output:** You'll see a **Horizontal Bar Chart** visually ranking the top 5 targeted categories."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "category_stats = df.groupby('Primary Category').size().sort_values(ascending=False)\n",
    "\n",
    "print(\"Primary Category counts:\")\n",
    "print(category_stats.head())\n",
    "\n",
    "# Visualization: Horizontal Bar Chart\n",
    "plt.figure(figsize=(10, 5))\n",
    "category_stats.head(5).plot(kind='barh', color='#9b59b6')\n",
    "plt.title('Top 5 Target Categories', fontweight='bold')\n",
    "plt.xlabel('Number of Entries')\n",
    "plt.ylabel('Category')\n",
    "plt.gca().invert_yaxis() # Invert to have the highest on top\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10. Pivot Tables (Slide 17)\n",
    "Pivot tables provide multidimensional summarization. \n",
    "\n",
    "**What this does:** Here, we cross-tabulate `Language` and `Severity` to understand how the severity spreads across different dialects.\n",
    "**Expected Output:** A **Pivot Table** will appear showing a cross-grid of Language vs Severity, letting you see trends immediately."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pivot = df.pivot_table(\n",
    "    values='Entry ID',\n",
    "    index='Language',\n",
    "    columns='Severity',\n",
    "    aggfunc='count',\n",
    "    fill_value=0,\n",
    "    margins=True\n",
    ")\n",
    "\n",
    "display(pivot)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11. Automated Data Quality Report (Slide 18)\n",
    "Before running any Machine Learning algorithm, you'll generally want to certify that the dataset follows specific ethical and validity rules.\n",
    "\n",
    "**What this does:** Gives a custom summary panel verifying the integrity of the data (Total valid, Missing fields, Ethics checks, Invalid severities).\n",
    "**Important Update:** Note that I changed the last bit of the original script! Instead of just printing `True` or `False` for whether it's ready for ML, I updated the logic to print `READY FOR AI/ML : [YES]` or `[NO] (Requires Cleaning)`. I also added improved logic internally to safely strip accidental spaces from the 'Anon?' column so it evaluates properly.\n",
    "**Expected Output:** A neat console display acting as a checkpoint showing you if the dataset is entirely ready to use!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dataset_quality_report(df):\n",
    "    \"\"\"Generate a human-readable quality report for the dataset.\"\"\"\n",
    "    total = len(df.dropna(subset=['Entry ID']))\n",
    "\n",
    "    # 1. Completeness Check\n",
    "    missing = df.isnull().sum().sum()\n",
    "\n",
    "    # 2. Ethics check: Ensure all records are marked Anonymous\n",
    "    # Using .str.strip() protects us against accidental spaces like ' Yes'\n",
    "    ethics_ok = (df['Anon?'].str.strip() == 'Yes').sum()\n",
    "\n",
    "    # 3. Validity: Ensure severity adheres precisely to allowed types\n",
    "    VALID_SEV = ['Mild', 'Moderate', 'Severe', 'Medium']\n",
    "    invalid_sev = (~df['Severity'].isin(VALID_SEV)).sum()\n",
    "\n",
    "    # Final Verdict\n",
    "    passed = (ethics_ok == total and invalid_sev == 0 and total >= 50)\n",
    "\n",
    "    print(\"=\" * 45)\n",
    "    print(\"        [DATASET QUALITY REPORT]\")\n",
    "    print(\"=\" * 45)\n",
    "    print(f\"Total valid entries : {total}\")\n",
    "    print(f\"Total missing fields: {missing}\")\n",
    "    print(f\"Ethics OK (Anon=Yes): {ethics_ok} / {total}\")\n",
    "    print(f\"Invalid severities  : {invalid_sev}\")\n",
    "    print(\"-\" * 45)\n",
    "    \n",
    "    if passed:\n",
    "        print(\"READY FOR AI/ML     : [YES]\")\n",
    "    else:\n",
    "        print(\"READY FOR AI/ML     : [NO] (Requires Cleaning)\")\n",
    "    print(\"=\" * 45)\n",
    "\n",
    "dataset_quality_report(df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

out_path = r'c:\Users\Chris\Desktop\scientific computing\testing_analysis.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully wrote {out_path}")
