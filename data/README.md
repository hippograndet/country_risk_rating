# Data Directory Overview

This directory contains all datasets used throughout the project, organised by **stage in the data lifecycle**.

The guiding principle is:
- raw data is never modified
- interim data is reproducible
- processed data is directly usable for modeling

---

## Directory Structure

data/
├── 0-metadata/
├── 1-raw/
├── 2-interim/
├── 3-processed/
└── README.md

---

## 0. Metadata (`0-metadata/`)

**Purpose:**  
Static reference data that provides context and mappings for the main datasets. Prestructured and retrieved static datasets, containg information and codes for each countries.

**Contents:**
- `country_mapping.csv`  
  Mapping between country names, codes, and identifiers across data sources.
- `countries_info.csv`  
  Country-level metadata (region, income group, etc.).

**Notes:**
- These files are small, stable, and version-controlled.
- They are safe to track in git.

---

## 1. Raw data (`1-raw/`)

**Purpose:**  
Unmodified data from OECD Risk Ratings Data, before any processing. Raw pdfs of OECD reports at different dates, containing ratings, and the datasets extracted from the pdfs.

**Contents:**
- `oecd_country_ratings_pdfs/`  
  Original OECD country risk rating PDFs.
- `oecd_country_ratings_datasets/`  
  Raw OECD datasets as csv.

**Notes:**
- Files in this folder must never be edited manually.
- This folder is ignored by git due to file size and licensing constraints.
- Scripts in `src/data/` are responsible for reading from this directory.

---

## 2. Interim data (`2-interim/`)

**Purpose:**  
Intermediate, reproducible datasets produced during cleaning, merging, and feature preparation.


**Contents include:**
- `ratings_formatted_yearly.csv`
- `merged_dataset.csv`
- `worldbank_dataset_extracted.csv`
- OECD rating matrices (quarterly and yearly)
- Feature-selected intermediate datasets

**Notes:**
- These datasets are derived from raw data and metadata.
- They may change frequently as preprocessing evolves.
- This folder is not tracked in git.

---

## 3. Processed data (`3-processed/`)

**Purpose:**  
Final, canonical datasets used directly for modeling and evaluation.

**Contents:**
- `X.csv`  
  Final feature matrix.
- `y.csv`  
  Target variable aligned with `X.csv`.

**Notes:**
- These datasets represent the agreed-upon modeling inputs.
- Any model trained in this project should be reproducible from these files.
- Changes here should be deliberate and documented.

---

## Data Flow Summary

Raw data
↓
Interim transformations
↓
Final processed dataset
↓
Model training & evaluation