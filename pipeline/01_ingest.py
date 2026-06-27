"""
Phase 1 (Improved): Load ALL GDELT CSV files from a folder.
- Skips incomplete (.filepart) files
- Handles nested folders
- Deduplicates URLs
- Saves clean parquet

Usage:
    python 01_ingest_folder.py --folder /path/to/2025 --output data/raw/gdelt_filtered.parquet
"""

import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm

# ── Columns we actually need ───────────────────────────────────────────────
COLUMNS_TO_KEEP = [
    "GlobalEventID", "Day", "Year",
    "Actor1Name", "Actor1CountryCode",
    "Actor2Name", "Actor2CountryCode",
    "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles",
    "AvgTone",
    "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_Lat", "ActionGeo_Long",
    "DATEADDED", "SOURCEURL"
]

QUADCLASS_MAP = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict"
}

# Full GDELT header list (for raw files without header)
GDELT_HEADERS = [
    "GlobalEventID","Day","MonthYear","Year","FractionDate",
    "Actor1Code","Actor1Name","Actor1CountryCode","Actor1KnownGroupCode",
    "Actor1EthnicCode","Actor1Religion1Code","Actor1Religion2Code",
    "Actor1Type1Code","Actor1Type2Code","Actor1Type3Code",
    "Actor2Code","Actor2Name","Actor2CountryCode","Actor2KnownGroupCode",
    "Actor2EthnicCode","Actor2Religion1Code","Actor2Religion2Code",
    "Actor2Type1Code","Actor2Type2Code","Actor2Type3Code",
    "IsRootEvent","EventCode","EventBaseCode","EventRootCode",
    "QuadClass","GoldsteinScale","NumMentions","NumSources","NumArticles",
    "AvgTone",
    "Actor1Geo_Type","Actor1Geo_FullName","Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code","Actor1Geo_Lat","Actor1Geo_Long","Actor1Geo_FeatureID",
    "Actor2Geo_Type","Actor2Geo_FullName","Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code","Actor2Geo_Lat","Actor2Geo_Long","Actor2Geo_FeatureID",
    "ActionGeo_Type","ActionGeo_FullName","ActionGeo_CountryCode",
    "ActionGeo_ADM1Code","ActionGeo_Lat","ActionGeo_Long","ActionGeo_FeatureID",
    "DATEADDED","SOURCEURL"
]


# ──────────────────────────────────────────────────────────────────────────
def detect_has_header(path: str) -> bool:
    with open(path, "r", errors="replace") as f:
        first_line = f.readline()
    return "GlobalEventID" in first_line or "SOURCEURL" in first_line


def read_one_csv(path: str) -> pd.DataFrame | None:
    try:
        has_header = detect_has_header(path)

        if has_header:
            df = pd.read_csv(
                path,
                usecols=lambda c: c in COLUMNS_TO_KEEP,
                low_memory=False,
                on_bad_lines="skip",
            )
        else:
            df = pd.read_csv(
                path,
                header=None,
                names=GDELT_HEADERS,
                sep="\t",
                usecols=lambda c: c in COLUMNS_TO_KEEP,
                low_memory=False,
                on_bad_lines="skip",
            )

        return df

    except Exception as e:
        print(f"[WARN] Failed: {os.path.basename(path)} -> {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────
def find_csv_files(folder: str) -> list[str]:
    patterns = [
        os.path.join(folder, "*.CSV"),
        os.path.join(folder, "*.csv"),
        os.path.join(folder, "**", "*.CSV"),
        os.path.join(folder, "**", "*.csv"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    files = sorted(set(files))

    clean_files = []
    skipped = 0

    for f in files:
        if f.endswith(".filepart"):
            skipped += 1
            continue
        clean_files.append(f)

    print(f"Skipped {skipped} incomplete (.filepart) files")

    return clean_files


# ──────────────────────────────────────────────────────────────────────────
def load_all_csvs(folder: str) -> pd.DataFrame:
    files = find_csv_files(folder)

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    print(f"Found {len(files):,} CSV files")

    chunks = []
    total_rows = 0
    failed = 0

    for path in tqdm(files, desc="Reading CSVs"):
        df = read_one_csv(path)

        if df is None or df.empty:
            failed += 1
            continue

        chunks.append(df)
        total_rows += len(df)

    if not chunks:
        raise ValueError("All files failed to load")

    print(f"\nLoaded {total_rows:,} rows from {len(chunks)} files ({failed} failed)")

    return pd.concat(chunks, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
def clean_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.dropna(subset=["SOURCEURL"])
    print(f"Dropped {before - len(df):,} rows without URL")

    df = df[df["SOURCEURL"].str.startswith("http", na=False)]

    df["date_str"] = pd.to_datetime(
        df["Day"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    df["event_type_label"] = df["QuadClass"].map(QUADCLASS_MAP).fillna("Unknown")

    df["Actor1Name"] = df["Actor1Name"].fillna("Unknown")
    df["Actor2Name"] = df["Actor2Name"].fillna("Unknown")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["SOURCEURL"]).reset_index(drop=True)

    print(f"Removed {before_dedup - len(df):,} duplicate URLs")
    print(f"Final rows: {len(df):,}")

    return df


# ──────────────────────────────────────────────────────────────────────────
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    🔥 Fix mixed datatype issues for Parquet (IMPORTANT)
    """
    print("\nFixing datatypes for Parquet...")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    return df


# ──────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n── Summary ───────────────────────────────")
    print(f"Articles: {len(df):,}")
    print(f"Date range: {df['date_str'].min()} → {df['date_str'].max()}")

    print("\nTop locations:")
    print(df["ActionGeo_FullName"].value_counts().head(5))

    print("\nEvent types:")
    print(df["event_type_label"].value_counts())

    print("──────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", default="data/raw/gdelt_filtered.parquet")
    args = parser.parse_args()

    df = load_all_csvs(args.folder)

    print("\nCleaning...")
    df = clean_gdelt(df)

    print_summary(df)

    # 🔥 FIX APPLIED HERE
    df = fix_dtypes(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df.to_parquet(
        args.output,
        index=False,
        engine="pyarrow"
    )

    print(f"Saved: {args.output}")
    print(f"Size: {os.path.getsize(args.output)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()