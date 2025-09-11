#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess clinical table for survival + lymph node metastasis prediction.

Output CSV includes:
  - patient_id, time, event
  - ln_label (淋巴结转移标签，0/1)
  - standardized numeric features
  - one-hot categorical features
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATE_COLS_DEFAULT = [
    "admission_date", "surgery_date", "discharge_date",
    "date_survival", "last_follow_up_time"
]

CATEGORICAL_DEFAULT = [
    "gender", "age_group",            # 性别、年龄分组
    "T_standard", "T", "N", "M",      # TNM
    "stage_standard", "stage"         # 分期
]

OUTCOME_STATUS_COL = "status"
OUTCOME_TIME_MONTH_COL = "survival_month"
OUTCOME_TIME_DAY_COL = "survival_day"

LYMPH_NODE_COL = "lymph_node_metastasis"  # 淋巴结转移标签


def _ensure_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" not in df.columns:
        if "id" in df.columns:
            df["patient_id"] = df["id"]
        else:
            raise ValueError("Neither 'patient_id' nor 'id' found in clinical CSV.")
    return df


def _build_time_event(df: pd.DataFrame) -> pd.DataFrame:
    # event
    if OUTCOME_STATUS_COL not in df.columns:
        raise ValueError(f"Outcome column '{OUTCOME_STATUS_COL}' not found.")
    event_raw = df[OUTCOME_STATUS_COL]

    mapping = {
        "dead": 1, "死亡": 1, "deceased": 1, "death": 1, "1": 1, 1: 1, True: 1,
        "alive": 0, "存活": 0, "censored": 0, "0": 0, 0: 0, False: 0
    }
    df["event"] = event_raw.map(mapping).fillna(event_raw).astype(float)
    df["event"] = (df["event"] != 0).astype(int)

    # time
    if OUTCOME_TIME_MONTH_COL in df.columns:
        time_vals = pd.to_numeric(df[OUTCOME_TIME_MONTH_COL], errors="coerce")
    elif OUTCOME_TIME_DAY_COL in df.columns:
        time_vals = pd.to_numeric(df[OUTCOME_TIME_DAY_COL], errors="coerce") / 30.0
    else:
        raise ValueError("Neither 'survival_month' nor 'survival_day' found for time.")
    df["time"] = time_vals

    df["time"] = df["time"].replace([np.inf, -np.inf], np.nan)
    if df["time"].isna().all():
        raise ValueError("All 'time' values are NaN after cleaning.")
    df["time"] = df["time"].fillna(df["time"].median())

    return df


def preprocess_clinical(
    input_csv: str = "data/raw/clinical.csv",
    output_csv: str = "data/processed/clinical_processed.csv",
    date_cols=None,
    categorical_cols=None
):
    print(f"[INFO] Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    df = _ensure_patient_id(df)

    if date_cols is None:
        date_cols = DATE_COLS_DEFAULT
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_DEFAULT

    df = _build_time_event(df)

    # --- New: lymph node label ---
    if LYMPH_NODE_COL in df.columns:
        df["ln_label"] = pd.to_numeric(df[LYMPH_NODE_COL], errors="coerce").fillna(0).astype(int)
    else:
        df["ln_label"] = np.nan  # 如果不存在，填充 NaN，保持列结构

    # drop columns
    drop_cols = set(date_cols + [
        "id",
        OUTCOME_STATUS_COL,
        OUTCOME_TIME_MONTH_COL,
        OUTCOME_TIME_DAY_COL,
        LYMPH_NODE_COL  # 避免混进特征
    ])

    # categorical
    valid_cats = [c for c in categorical_cols if c in df.columns]
    for c in valid_cats:
        df[c] = df[c].astype(str).fillna("missing")

    exclude_for_numeric = set(valid_cats) | drop_cols | {"patient_id", "time", "event", "ln_label"}
    candidate_numeric = [c for c in df.columns if c not in exclude_for_numeric]

    numeric_cols = []
    for c in candidate_numeric:
        coerced = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = 1.0 - coerced.isna().mean()
        if valid_ratio >= 0.8:
            df[c] = coerced
            numeric_cols.append(c)
        else:
            valid_cats.append(c)
            df[c] = df[c].astype(str).fillna("missing")

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].median())

    print(f"[INFO] Valid categorical cols: {valid_cats}")
    print(f"[INFO] Valid numeric cols: {numeric_cols}")

    if valid_cats:
        df_cat = pd.get_dummies(df[valid_cats], drop_first=True, prefix=valid_cats)
    else:
        df_cat = pd.DataFrame(index=df.index)

    if numeric_cols:
        scaler = StandardScaler()
        df_num = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )
    else:
        df_num = pd.DataFrame(index=df.index)

    # final concat
    df_out = pd.concat([df[["patient_id", "time", "event", "ln_label"]], df_num, df_cat], axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Saved to: {output_csv}")
    print(f"[INFO] Shape: {df_out.shape}")
    print(f"[INFO] Example row:\n{df_out.head(1).T}")

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="data/raw/clinical.csv", type=str)
    parser.add_argument("--out_csv", default="data/processed/clinical_processed_multimodal.csv", type=str)
    args = parser.parse_args()

    preprocess_clinical(
        input_csv=args.in_csv,
        output_csv=args.out_csv
    )
