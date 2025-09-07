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

def _ensure_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    # 如果没有 patient_id，但有 id，则复制一列
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

    # 尽量鲁棒：把可能的字符串/非常规编码映射到 {0,1}
    mapping = {
        "dead": 1, "死亡": 1, "deceased": 1, "death": 1, "1": 1, 1: 1, True: 1,
        "alive": 0, "存活": 0, "censored": 0, "0": 0, 0: 0, False: 0
    }
    df["event"] = event_raw.map(mapping).fillna(event_raw).astype(float)
    # 若仍不是 {0,1}，则将非零视为 1
    df["event"] = (df["event"] != 0).astype(int)

    # time：优先用 survival_month，其次 survival_day / 30
    time_vals = None
    if OUTCOME_TIME_MONTH_COL in df.columns:
        time_vals = pd.to_numeric(df[OUTCOME_TIME_MONTH_COL], errors="coerce")
    elif OUTCOME_TIME_DAY_COL in df.columns:
        time_vals = pd.to_numeric(df[OUTCOME_TIME_DAY_COL], errors="coerce") / 30.0
    else:
        raise ValueError(
            f"Neither '{OUTCOME_TIME_MONTH_COL}' nor '{OUTCOME_TIME_DAY_COL}' found for time."
        )
    df["time"] = time_vals

    # 基本清洗
    df["time"] = df["time"].replace([np.inf, -np.inf], np.nan)
    if df["time"].isna().all():
        raise ValueError("All 'time' values are NaN after cleaning.")
    # 用中位数填补少量缺失
    df["time"] = df["time"].fillna(df["time"].median())

    return df

def preprocess_clinical(
    input_csv: str = "data/raw/clinical.csv",
    output_csv: str = "data/processed/clinical_processed.csv",
    date_cols = None,
    categorical_cols = None
):
    print(f"[INFO] Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    df = _ensure_patient_id(df)

    # 列集合（按你给表设计）
    if date_cols is None:
        date_cols = DATE_COLS_DEFAULT
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_DEFAULT

    # 生成 time / event
    df = _build_time_event(df)

    # 要剔除的列
    drop_cols = set(date_cols + [
        "id",                      # 如果存在就丢掉，以 patient_id 为准
        OUTCOME_STATUS_COL,        # 已组装到 event
        OUTCOME_TIME_MONTH_COL,    # 已组装到 time
        OUTCOME_TIME_DAY_COL       # 已组装到 time
    ])

    # 有效的分类列（存在于表中）
    valid_cats = [c for c in categorical_cols if c in df.columns]
    # 先统一转成字符串，避免数值型 TNM 被当成数值处理
    for c in valid_cats:
        df[c] = df[c].astype(str).fillna("missing")

    # 数值候选列：去掉 patient_id / 时间事件 / 日期列 / 分类列
    exclude_for_numeric = set(valid_cats) | drop_cols | {"patient_id", "time", "event"}
    candidate_numeric = [c for c in df.columns if c not in exclude_for_numeric]

    # 只保留确实是数值或可转为数值的
    numeric_cols = []
    for c in candidate_numeric:
        # 尝试转为数值，统计非 NaN 的占比，过低就不当数值列
        coerced = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = 1.0 - coerced.isna().mean()
        if valid_ratio >= 0.8:  # 至少 80% 能转为数值
            df[c] = coerced
            numeric_cols.append(c)
        else:
            # 如果转换很差，退回当成分类特征
            valid_cats.append(c)
            df[c] = df[c].astype(str).fillna("missing")

    # 清洗数值列：替换 inf/缺失 → 中位数
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].median())

    print(f"[INFO] Valid categorical cols: {valid_cats}")
    print(f"[INFO] Valid numeric cols: {numeric_cols}")

    # One-hot 编码分类列（若无则给空表）
    if valid_cats:
        df_cat = pd.get_dummies(df[valid_cats], drop_first=True, prefix=valid_cats)
    else:
        df_cat = pd.DataFrame(index=df.index)

    # 标准化数值列（若无则给空表）
    if numeric_cols:
        scaler = StandardScaler()
        df_num = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )
    else:
        df_num = pd.DataFrame(index=df.index)

    # 拼接：patient_id + time + event + 特征
    df_out = pd.concat([df[["patient_id", "time", "event"]], df_num, df_cat], axis=1)

    # 保存
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Saved to: {output_csv}")
    print(f"[INFO] Shape: {df_out.shape}")
    print(f"[INFO] Example row:\n{df_out.head(1).T}")

    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="data/raw/clinical.csv", type=str)
    parser.add_argument("--out_csv", default="data/processed/clinical_processed.csv", type=str)
    args = parser.parse_args()

    preprocess_clinical(
        input_csv=args.in_csv,
        output_csv=args.out_csv
    )
