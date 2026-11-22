from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

_DATE_HEADERS = {"date","datetime","time","timestamp","日付","日時","年月日","年月","month","period"}


def _name(src) -> str:
    return getattr(src, "name", str(src)).lower()


def _is_date_header(value) -> bool:
    return str(value).strip().lower().replace(" ", "").replace("　", "") in _DATE_HEADERS


def _parse_dates(values: pd.Series) -> pd.DatetimeIndex:
    text = values.astype(str).str.strip()
    if text.str.match(r"^\d{6}$").all():
        parsed = pd.to_datetime(text, format="%Y%m", errors="coerce")
    elif text.str.match(r"^\d{8}$").all():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if parsed.isna().any():
        bad = text[parsed.isna()].iloc[0]
        raise ValueError(f"日付を解釈できません: {bad}")
    return pd.DatetimeIndex(parsed)


def _orient_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    raw = raw.copy()
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty or raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError("時系列データにはヘッダーと3点以上の観測が必要です。")

    raw = raw.map(lambda x: x.strip() if isinstance(x, str) else x)
    first = raw.iloc[0].tolist()
    first_cell = first[0] if first else ""
    tail = pd.Series(first[1:])
    date_ratio = pd.to_datetime(tail.astype(str).str.strip(), errors="coerce").notna().mean() if len(tail) else 0

    # Transposed form:
    # date,2024-01-01,2024-02-01,...
    # return,0.01,0.02,...
    if _is_date_header(first_cell) and len(tail) >= 3 and date_ratio >= 0.75:
        raw = raw.T.reset_index(drop=True)
        return raw, "transposed"
    return raw.reset_index(drop=True), "row"


def load_table(src, sheet: str | None = None) -> pd.DataFrame:
    name = _name(src)
    if hasattr(src, "seek"):
        try:
            src.seek(0)
        except Exception:
            pass

    if name.endswith(".xlsx"):
        raw = pd.read_excel(src, sheet_name=sheet if sheet else 0, header=None, dtype=object)
    else:
        raw = pd.read_csv(
            src,
            sep=None,
            engine="python",
            header=None,
            dtype=object,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

    raw, orientation = _orient_raw(raw)
    header = [str(v).strip() for v in raw.iloc[0].tolist()]
    if any(not h for h in header):
        raise ValueError("空の列名があります。")
    lowered = [h.lower() for h in header]
    if len(set(lowered)) != len(lowered):
        raise ValueError("重複した列名があります。")

    df = raw.iloc[1:].copy()
    df.columns = header
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("データ行がありません。")

    index_col = header[0]
    idx = _parse_dates(df[index_col])
    if idx.duplicated().any():
        dup = idx[idx.duplicated()][0]
        raise ValueError(f"重複日付があります: {dup.date()}")

    df = df.drop(columns=[index_col])
    for col in df.columns:
        # Empty strings are intentionally preserved as NaN so prepare_series can
        # distinguish a broken internal gap from valid leading/trailing blanks.
        df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    df.index = idx
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        df.attrs["sorted_input"] = True
    else:
        df.attrs["sorted_input"] = False
    df.attrs["orientation"] = orientation
    return df


def prepare_series(df: pd.DataFrame, column: str, kind: str = "return") -> tuple[pd.Series, list[str]]:
    if column not in df.columns:
        raise ValueError(f"列が見つかりません: {column}")
    if kind not in {"return", "price"}:
        raise ValueError("kind must be return or price")

    raw = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = raw.notna()
    if not valid.any():
        raise ValueError("有効な数値がありません。")
    first = int(np.flatnonzero(valid.to_numpy())[0])
    last = int(np.flatnonzero(valid.to_numpy())[-1])
    segment = raw.iloc[first:last + 1]
    if segment.isna().any():
        bad_date = segment[segment.isna()].index[0]
        raise ValueError(f"時系列の途中に欠損または不正値があります: {bad_date.date()}")

    s = segment.astype(float)
    warnings: list[str] = []
    if len(s) < 3:
        raise ValueError("有効な観測が3点未満です。")

    if kind == "price":
        if (s <= 0).any():
            raise ValueError("価格系列には0以下の値を使用できません。")
        s = s.pct_change().dropna()
    else:
        if (s < -1).any():
            raise ValueError("-100%未満のリターンは不正です。")

    if len(s) < 3:
        raise ValueError("変化率変換後の観測が3点未満です。")

    sd = float(s.std(ddof=1))
    if np.isfinite(sd) and sd > 0:
        z = ((s - float(s.mean())) / sd).abs()
        extreme = int((z >= 5).sum())
        if extreme:
            warnings.append(f"平均から5標準偏差以上離れた観測が{extreme}件あります。自動除外はしていません。")

    if df.attrs.get("sorted_input"):
        warnings.append("日付が昇順でなかったため、安全に並べ替えました。")
    if df.attrs.get("orientation") == "transposed":
        warnings.append("転置形式を自動認識して通常の時系列へ変換しました。")
    return s, warnings
