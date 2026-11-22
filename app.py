"""将来予測ラボ（Forecast Lab）.

価格・リターン系列を入力として、モンテカルロ / Bootstrap / GARCH で
リターン・ボラティリティの将来予測を行う Streamlit アプリ。

入力ソース:
- .shared/factor_data/ の既存ファイル（Step 6 の規約に従う）
- Excel/CSV のアップロード

出力先: 40-finance/forecast/work/<timestamp>/
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from forecast_data import load_table as load_table_strict, prepare_series

APP_DIR = Path(__file__).resolve().parent
APPS_DIR = APP_DIR.parents[1]
SHARED_DIR = APPS_DIR / ".shared" / "factor_data"
WORK_ROOT = APP_DIR / "work"
WORK_ROOT.mkdir(parents=True, exist_ok=True)


# ============ データ読込 ============
def list_shared() -> list[Path]:
    if not SHARED_DIR.exists():
        return []
    files: list[Path] = []
    for ext in ("*.xlsx", "*.csv"):
        files.extend(SHARED_DIR.glob(ext))
    return sorted(files)


def list_sheets(path: Path) -> list[str]:
    if path.suffix.lower() != ".xlsx":
        return []
    try:
        return pd.ExcelFile(path).sheet_names
    except Exception:
        return []


def load_table(src, sheet: str | None) -> pd.DataFrame | None:
    try:
        return load_table_strict(src, sheet)
    except Exception as e:
        st.error(f"読み込み失敗: {e}")
        return None


def make_run_dir(prefix: str) -> Path:
    d = WORK_ROOT / f"{prefix}_{datetime.now():%Y%m%d-%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============ 入力UI（共通） ============
def input_panel(key_prefix: str) -> tuple[pd.Series | None, str, str]:
    """戻り値: (returns_series, source_label, freq_label)"""
    source = st.radio(
        "📂 データソース",
        ["☁️ 共有ストレージ", "📥 ファイルをアップロード"],
        horizontal=True,
        key=f"{key_prefix}_src",
    )

    df: pd.DataFrame | None = None
    label = ""
    if source.startswith("☁️"):
        files = list_shared()
        if not files:
            st.info("共有ストレージが空です。factor-data-hub で取得するか、アップロードを使ってください。")
            return None, "", ""
        pick = st.selectbox("ファイル選択", ["（選択なし）"] + [p.name for p in files], key=f"{key_prefix}_file")
        if pick == "（選択なし）":
            return None, "", ""
        path = SHARED_DIR / pick
        sheet = None
        if path.suffix.lower() == ".xlsx":
            sheets = list_sheets(path)
            if len(sheets) > 1:
                sheet = st.selectbox("シート", sheets, key=f"{key_prefix}_sheet")
            elif sheets:
                sheet = sheets[0]
        df = load_table(path, sheet)
        label = f"{pick}" + (f" / {sheet}" if sheet else "")
    else:
        up = st.file_uploader("CSV または Excel", type=["csv", "xlsx"], key=f"{key_prefix}_up")
        if up is None:
            return None, "", ""
        sheet = None
        if up.name.lower().endswith(".xlsx"):
            tmp = WORK_ROOT / f"_tmp_{up.name}"
            tmp.write_bytes(up.getbuffer())
            sheets = list_sheets(tmp)
            if len(sheets) > 1:
                sheet = st.selectbox("シート", sheets, key=f"{key_prefix}_sheet")
            elif sheets:
                sheet = sheets[0]
            df = load_table(tmp, sheet)
        else:
            df = load_table(up, None)
        label = up.name + (f" / {sheet}" if sheet else "")

    if df is None or df.empty:
        return None, label, ""

    df = df.dropna(how="all")
    st.success(f"読込: {len(df)} 行 × {len(df.columns)} 列")
    with st.expander("先頭プレビュー"):
        st.dataframe(df.head(8), use_container_width=True)

    col = st.selectbox("対象列（リターン or 価格）", list(df.columns), key=f"{key_prefix}_col")
    series_kind = st.radio(
        "この列の種類",
        ["リターン（既に変化率）", "価格（変化率に変換する）"],
        horizontal=True,
        key=f"{key_prefix}_kind",
    )

    try:
        s, input_warnings = prepare_series(
            df,
            col,
            "price" if series_kind.startswith("価格") else "return",
        )
    except ValueError as e:
        st.error(f"時系列入力エラー: {e}")
        return None, label, ""
    for warning in input_warnings:
        st.warning(warning)

    # 頻度の自動推定
    freq = "?"
    if isinstance(s.index, pd.DatetimeIndex) and len(s) > 2:
        days = (s.index[-1] - s.index[0]).days / max(len(s) - 1, 1)
        if days <= 2:
            freq = "daily"
        elif days <= 10:
            freq = "weekly"
        elif days <= 40:
            freq = "monthly"
        else:
            freq = "quarterly"

    c1, c2, c3 = st.columns(3)
    c1.metric("サンプル", f"{len(s)}")
    c2.metric("平均", f"{s.mean():.4%}")
    c3.metric("標準偏差", f"{s.std():.4%}")
    st.caption(f"推定頻度: **{freq}** / 期間: {s.index.min()} 〜 {s.index.max()}")

    return s, f"{label} :: {col}", freq


# ============ Tab 1: モンテカルロ ============
def render_monte_carlo() -> None:
    st.subheader("🎲 モンテカルロ予測")
    st.caption("ヒストリカル収益率の平均・分散をもとに、将来パスをシミュレーションします。")
    s, label, freq = input_panel("mc")
    if s is None:
        return

    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.number_input("予測期間（ステップ数）", 1, 1000, 60, key="mc_h")
    n_paths = c2.number_input("パス数", 100, 50000, 2000, step=100, key="mc_n")
    method = c3.selectbox("方式", ["GBM（正規分布）", "Bootstrap（実データ再抽出）"], key="mc_m")
    seed = c4.number_input("乱数シード", 0, 99999, 42, key="mc_seed")

    initial = st.number_input("初期値（100 にすると指数表示）", 0.01, 1e6, 100.0, key="mc_init")

    if st.button("🚀 実行", type="primary", key="mc_run"):
        rng = np.random.default_rng(int(seed))
        mu = float(s.mean())
        sigma = float(s.std(ddof=1))

        if method.startswith("GBM"):
            shocks = rng.normal(mu, sigma, size=(int(horizon), int(n_paths)))
        else:
            arr = s.to_numpy()
            idx = rng.integers(0, len(arr), size=(int(horizon), int(n_paths)))
            shocks = arr[idx]

        # 累積パス（指数）
        paths = np.empty((int(horizon) + 1, int(n_paths)))
        paths[0] = initial
        paths[1:] = initial * np.cumprod(1.0 + shocks, axis=0)

        # 統計
        pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        summary = pd.DataFrame(
            {
                "p05": pct[0], "p25": pct[1], "median": pct[2],
                "p75": pct[3], "p95": pct[4], "mean": paths.mean(axis=1),
            }
        )
        summary.index.name = "step"

        st.markdown("### 予測サマリー（パーセンタイル）")
        st.line_chart(summary[["p05", "p25", "median", "p75", "p95"]])

        terminal = paths[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("終端 中央値", f"{np.median(terminal):.2f}")
        c2.metric("終端 平均", f"{terminal.mean():.2f}")
        c3.metric("終端 5%分位", f"{np.percentile(terminal, 5):.2f}")
        c4.metric("終端 95%分位", f"{np.percentile(terminal, 95):.2f}")

        # VaR / CVaR（終端のリターン基準）
        rets = terminal / initial - 1.0
        var95 = -np.percentile(rets, 5)
        cvar95 = -rets[rets <= np.percentile(rets, 5)].mean()
        st.markdown("### リスク指標（終端リターン基準）")
        c1, c2 = st.columns(2)
        c1.metric("VaR (95%)", f"{var95:.2%}")
        c2.metric("CVaR (95%)", f"{cvar95:.2%}")

        with st.expander("ヒストグラム: 終端値の分布"):
            hist = pd.Series(terminal).rename("terminal")
            st.bar_chart(np.histogram(terminal, bins=50)[0])
            st.dataframe(hist.describe().to_frame(), use_container_width=True)

        # 保存
        out_dir = make_run_dir("monte_carlo")
        summary.to_csv(out_dir / "mc_percentiles.csv")
        pd.DataFrame(paths).to_csv(out_dir / "mc_paths.csv", index_label="step")
        meta = {
            "source": label, "freq": freq, "method": method,
            "horizon": int(horizon), "n_paths": int(n_paths),
            "mu": mu, "sigma": sigma, "initial": float(initial), "seed": int(seed),
        }
        pd.Series(meta).to_csv(out_dir / "mc_metadata.csv")
        st.success(f"💾 保存: `{out_dir}`")
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ パーセンタイルCSV",
            (out_dir / "mc_percentiles.csv").read_bytes(),
            file_name="mc_percentiles.csv",
            key="mc_dl_pct",
        )
        c2.download_button(
            "⬇️ 全パスCSV（重い）",
            (out_dir / "mc_paths.csv").read_bytes(),
            file_name="mc_paths.csv",
            key="mc_dl_paths",
        )


# ============ Tab 2: GARCH ============
def render_garch() -> None:
    st.subheader("📈 GARCH ボラティリティ予測")
    st.caption("GARCH(1,1) でボラティリティの時間変化を捉え、将来のボラティリティを予測します。")

    try:
        from arch import arch_model
    except ImportError:
        st.error(
            "`arch` パッケージが未インストールです。次のコマンドで導入してください:\n\n"
            "```\npip install arch\n```"
        )
        return

    s, label, freq = input_panel("garch")
    if s is None:
        return

    c1, c2, c3 = st.columns(3)
    p = c1.number_input("p（GARCH 項）", 1, 5, 1, key="g_p")
    q = c2.number_input("q（ARCH 項）", 1, 5, 1, key="g_q")
    horizon = c3.number_input("予測期間", 1, 250, 30, key="g_h")

    c4, c5 = st.columns(2)
    dist = c4.selectbox("誤差分布", ["normal", "t", "skewt", "ged"], key="g_dist")
    mean_model = c5.selectbox("平均モデル", ["Constant", "Zero", "AR"], key="g_mean")

    rescale = st.checkbox(
        "リターンを 100 倍にスケールして推定（収束安定）",
        value=True,
        help="日次・月次の小さなリターンは数値計算が不安定になりやすいので、慣例的に %表記で推定します。",
        key="g_scale",
    )

    if st.button("🚀 実行", type="primary", key="g_run"):
        try:
            y = s.dropna()
            scale = 100.0 if rescale else 1.0
            am = arch_model(
                y * scale,
                mean=mean_model,
                vol="GARCH",
                p=int(p), q=int(q),
                dist=dist,
            )
            with st.spinner("推定中..."):
                res = am.fit(disp="off")
            st.markdown("### 推定結果")
            st.code(str(res.summary()), language="text")

            # 適合ボラ
            cond_vol = pd.Series(res.conditional_volatility / scale, index=y.index, name="cond_vol")
            st.markdown("### 過去の条件付きボラティリティ")
            st.line_chart(cond_vol)

            # 予測
            fc = res.forecast(horizon=int(horizon), reindex=False)
            # variance の取り出し
            var_fc = fc.variance.values[-1] / (scale ** 2)
            vol_fc = np.sqrt(var_fc)
            mean_fc = fc.mean.values[-1] / scale

            fc_df = pd.DataFrame({
                "h": np.arange(1, int(horizon) + 1),
                "mean_return": mean_fc,
                "volatility": vol_fc,
                "variance": var_fc,
            }).set_index("h")
            st.markdown("### 予測ボラティリティ")
            st.line_chart(fc_df[["volatility"]])
            st.dataframe(fc_df.style.format("{:.6f}"), use_container_width=True)

            # 保存
            out_dir = make_run_dir("garch")
            cond_vol.to_csv(out_dir / "garch_cond_volatility.csv")
            fc_df.to_csv(out_dir / "garch_forecast.csv")
            meta = {
                "source": label, "freq": freq,
                "p": int(p), "q": int(q), "horizon": int(horizon),
                "dist": dist, "mean_model": mean_model, "rescale": rescale,
                "aic": float(res.aic), "bic": float(res.bic),
                "loglikelihood": float(res.loglikelihood),
            }
            pd.Series(meta).to_csv(out_dir / "garch_metadata.csv")
            (out_dir / "garch_summary.txt").write_text(str(res.summary()), encoding="utf-8")
            st.success(f"💾 保存: `{out_dir}`")

            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "⬇️ 予測CSV",
                (out_dir / "garch_forecast.csv").read_bytes(),
                file_name="garch_forecast.csv",
                key="g_dl_fc",
            )
            c2.download_button(
                "⬇️ 条件付ボラCSV",
                (out_dir / "garch_cond_volatility.csv").read_bytes(),
                file_name="garch_cond_volatility.csv",
                key="g_dl_cv",
            )
            c3.download_button(
                "⬇️ 推定サマリー",
                (out_dir / "garch_summary.txt").read_bytes(),
                file_name="garch_summary.txt",
                key="g_dl_sum",
            )
        except Exception as e:
            st.error(f"GARCH 推定でエラー: {type(e).__name__}: {e}")
            import traceback as _tb
            st.code(_tb.format_exc())


# ============ メイン ============
st.set_page_config(page_title="将来予測ラボ", page_icon="🔮", layout="wide")
st.title("🔮 将来予測ラボ（Forecast Lab）")
st.caption(
    "共有データまたはアップロードしたリターン系列に対して、モンテカルロ / GARCH で将来予測を行います。"
    " 入力規約は `.shared/factor_data/README.md` を参照。"
)

with st.sidebar:
    st.header("📂 共有ストレージ")
    shared = list_shared()
    if shared:
        for f in shared:
            st.caption(f"・{f.name}")
    else:
        st.caption("（空）factor-data-hub で取得してください")
    st.markdown("---")
    st.header("📁 過去の実行")
    runs = sorted((d for d in WORK_ROOT.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    st.caption(f"{len(runs)} 件")
    for d in runs[:10]:
        st.caption(f"・{d.name}")

tab1, tab2 = st.tabs(["🎲 モンテカルロ", "📈 GARCH"])
with tab1:
    render_monte_carlo()
with tab2:
    render_garch()
