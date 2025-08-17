
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from scipy import stats
from seaborn_analyzer import hist as sbhist
import warnings, os

# ============================
# ページ & サイドバー安定化
# ============================
st.set_page_config(
    page_title="統計計算ミニ v18a",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
/* サイドバー幅固定 + 常時表示（Streamlitの内部構造差異にも対応） */
section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
  min-width: 300px; max-width: 340px; display: block !important;
}
/* 本文の最大幅少し緩める */
.block-container { max-width: 1400px; }
</style>
""", unsafe_allow_html=True)

# ============================
# Matplotlib フォント（画面）
#  - 英字: DejaVu Sans（字間が安定 / "I" 詰まり対策）
#  - 日本語: IPAexGothic/Meiryo/等にフォールバック
# ============================
import matplotlib
from matplotlib import rcParams, font_manager
from matplotlib.font_manager import FontProperties

def setup_matplotlib_fonts():
    candidates = [
        os.path.join(os.path.dirname(__file__), "fonts", "ipaexg.ttf"),
        os.path.join(os.getcwd(), "fonts", "ipaexg.ttf"),
    ]
    jp_family = None
    for p in candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                jp_family = FontProperties(fname=p).get_name()
                break
            except Exception:
                pass
    if not jp_family:
        for f in ["IPAexGothic","Noto Sans CJK JP","Yu Gothic Medium","Yu Gothic","Meiryo","Hiragino Sans"]:
            try:
                matplotlib.font_manager.findfont(f, fallback_to_default=False)
                jp_family = f; break
            except Exception:
                continue
    fams = ["DejaVu Sans"]
    if jp_family: fams.append(jp_family)
    rcParams["font.family"] = fams
    rcParams["axes.unicode_minus"] = False

setup_matplotlib_fonts()

# ============================
# ReportLab（PDF）フォント
# ============================
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def setup_reportlab_font():
    candidates = [
        os.path.join(os.path.dirname(__file__), "fonts", "ipaexg.ttf"),
        os.path.join(os.getcwd(), "fonts", "ipaexg.ttf"),
        "fonts/ipaexg.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("JPEmbed", p))
                return "JPEmbed"
            except Exception:
                pass
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    return "HeiseiKakuGo-W5"

REPORTLAB_FONT = setup_reportlab_font()

def P_ja(text, size=10, bold=False, color=None):
    stl = ParagraphStyle(
        "ja",
        fontName=REPORTLAB_FONT,
        fontSize=size,
        leading=size*1.2,
    )
    if color is not None:
        stl.textColor = color
    return Paragraph(text, stl)

def P_en(text, size=10, bold=False, color=None):
    base = "Helvetica-Bold" if bold else "Helvetica"
    stl = ParagraphStyle(
        "en",
        fontName=base,
        fontSize=size,
        leading=size*1.2,
    )
    if color is not None:
        stl.textColor = color
    return Paragraph(f'<font name="{base}">{text}</font>', stl)

# ============================
# 定義・ユーティリティ
# ============================
COMPACT_COLS = ["データ数(n)","欠損数","X","u^2","u","Q1","Q2","Q3","Q4","IQR","最頻値"]
DIST_JA = {"norm":"正規","lognorm":"対数正規","gamma":"ガンマ","t":"t","cauchy":"コーシー","uniform":"一様","beta":"ベータ","bgnorm":"ベキ正規"}

def load_excel(uploaded_file, sheet_name=None):
    try:
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        sheets = xls.sheet_names
        if sheet_name is None:
            return None, sheets
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
        return df, sheets
    except Exception as e:
        st.error(f"Excelの読み込みでエラー: {e}")
        return None, []

def load_csv(uploaded_file, encoding: str):
    try:
        enc = "cp932" if encoding == "cp932 (ANSI/日本語Windows)" else encoding
        df = pd.read_csv(uploaded_file, encoding=enc)
        return df
    except Exception as e:
        st.error(f"CSVの読み込みでエラー: {e}")
        return None

def load_db(connection_url: str, sql_query: str) -> pd.DataFrame | None:
    try:
        engine: Engine = create_engine(connection_url, future=True)
        with engine.begin() as conn:
            df = pd.read_sql_query(text(sql_query), conn)
        return df
    except Exception as e:
        st.error(f"DB読み込みでエラー: {e}")
        st.info("例: SQLite: sqlite:///C:/data/my.db  /  PostgreSQL: postgresql+psycopg2://user:pass@host/db  /  MariaDB: mysql+pymysql://user:pass@host/db")
        return None

def explain_excluded_columns(df: pd.DataFrame):
    msgs = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            msgs.append(f"- {c}: 数値ではないため除外（dtype={df[c].dtype}）")
    if msgs:
        st.warning("数値でない列は除外しました:\n" + "\n".join(msgs))

def compute_stats(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for col in cols:
        s = df[col]
        n_total = len(s)
        n_missing = int(s.isna().sum())
        s_num = pd.to_numeric(s, errors="coerce")
        n_nonnull = int(s_num.notna().sum())
        mean = float(s_num.mean()) if n_nonnull > 0 else np.nan
        var_unbiased = float(s_num.var(ddof=1)) if n_nonnull > 1 else np.nan
        std_sample = float(np.sqrt(var_unbiased)) if pd.notna(var_unbiased) else np.nan
        q1 = float(s_num.quantile(0.25)) if n_nonnull > 0 else np.nan
        q2 = float(s_num.quantile(0.5)) if n_nonnull > 0 else np.nan
        q3 = float(s_num.quantile(0.75)) if n_nonnull > 0 else np.nan
        q4 = float(s_num.max()) if n_nonnull > 0 else np.nan
        iqr = float(q3 - q1) if (pd.notna(q3) and pd.notna(q1)) else np.nan
        mode_vals = s_num.mode(dropna=True)
        if len(mode_vals) == 0:
            mode_val = np.nan
        else:
            try: mode_val = float(np.nanmin(mode_vals.astype(float)))
            except Exception:
                try: mode_val = mode_vals.min()
                except Exception: mode_val = mode_vals.iloc[0]
        records.append({
            "列名": col, "データ数(n)": n_total, "欠損数": n_missing,
            "X": mean, "u^2": var_unbiased, "u": std_sample,
            "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "IQR": iqr,
            "最頻値": mode_val
        })
    raw = pd.DataFrame.from_records(records).set_index("列名")
    display = raw.copy()
    num_cols = ["X","u^2","u","Q1","Q2","Q3","Q4","IQR","最頻値"]
    display[num_cols] = display[num_cols].apply(pd.to_numeric, errors="coerce").round(4)
    return raw, display

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="stats")
    return bio.getvalue()

def _table_data_from_stats_compact(stats_df: pd.DataFrame) -> list[list[str]]:
    header = ["列名"] + COMPACT_COLS
    body = [header]
    for idx, row in stats_df[COMPACT_COLS].iterrows():
        rec = [str(idx)]
        rec.append(f"{int(row['データ数(n)'])}" if pd.notna(row['データ数(n)']) else "")
        rec.append(f"{int(row['欠損数'])}" if pd.notna(row['欠損数']) else "")
        for c in ["X","u^2","u","Q1","Q2","Q3","Q4","IQR","最頻値"]:
            v = row[c]; rec.append("" if pd.isna(v) else f"{float(v):.4f}")
        body.append(rec)
    return body

# ===== 分布フィット =====
def fit_one_column_for_pdf(df_num: pd.DataFrame, col: str, bins, norm_hist: bool, dist_list_ui: list[str]):
    _distmap = {"norm": stats.norm, "lognorm": stats.lognorm, "gamma": stats.gamma,
                "t": stats.t, "cauchy": stats.cauchy, "uniform": stats.uniform, "beta": stats.beta}
    s_raw = pd.to_numeric(df_num[col], errors="coerce").dropna()
    use_beta  = "beta" in dist_list_ui
    use_logn  = "lognorm" in dist_list_ui
    use_gamma = "gamma" in dist_list_ui
    beta_ok = bool((s_raw.gt(0) & s_raw.lt(1)).all())
    positive_ok = bool((s_raw > 0).all())
    dist_list = dist_list_ui[:]; notes = []; df_fit = df_num; x_col  = col
    if use_beta and not beta_ok:
        dist_list = [d for d in dist_list if d != "beta"]; notes.append("beta: (0,1)外で除外")
    if (use_logn or use_gamma) and not positive_ok:
        dist_list = [d for d in dist_list if d not in ["lognorm","gamma"]]; notes.append("lognorm/gamma: 値>0でないため除外")
    dist_objs = [_distmap[d] for d in dist_list if d in _distmap]
    fig = plt.figure(figsize=(9, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_params, all_scores = sbhist.fit_dist(df_fit, x=x_col, dist=dist_objs, bins=bins, norm_hist=norm_hist)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight"); plt.close(fig)
    df_scores = None
    if all_scores:
        df_scores = pd.DataFrame(all_scores).T.sort_values("AIC").head(12)
    return {"column_name": col, "figure_png": buf.getvalue(), "scores_df": df_scores, "notes": " / ".join(notes)}

# ===== PDF レポート =====
def build_stats_pdf(app_title: str, stats_display_df: pd.DataFrame, source_note: str, original_df: pd.DataFrame, selected_cols: list[str], fit_results: list[dict] | None = None) -> bytes:
    pdf_buf = BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    INNER_W = doc.width
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleJP", parent=styles["Title"], fontName=REPORTLAB_FONT, fontSize=22, leading=28, spaceAfter=6*mm)
    h2 = ParagraphStyle("H2JP", parent=styles["Heading2"], fontName=REPORTLAB_FONT, leading=18, spaceAfter=4*mm)
    h3 = ParagraphStyle("H3JP", parent=styles["Heading3"], fontName=REPORTLAB_FONT, leading=16, spaceAfter=3*mm)
    normal = ParagraphStyle("NormJP", parent=styles["Normal"], fontName=REPORTLAB_FONT, leading=14)
    italic = ParagraphStyle("ItalicJP", parent=styles["Italic"], fontName=REPORTLAB_FONT, leading=14)
    story = []
    story.append(Paragraph('統計計算ミニ (<font name="Helvetica-Bold">Basic Stats Inspector</font>)', title_style))
    story.append(Paragraph("基本統計量レポート", h2))
    story.append(Paragraph(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), normal))
    if source_note: story.append(Paragraph(f"データソース：{source_note}", italic))
    story.append(Spacer(1, 6*mm))

    # ■ 表（コンパクト & 幅フィット）
    body = _table_data_from_stats_compact(stats_display_df)
    header = body[0]
    new_header = []
    for h in header:
        if h in ["X","u^2","u","Q1","Q2","Q3","Q4","IQR","AIC","BIC","RSS"]:
            new_header.append(P_en(h, size=9, bold=True))
        else:
            new_header.append(P_ja(h, size=9, bold=True))
    body[0] = new_header

    ncols = len(body[0])
    col_w = INNER_W / ncols
    table = Table(body, repeatRows=1, colWidths=[col_w]*ncols)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#E6EEF8")),
        ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONTNAME",(0,1),(-1,-1), REPORTLAB_FONT),
        ("FONTSIZE",(0,0),(-1,-1), 6.0),
        ("ALIGN",(1,1),(-1,-1), "RIGHT"),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),2),
        ("RIGHTPADDING",(0,0),(-1,-1),2),
        ("TOPPADDING",(0,0),(-1,-1),1),
        ("BOTTOMPADDING",(0,0),(-1,-1),1),
    ]))
    story.append(Paragraph("■ 基本統計量（一覧・コンパクト）", h3)); story.append(table); story.append(Spacer(1,6*mm))

    # ■ 図（横並び）
    SAFE_GAP = 6*mm; IMG_W = (INNER_W - SAFE_GAP) / 2; IMG_H = IMG_W * 0.70
    for col in selected_cols:
        story.append(Paragraph(f"■ グラフ：{col}", h3))
        s = pd.to_numeric(original_df[col], errors="coerce").dropna()
        fig1, ax1 = plt.subplots(figsize=(5.0, 3.0), dpi=200)
        ax1.hist(s, bins="auto"); ax1.set_title("ヒストグラム"); ax1.set_xlabel("値"); ax1.set_ylabel("度数")
        b1 = BytesIO(); fig1.tight_layout(); fig1.savefig(b1, format="png"); plt.close(fig1)
        fig2, ax2 = plt.subplots(figsize=(5.0, 3.0), dpi=200)
        stats.probplot(s, dist="norm", plot=ax2); ax2.set_title("正規性 Q–Q プロット")
        b2 = BytesIO(); fig2.tight_layout(); fig2.savefig(b2, format="png"); plt.close(fig2)
        row_tbl = Table([[Image(BytesIO(b1.getvalue()), width=IMG_W, height=IMG_H, kind="proportional"),
                          Image(BytesIO(b2.getvalue()), width=IMG_W, height=IMG_H, kind="proportional")]],
                        colWidths=[IMG_W, IMG_W], hAlign="LEFT")
        row_tbl.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
        story.append(row_tbl); story.append(Spacer(1,6*mm))

    # ■ 分布フィット＋総括
    if fit_results:
        story.append(Paragraph("■ 分布フィット（各列の重ね描画とスコア）", h3)); story.append(Spacer(1,2*mm))
        FIT_W = INNER_W - 6*mm; FIT_H = FIT_W * 0.44
        for fr in fit_results:
            col_name = fr.get("column_name",""); story.append(Paragraph(f"【対象列】{col_name}", h3))
            if fr.get("figure_png"):
                story.append(Image(BytesIO(fr["figure_png"]), width=FIT_W, height=FIT_H, kind="proportional")); story.append(Spacer(1,2*mm))
            notes = fr.get("notes") or ""
            if notes: story.append(Paragraph(f"備考：{notes}", italic)); story.append(Spacer(1,1*mm))
            if fr.get("scores_df") is not None:
                sc = fr["scores_df"].copy().reset_index().rename(columns={"index":"分布"})
                sc["分布"] = sc["分布"].map(lambda x: DIST_JA.get(str(x), str(x)))
                sc = sc.rename(columns={"RSS":"RSS","AIC":"AIC","BIC":"BIC"})
                cols = list(sc.columns)
                head_cells = [P_ja("分布", size=8, bold=True)]
                for c in cols[1:]:
                    head_cells.append(P_en(c, size=8, bold=True))
                tdata = [head_cells] + sc.values.tolist()

                t = Table(tdata, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#F0F4FA")),
                    ("GRID",(0,0),(-1,-1), 0.3, colors.grey),
                    ("FONTNAME",(0,1),(-1,-1), REPORTLAB_FONT),
                    ("FONTSIZE",(0,0),(-1,-1), 8),
                    ("ALIGN",(1,1),(-1,-1), "RIGHT"),
                    ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
                ]))
                story.append(t); story.append(Spacer(1,5*mm))

        story.append(Paragraph('■ 総括：列ごとの最良分布（<font name="Helvetica-Bold">AIC</font>最小）', h3)); story.append(Spacer(1,2*mm))
        rows = [[P_ja("列名", size=8, bold=True), P_ja("最良分布", size=8, bold=True), P_en("AIC", size=8, bold=True), P_en("BIC", size=8, bold=True), P_en("RSS", size=8, bold=True), P_ja("備考", size=8, bold=True)]]
        for fr in fit_results:
            name = fr.get("column_name",""); df_sc = fr.get("scores_df"); note = fr.get("notes","") or ""
            if df_sc is None or df_sc.empty or ("AIC" not in df_sc.columns):
                rows.append([name,"—","—","—","—",note])
            else:
                best_idx = df_sc["AIC"].idxmin(); best_row = df_sc.loc[best_idx]
                best_name = DIST_JA.get(str(best_idx), str(best_idx))
                aic = f"{best_row['AIC']:.4f}" if pd.notna(best_row.get("AIC")) else "—"
                bic = f"{best_row['BIC']:.4f}" if "BIC" in df_sc.columns and pd.notna(best_row.get("BIC")) else "—"
                rss = f"{best_row['RSS']:.4f}" if "RSS" in df_sc.columns and pd.notna(best_row.get("RSS")) else "—"
                rows.append([name, best_name, aic, bic, rss, note])
        summary_tbl = Table(rows, repeatRows=1)
        summary_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#EDEFF5")),
            ("GRID",(0,0),(-1,-1), 0.3, colors.grey),
            ("FONTNAME",(0,0),(-1,-1), REPORTLAB_FONT),
            ("FONTSIZE",(0,0),(-1,-1), 8),
            ("ALIGN",(2,1),(-2,-1), "RIGHT"),
            ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
            ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
        ]))
        story.append(summary_tbl)

    doc.build(story)
    return pdf_buf.getvalue()

# ============================
# UI：サイドバー
# ============================
with st.sidebar:
    st.header("データソース")
    source = st.radio("選択", ["Excelから読み込む","CSVから読み込む","DBから読み込む"])
    st.caption("※ サイドバーは常時展開・幅固定です")

df = None
if source == "Excelから読み込む":
    uploaded = st.file_uploader("Excelファイル（.xlsx）を選択", type=["xlsx"])
    if uploaded:
        df_dummy, sheets = load_excel(uploaded)
        if sheets:
            sheet = st.selectbox("シートを選択", sheets)
            df, _ = load_excel(uploaded, sheet)
            if df is not None:
                st.success(f"読み込み完了：{sheet}（{len(df)} 行 × {len(df.columns)} 列）")
elif source == "CSVから読み込む":
    uploaded = st.file_uploader("CSVファイル（.csv）を選択", type=["csv"])
    enc = st.selectbox("エンコーディング", ["utf-8","shift_jis","cp932 (ANSI/日本語Windows)"], index=0)
    if uploaded:
        df = load_csv(uploaded, enc)
        if df is not None:
            st.success(f"読み込み完了：{len(df)} 行 × {len(df.columns)} 列")
else:
    st.write("**DB読み込み**（接続URLとSQLを指定）")
    with st.expander("接続URLの例", expanded=False):
        st.markdown("""
- SQLite: `sqlite:///C:/data/my.db`
- PostgreSQL: `postgresql+psycopg2://user:pass@host:5432/dbname`
- MariaDB/MySQL: `mysql+pymysql://user:pass@host:3306/dbname`
""")
    conn_url = st.text_input("接続URL")
    sql = st.text_area("SQL（例: SELECT * FROM your_table LIMIT 10000）", height=120)
    if st.button("DBから読み込む"):
        if conn_url.strip() and sql.strip():
            df = load_db(conn_url.strip(), sql.strip())

# ============================
# 本体：統計 → 可視化 → フィット → PDF
# ============================
if df is not None:
    num_df = df.select_dtypes(include="number")
    non_num_cols = [c for c in df.columns if c not in num_df.columns]
    if len(non_num_cols) > 0: explain_excluded_columns(df)
    if num_df.empty:
        st.warning("数値列がありません。"); st.stop()

    default_cols = list(num_df.columns)[:min(5, len(num_df.columns))]
    selected_cols = st.multiselect("統計を計算する列を選んでください（複数可）", list(num_df.columns), default=default_cols)

    if selected_cols:
        raw_stats, show_stats = compute_stats(num_df, selected_cols)
        st.subheader("基本統計量（列 × 指標）")
        st.dataframe(show_stats, use_container_width=True)
        st.download_button("CSVでダウンロード", data=raw_stats.to_csv(index=True), file_name="basic_stats.csv", mime="text/csv")
        st.download_button("Excelでダウンロード", data=to_excel_bytes(raw_stats), file_name="basic_stats.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("---")
        st.subheader("可視化と分布フィット（seaborn-analyzer）")
        viz_col = st.selectbox("対象列を選択", selected_cols, index=0)
        bins_mode = st.radio("ビン数", ["自動","手動"], horizontal=True)
        bins = "auto" if bins_mode == "自動" else st.slider("ビン数（手動）", 5, 200, 30, 1)
        norm_hist = st.checkbox("縦軸を確率密度（面積=1）にする", value=True)

        tab1, tab2 = st.tabs(["ヒスト+QQ（横並び）", "分布フィット（グラフ+表）"])
        with tab1:
            s = pd.to_numeric(num_df[viz_col], errors="coerce").dropna()
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                fig1, ax1 = plt.subplots(figsize=(5,3))
                ax1.hist(s, bins=bins if bins != "auto" else "auto", density=norm_hist)
                ax1.set_title("ヒストグラム"); ax1.set_xlabel("値"); ax1.set_ylabel("度数" if not norm_hist else "確率密度")
                st.pyplot(fig1, clear_figure=True)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(5,3))
                stats.probplot(s, dist="norm", plot=ax2)
                ax2.set_title("正規性 Q–Q プロット")
                st.pyplot(fig2, clear_figure=True)

        with tab2:
            st.caption("候補分布を重ね描画し、同じタブでスコア表も表示します。")
            dist_list = st.multiselect("分布（複数可）", ["norm","lognorm","gamma","uniform","t","cauchy","beta"], default=["norm","lognorm","gamma"])
            _distmap = {"norm": stats.norm, "lognorm": stats.lognorm, "gamma": stats.gamma, "t": stats.t, "cauchy": stats.cauchy, "uniform": stats.uniform, "beta": stats.beta}
            s_raw = pd.to_numeric(num_df[viz_col], errors="coerce").dropna()
            use_beta   = "beta" in dist_list
            use_logn   = "lognorm" in dist_list
            use_gamma  = "gamma" in dist_list
            beta_ok = bool((s_raw.gt(0) & s_raw.lt(1)).all())
            beta_scale = False
            if use_beta and not beta_ok:
                beta_scale = st.checkbox("β分布用に 0–1 にスケーリングして当てはめる", value=False)
                if not beta_scale:
                    st.info("β分布は (0,1) の範囲が前提のため除外しました。"); dist_list = [d for d in dist_list if d != "beta"]
            positive_ok = bool((s_raw > 0).all())
            shift_positive = False
            if (use_logn or use_gamma) and not positive_ok:
                shift_positive = st.checkbox("lognorm/gamma 用に正の領域へシフトして当てはめる", value=False)
                if not shift_positive:
                    msg = []; 
                    if use_logn: msg.append("lognorm")
                    if use_gamma: msg.append("gamma")
                    st.info(f"{' / '.join(msg)} は値>0が前提のため除外しました。")
                    dist_list = [d for d in dist_list if not (d in ['lognorm','gamma'])]
            df_fit = num_df; x_col = viz_col; s_fit = s_raw.copy()
            if use_beta and not beta_ok and beta_scale:
                rng = s_fit.max() - s_fit.min(); s_scaled = (s_fit - s_fit.min()) / rng if rng > 0 else s_fit*0 + 0.5
                df_fit = num_df.copy(); x_col = f"{viz_col}__scaled01"; df_fit[x_col] = pd.Series(s_scaled, index=s_fit.index)
                st.caption("※ β分布は 0–1 スケーリング後の値で当てはめ。")
            if (use_logn or use_gamma) and not positive_ok and shift_positive:
                eps = 1e-9; shift = -(s_fit.min()) + eps; s_shift = s_fit + shift
                df_fit = num_df.copy(); x_col = f"{viz_col}__shift_pos"; df_fit[x_col] = pd.Series(s_shift, index=s_fit.index)
                st.caption("※ lognorm/gamma は正の領域へシフト後の値で当てはめ。")
            dist_objs = [_distmap[d] for d in dist_list if d in _distmap]
            fig = plt.figure(figsize=(9, 4))
            with warnings.catch_warnings(): warnings.simplefilter("ignore", RuntimeWarning)
            all_params, all_scores = sbhist.fit_dist(df_fit, x=x_col, dist=dist_objs, bins=bins, norm_hist=norm_hist)
            st.pyplot(fig, clear_figure=True)
            df_scores = None
            if all_scores:
                df_scores = pd.DataFrame(all_scores).T
                st.dataframe(df_scores.sort_values("AIC"), use_container_width=True)
            else:
                st.info("スコアが取得できませんでした。")
            buf = BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.session_state["fit_result"] = {
                "column_name": x_col, "figure_png": buf.getvalue(),
                "scores_df": df_scores.sort_values("AIC").head(12) if df_scores is not None else None,
                "notes": ("β:0–1スケーリング適用" if (use_beta and not beta_ok and beta_scale) else "") + (" / shift>0適用" if ((use_logn or use_gamma) and not positive_ok and shift_positive) else "")
            }
            st.divider()
            include_all_fits = st.checkbox("PDFに選択列すべての分布フィットを含める", value=True)
            if include_all_fits:
                fit_results_for_pdf = []
                current_dists = dist_list
                for c in selected_cols:
                    fr = fit_one_column_for_pdf(num_df, c, bins=bins, norm_hist=norm_hist, dist_list_ui=current_dists)
                    fit_results_for_pdf.append(fr)
                st.session_state["fit_results_all"] = fit_results_for_pdf
            else:
                st.session_state["fit_results_all"] = [st.session_state.get("fit_result")] if st.session_state.get("fit_result") else None

        st.markdown("---")
        st.subheader("📄 基本統計量レポート（PDF）")
        source_note = "Excel/CSV/DBのいずれかから読み込み"
        pdf_bytes = build_stats_pdf("統計計算ミニ（Basic Stats Inspector）", show_stats, source_note, num_df, selected_cols, st.session_state.get("fit_results_all"))
        st.download_button("PDFをダウンロード（分布フィット＋総括付き）", data=pdf_bytes, file_name="basic_stats_report_v18a.pdf", mime="application/pdf")
    else:
        st.info("列を選択してください。")
else:
    st.info("左のサイドバーからデータソースを選び、読み込みを行ってください。")
