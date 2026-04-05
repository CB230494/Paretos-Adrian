# ============================================================================
# ============================== PARTE 1/12 =================================
# ============ Encabezado, imports, config y estilos Matplotlib =============
# ============================================================================

# App — Pareto 80/20 + Portafolio + Unificado + Sheets + Informe PDF + MIC MAC
# Ejecuta: streamlit run app.py

import io
from pathlib import Path
from textwrap import wrap
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ====== Google Sheets (DB) ======
import gspread
from google.oauth2.service_account import Credentials

# ====== PDF (ReportLab/Platypus) ======
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    PageBreak, NextPageTemplate
)
from reportlab.platypus.flowables import KeepTogether

# ----------------- CONFIG (Google Sheets) -----------------
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1aIyU2hPQHCUgvd-Jtp4pV6sxsqff5uouysdjjpKX8Qg/edit?gid=0#gid=0"

WS_PARETOS = "Paretos"

# ----------------- CONFIG CATÁLOGO EXTERNO -----------------
CATALOGO_XLSX = "catalogo_descriptores.xlsx"
CATALOGO_HOJA = "catalogo"

st.set_page_config(page_title="Pareto de Descriptores", layout="wide")

# Paleta
VERDE = "#1B9E77"
AZUL = "#2C7FB8"
TEXTO = "#124559"
GRIS = "#6B7280"
ROJO = "#CC3D3D"
NARANJA = "#E67E22"

# Matplotlib
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "axes.titlesize": 18,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ============================================================================
# ============================== PARTE 2/12 =================================
# =========================== Catálogo desde Excel ===========================
# ============================================================================

def cargar_catalogo_desde_excel(
    path: str = CATALOGO_XLSX,
    sheet_name: str = CATALOGO_HOJA
) -> List[Dict[str, str]]:
    """
    Carga el catálogo desde un Excel ubicado en la raíz del proyecto.
    Requiere columnas: categoria, descriptor
    """
    try:
        p = Path(path)
        if not p.exists():
            st.error(
                f"No se encontró el archivo '{path}' en la raíz del proyecto. "
                f"Debes subirlo al repositorio junto a app.py."
            )
            return []

        df = pd.read_excel(p, sheet_name=sheet_name)
        df.columns = [str(c).strip().lower() for c in df.columns]

        required = {"categoria", "descriptor"}
        if not required.issubset(set(df.columns)):
            st.error(
                f"El archivo '{path}' debe tener exactamente las columnas: "
                f"{', '.join(sorted(required))}"
            )
            return []

        df = df[["categoria", "descriptor"]].copy()
        df["categoria"] = df["categoria"].astype(str).str.strip()
        df["descriptor"] = df["descriptor"].astype(str).str.strip()

        df = df[
            (df["categoria"] != "") &
            (df["descriptor"] != "") &
            (df["categoria"].str.lower() != "nan") &
            (df["descriptor"].str.lower() != "nan")
        ].drop_duplicates(subset=["descriptor"], keep="first")

        return df.to_dict(orient="records")

    except Exception as e:
        st.error(f"Error cargando catálogo desde Excel: {e}")
        return []


CATALOGO: List[Dict[str, str]] = cargar_catalogo_desde_excel()


# ============================================================================
# ============================== PARTE 3/12 =================================
# ====================== Utilidades base y cálculo Pareto ====================
# ============================================================================

def _map_descriptor_a_categoria() -> Dict[str, str]:
    if not CATALOGO:
        return {}
    df = pd.DataFrame(CATALOGO)
    return dict(zip(df["descriptor"], df["categoria"]))


DESC2CAT = _map_descriptor_a_categoria()


def normalizar_freq_map(freq_map: Dict[str, int]) -> Dict[str, int]:
    out = {}
    for d, v in (freq_map or {}).items():
        try:
            dd = str(d).strip()
            vv = int(pd.to_numeric(v, errors="coerce"))
            if dd and vv > 0:
                out[dd] = vv
        except Exception:
            continue
    return out


def df_desde_freq_map(freq_map: Dict[str, int]) -> pd.DataFrame:
    items = []
    for d, f in normalizar_freq_map(freq_map).items():
        items.append({
            "descriptor": d,
            "categoria": DESC2CAT.get(d, "Manual / sin categoría"),
            "frecuencia": int(f)
        })
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=["descriptor", "categoria", "frecuencia"])
    return df


def combinar_maps(maps: List[Dict[str, int]]) -> Dict[str, int]:
    total = {}
    for m in maps:
        for d, f in normalizar_freq_map(m).items():
            total[d] = total.get(d, 0) + int(f)
    return total


def info_pareto(freq_map: Dict[str, int]) -> Dict[str, int]:
    d = normalizar_freq_map(freq_map)
    return {"descriptores": len(d), "total": int(sum(d.values()))}


def calcular_pareto(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "descriptor", "categoria", "frecuencia", "porcentaje",
            "acumulado", "pct_acum", "segmento_real", "segmento"
        ])

    df["frecuencia"] = pd.to_numeric(df["frecuencia"], errors="coerce").fillna(0).astype(int)
    df = df[df["frecuencia"] > 0]
    if df.empty:
        return pd.DataFrame(columns=[
            "descriptor", "categoria", "frecuencia", "porcentaje",
            "acumulado", "pct_acum", "segmento_real", "segmento"
        ])

    df = df.sort_values(["frecuencia", "descriptor"], ascending=[False, True]).reset_index(drop=True)
    total = int(df["frecuencia"].sum())

    df["porcentaje"] = (df["frecuencia"] / total * 100).round(2)
    df["acumulado"] = df["frecuencia"].cumsum()
    df["pct_acum"] = (df["acumulado"] / total * 100).round(2)

    # tramo clásico visual
    df["segmento_real"] = np.where(df["pct_acum"] <= 80.00, "80%", "20%")

    # tramo priorizado estricto:
    # SOLO lo que esté en 80.00% o menos
    df["segmento"] = np.where(df["pct_acum"] <= 80.00, "PRIORIZADO", "NO PRIORIZADO")

    # seguridad: si por algún caso extremo ninguno quedó <=80,
    # al menos deja el primero priorizado
    if not df.empty and (df["segmento"] == "PRIORIZADO").sum() == 0:
        df.loc[0, "segmento"] = "PRIORIZADO"

    return df.reset_index(drop=True)


def obtener_df_priorizado(df_par: pd.DataFrame) -> pd.DataFrame:
    if df_par.empty:
        return df_par.copy()
    df = df_par.copy()
    out = df[df["segmento"] == "PRIORIZADO"].copy()
    if out.empty and not df.empty:
        out = df.head(1).copy()
    return out.reset_index(drop=True)


def _colors_for_segments(segments: List[str]) -> List[str]:
    out = []
    for s in segments:
        if s in ("80%", "PRIORIZADO"):
            out.append(VERDE)
        else:
            out.append(AZUL)
    return out


def _wrap_labels(labels: List[str], width: int = 22) -> List[str]:
    if len(labels) > 30:
        width = 15
    elif len(labels) > 20:
        width = 18
    elif len(labels) > 12:
        width = 20
    return ["\n".join(wrap(str(t), width=width)) for t in labels]


def _wrap_for_two_lines(labels: List[str]) -> List[str]:
    if not labels:
        return labels
    max_len = max(len(str(x)) for x in labels)
    if max_len > 60:
        w = 25
    elif max_len > 40:
        w = 20
    elif max_len > 25:
        w = 16
    else:
        w = 14

    wrapped = []
    for label in labels:
        txt = "\n".join(wrap(str(label), width=w))
        parts = txt.split("\n")
        if len(parts) > 2:
            txt = "\n".join(parts[:2])
        wrapped.append(txt)
    return wrapped


# ============================================================================
# ============================== PARTE 4/12 =================================
# ===================== Pareto UI / PNG / Excel / Descargas ==================
# ============================================================================

def dibujar_pareto(df_par: pd.DataFrame, titulo: str):
    if df_par.empty:
        st.info("Ingresa frecuencias (>0) para ver el gráfico.")
        return

    n_labels = len(df_par)
    x = np.arange(n_labels)
    freqs = df_par["frecuencia"].to_numpy()
    pct_acum = df_par["pct_acum"].to_numpy()
    colors_b = _colors_for_segments(df_par["segmento"].tolist())
    labels = [str(t) for t in df_par["descriptor"].tolist()]
    labels_w = _wrap_for_two_lines(labels)

    fig_w = max(12.0, 0.60 * n_labels)
    fs = 9 if n_labels > 28 else 10

    fig, ax1 = plt.subplots(figsize=(fig_w, 6.6))
    ax1.bar(x, freqs, color=colors_b)
    ax1.set_ylabel("Frecuencia")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_w, rotation=90, ha="center", va="top", fontsize=fs)
    fig.subplots_adjust(bottom=0.30)

    ax1.set_title(titulo if titulo.strip() else "Diagrama de Pareto", color=TEXTO, fontsize=16)

    ax2 = ax1.twinx()
    ax2.plot(x, pct_acum, marker="o", linewidth=2, color=TEXTO)
    ax2.set_ylabel("% acumulado")
    ax2.set_ylim(0, 110)

    df_pri = obtener_df_priorizado(df_par)
    if not df_pri.empty:
        cut_idx = len(df_pri) - 1
        ax1.axvline(cut_idx + 0.5, linestyle=":", color="k")

    ax2.axhline(80, linestyle="--", linewidth=1, color="#666666")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _pareto_png(df_par: pd.DataFrame, titulo: str, solo_priorizados: bool = False) -> bytes:
    """
    PNG del Pareto para PDF o descarga.
    """
    df_base = obtener_df_priorizado(df_par) if solo_priorizados else df_par.copy()
    if df_base.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
        ax.axis("off")
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return buf.getvalue()

    n_labels = len(df_base)
    labels = [str(t) for t in df_base["descriptor"].tolist()]
    x = np.arange(n_labels)
    freqs = df_base["frecuencia"].to_numpy()
    pct_acum = df_base["pct_acum"].to_numpy()
    colors_b = _colors_for_segments(df_base["segmento"].tolist())

    fig_w = max(12.0, 0.85 * n_labels)
    fig_h = 6.6

    if n_labels > 40:
        fs = 7
    elif n_labels > 28:
        fs = 8
    else:
        fs = 9

    if n_labels > 40:
        rot = 90
    elif n_labels > 25:
        rot = 70
    else:
        rot = 45

    dpi = 220

    fig, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax1.bar(x, freqs, color=colors_b, zorder=2)
    ax1.set_ylabel("Frecuencia")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=rot, ha="right", va="top", fontsize=fs)
    ax1.set_title(titulo if titulo.strip() else "Diagrama de Pareto", color=TEXTO, fontsize=16)

    ax2 = ax1.twinx()
    ax2.plot(x, pct_acum, marker="o", linewidth=2, color=TEXTO, zorder=3)
    ax2.set_ylabel("% acumulado")
    ax2.set_ylim(0, 110)

    df_pri = obtener_df_priorizado(df_base)
    if not df_pri.empty:
        cut_idx = len(df_pri) - 1
        ax1.axvline(cut_idx + 0.5, linestyle=":", color="k")

    ax2.axhline(80, linestyle="--", linewidth=1, color="#666666")
    ax1.grid(True, axis="y", alpha=0.25, zorder=1)
    fig.subplots_adjust(bottom=0.35)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


def exportar_excel_con_grafico(df_par: pd.DataFrame, titulo: str) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        hoja = "Pareto"
        df_x = df_par.copy()
        if df_x.empty:
            pd.DataFrame(columns=[
                "categoria", "descriptor", "frecuencia", "porcentaje",
                "pct_acum", "acumulado", "segmento"
            ]).to_excel(writer, sheet_name=hoja, index=False)
            return output.getvalue()

        df_x["porcentaje"] = (df_x["porcentaje"] / 100.0).round(4)
        df_x["pct_acum"] = (df_x["pct_acum"] / 100.0).round(4)
        df_x = df_x[[
            "categoria", "descriptor", "frecuencia",
            "porcentaje", "pct_acum", "acumulado", "segmento"
        ]]
        df_x.to_excel(writer, sheet_name=hoja, index=False, startrow=0, startcol=0)

        wb = writer.book
        ws = writer.sheets[hoja]
        pct_fmt = wb.add_format({"num_format": "0.00%"})
        total_fmt = wb.add_format({"bold": True})

        ws.set_column("A:A", 22)
        ws.set_column("B:B", 55)
        ws.set_column("C:C", 12)
        ws.set_column("D:D", 12, pct_fmt)
        ws.set_column("E:E", 18, pct_fmt)
        ws.set_column("F:F", 12)
        ws.set_column("G:G", 16)

        n = len(df_x)
        cats = f"=Pareto!$B$2:$B${n+1}"
        vals = f"=Pareto!$C$2:$C${n+1}"
        pcts = f"=Pareto!$E$2:$E${n+1}"
        total = int(df_par["frecuencia"].sum()) if not df_par.empty else 0

        ws.write(n + 2, 1, "TOTAL:", total_fmt)
        ws.write(n + 2, 2, total, total_fmt)

        chart = wb.add_chart({"type": "column"})
        points = [{"fill": {"color": (VERDE if s == "PRIORIZADO" else AZUL)}} for s in df_par["segmento"]]
        chart.add_series({
            "name": "Frecuencia",
            "categories": cats,
            "values": vals,
            "points": points
        })

        line = wb.add_chart({"type": "line"})
        line.add_series({
            "name": "% acumulado",
            "categories": cats,
            "values": pcts,
            "y2_axis": True,
            "marker": {"type": "circle"}
        })

        chart.combine(line)
        chart.set_y_axis({"name": "Frecuencia"})
        chart.set_y2_axis({
            "name": "Porcentaje acumulado",
            "min": 0, "max": 1.10,
            "major_unit": 0.10,
            "num_format": "0%"
        })

        chart.set_title({"name": titulo.strip() or "Diagrama de Pareto"})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 1180, "height": 420})
        ws.insert_chart("I2", chart)
    return output.getvalue()


# ============================================================================
# ============================== PARTE 5/12 =================================
# ======================== Conectores Google Sheets ==========================
# ============================================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


def _gc():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    return gspread.authorize(creds)


def _open_sheet():
    gc = _gc()
    return gc.open_by_url(SPREADSHEET_URL)


def _ensure_ws(sh, title: str, header: List[str]):
    try:
        ws = sh.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(10, len(header)))
        ws.append_row(header)
        return ws

    values = ws.get_all_values()
    if not values:
        ws.append_row(header)
    else:
        first = values[0]
        if [c.strip().lower() for c in first] != [c.strip().lower() for c in header]:
            ws.clear()
            ws.append_row(header)
    return ws


def sheets_cargar_portafolio() -> Dict[str, Dict[str, int]]:
    """
    Lee 'paretos' (nombre, descriptor, frecuencia). Ignora vacíos y <=0.
    """
    try:
        sh = _open_sheet()
        ws = _ensure_ws(sh, WS_PARETOS, ["nombre", "descriptor", "frecuencia"])
        rows = ws.get_all_records()
        port: Dict[str, Dict[str, int]] = {}
        for r in rows:
            nom = str(r.get("nombre", "")).strip()
            desc = str(r.get("descriptor", "")).strip()
            freq = int(pd.to_numeric(r.get("frecuencia", 0), errors="coerce") or 0)
            if not nom or not desc or freq <= 0:
                continue
            bucket = port.setdefault(nom, {})
            bucket[desc] = bucket.get(desc, 0) + freq
        return port
    except Exception:
        return {}


def sheets_guardar_pareto(nombre: str, freq_map: Dict[str, int], sobrescribir: bool = True):
    sh = _open_sheet()
    ws = _ensure_ws(sh, WS_PARETOS, ["nombre", "descriptor", "frecuencia"])

    if sobrescribir:
        vals = ws.get_all_values()
        header = vals[0] if vals else ["nombre", "descriptor", "frecuencia"]
        others = [r for r in vals[1:] if (len(r) > 0 and r[0].strip().lower() != nombre.strip().lower())]
        ws.clear()
        ws.update("A1", [header])
        if others:
            ws.append_rows(others, value_input_option="RAW")

    rows_new = [[nombre, d, int(f)] for d, f in normalizar_freq_map(freq_map).items()]
    if rows_new:
        ws.append_rows(rows_new, value_input_option="RAW")


def sheets_eliminar_pareto(nombre: str) -> bool:
    try:
        sh = _open_sheet()
        ws = _ensure_ws(sh, WS_PARETOS, ["nombre", "descriptor", "frecuencia"])
        vals = ws.get_all_values()
        if not vals or len(vals) <= 1:
            return False

        header = vals[0]
        others = [r for r in vals[1:] if (len(r) > 0 and r[0].strip().lower() != nombre.strip().lower())]
        ws.clear()
        ws.update("A1", [header])
        if others:
            ws.append_rows(others, value_input_option="RAW")
        return True
    except Exception as e:
        st.warning(f"No se pudo eliminar '{nombre}' de Google Sheets: {e}")
        return False
# ============================================================================
# ============================== PARTE 6/12 =================================
# =================== Estado de sesión + Estilos básicos PDF =================
# ============================================================================

st.session_state.setdefault("freq_map", {})
st.session_state.setdefault("portafolio", {})
st.session_state.setdefault("msel", [])
st.session_state.setdefault("editor_df", pd.DataFrame(columns=["descriptor", "frecuencia"]))
st.session_state.setdefault("last_msel", [])
st.session_state.setdefault("reset_after_save", False)
st.session_state.setdefault("manual_rows", pd.DataFrame(columns=["descriptor", "frecuencia"]))
st.session_state.setdefault("sheet_url_loaded", None)

# 🔥 IMPORTANTE: control de cambio de hoja (URL)
if st.session_state["sheet_url_loaded"] != SPREADSHEET_URL:
    st.session_state["portafolio"] = {}
    st.session_state["sheet_url_loaded"] = SPREADSHEET_URL

# 🔄 Carga inicial desde Google Sheets
if not st.session_state["portafolio"]:
    loaded = sheets_cargar_portafolio()
    if loaded:
        st.session_state["portafolio"].update(loaded)

# 🔄 Reset después de guardar
if st.session_state.get("reset_after_save", False):
    st.session_state["freq_map"] = {}
    st.session_state["msel"] = []
    st.session_state["editor_df"] = pd.DataFrame(columns=["descriptor", "frecuencia"])
    st.session_state["manual_rows"] = pd.DataFrame(columns=["descriptor", "frecuencia"])
    st.session_state["last_msel"] = []
    st.session_state.pop("editor_freq", None)
    st.session_state.pop("manual_editor", None)
    st.session_state["reset_after_save"] = False

PAGE_W, PAGE_H = A4


def _styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        name="CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#0B3954"),
        alignment=1,
        spaceAfter=10
    ))

    ss.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=ss["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1F5F8B"),
        alignment=1,
        spaceAfter=8
    ))

    ss.add(ParagraphStyle(
        name="CoverDate",
        parent=ss["Normal"],
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#4B5563"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=6
    ))

    ss.add(ParagraphStyle(
        name="TitleBig",
        parent=ss["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#0B3954"),
        alignment=0,
        spaceAfter=10
    ))

    ss.add(ParagraphStyle(
        name="TitleBigCenter",
        parent=ss["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#0B3954"),
        alignment=1,
        spaceAfter=10
    ))

    ss.add(ParagraphStyle(
        name="H1",
        parent=ss["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#124559"),
        spaceAfter=8
    ))

    ss.add(ParagraphStyle(
        name="H1Center",
        parent=ss["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#124559"),
        spaceAfter=8,
        alignment=1
    ))

    ss.add(ParagraphStyle(
        name="Body",
        parent=ss["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#1F2937"),
        alignment=4
    ))

    ss.add(ParagraphStyle(
        name="Small",
        parent=ss["Normal"],
        fontSize=9.6,
        leading=12.5,
        textColor=colors.HexColor("#6B7280")
    ))

    ss.add(ParagraphStyle(
        name="TableHead",
        parent=ss["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=12,
        textColor=colors.white,
        alignment=1
    ))

    ss.add(ParagraphStyle(
        name="BulletList",
        parent=ss["Body"],
        leftIndent=14,
        bulletIndent=2,
        spaceBefore=2,
        spaceAfter=4
    ))

    ss.add(ParagraphStyle(
        name="SectionNote",
        parent=ss["Normal"],
        fontSize=10.2,
        leading=13,
        textColor=colors.HexColor("#374151"),
        backColor=colors.HexColor("#F3F8FC"),
        borderPadding=8,
        borderRadius=4,
        borderWidth=0.6,
        borderColor=colors.HexColor("#C7DDF0")
    ))

    return ss


def _page_cover(canv, doc):
    canv.saveState()

    canv.setFillColor(colors.HexColor("#0B3954"))
    canv.rect(0, PAGE_H - 2.0 * cm, PAGE_W, 2.0 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#1F5F8B"))
    canv.rect(0, PAGE_H - 2.35 * cm, PAGE_W, 0.35 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#D9A404"))
    canv.rect(2 * cm, PAGE_H - 6.2 * cm, PAGE_W - 4 * cm, 0.08 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#EAF3F9"))
    canv.rect(0, 0, PAGE_W, 1.3 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#1F5F8B"))
    canv.rect(0, 0, PAGE_W, 0.28 * cm, fill=1, stroke=0)

    canv.restoreState()


def _page_normal(canv, doc):
    canv.saveState()

    canv.setFillColor(colors.HexColor("#0B3954"))
    canv.rect(doc.leftMargin, PAGE_H - 1.35 * cm, doc.width, 0.16 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#1F5F8B"))
    canv.rect(doc.leftMargin, PAGE_H - 1.58 * cm, doc.width * 0.55, 0.08 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#EAF3F9"))
    canv.rect(0, 0, PAGE_W, 1.0 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#1F5F8B"))
    canv.rect(0, 0, PAGE_W, 0.16 * cm, fill=1, stroke=0)

    canv.setFont("Helvetica", 8.5)
    canv.setFillColor(colors.HexColor("#4B5563"))
    canv.drawRightString(PAGE_W - doc.rightMargin, 0.42 * cm, f"Página {doc.page}")

    canv.restoreState()


def _page_last(canv, doc):
    canv.saveState()

    canv.setFillColor(colors.HexColor("#0B3954"))
    canv.rect(0, PAGE_H - 1.2 * cm, PAGE_W, 1.2 * cm, fill=1, stroke=0)

    canv.setFillColor(colors.HexColor("#D9A404"))
    canv.rect(0, 0, PAGE_W, 0.32 * cm, fill=1, stroke=0)

    canv.setFont("Helvetica", 8.5)
    canv.setFillColor(colors.HexColor("#4B5563"))
    canv.drawRightString(PAGE_W - doc.rightMargin, 0.6 * cm, f"Página {doc.page}")

    canv.restoreState()
# ============================================================================
# ============================== PARTE 7/12 =================================
# =========================== Helpers PDF / texto ============================
# ============================================================================

def _tema_descriptor(descriptor: str) -> str:
    d = descriptor.lower()
    if "droga" in d or "búnker" in d or "bunker" in d or "narco" in d or "venta de drogas" in d:
        return "drogas"
    if "robo" in d or "hurto" in d or "asalto" in d or "vehícul" in d or "comercio" in d:
        return "delitos contra la propiedad"
    if "violencia" in d or "lesion" in d or "homicidio" in d:
        return "violencia"
    if "infraestructura" in d or "alumbrado" in d or "lotes" in d:
        return "condiciones urbanas / entorno"
    return "seguridad y convivencia"


def _resumen_texto(df_par: pd.DataFrame) -> str:
    if df_par.empty:
        return "Sin datos disponibles."

    total = int(df_par["frecuencia"].sum())
    n = len(df_par)
    top = df_par.iloc[0]
    df_pri = obtener_df_priorizado(df_par)
    n_pri = len(df_pri)
    pct_pri = float(df_pri["porcentaje"].sum()) if not df_pri.empty else 0.0

    return (
        f"Se registran <b>{total}</b> hechos distribuidos en <b>{n}</b> descriptores. "
        f"El descriptor de mayor incidencia es <b>{top['descriptor']}</b>, con "
        f"<b>{int(top['frecuencia'])}</b> casos ({float(top['porcentaje']):.2f}%). "
        f"Para fines de priorización, se identificaron <b>{n_pri}</b> descriptores "
        f"que concentran aproximadamente <b>{pct_pri:.2f}%</b> de los hechos."
    )


def _texto_priorizados(df_par: pd.DataFrame) -> str:
    df_pri = obtener_df_priorizado(df_par)
    if df_pri.empty:
        return "No se identificaron problemáticas priorizadas."
    return (
        f"En el informe se muestran únicamente las <b>problemáticas priorizadas</b> "
        f"del tramo Pareto, con el fin de evitar saturación visual y mejorar la legibilidad. "
        f"Se presentan <b>{len(df_pri)}</b> descriptores priorizados."
    )


def _texto_modalidades(descriptor: str, pares: List[Tuple[str, float]]) -> str:
    pares_filtrados = [(l, p) for l, p in pares if str(l).strip() and (p or 0) > 0]
    pares_orden = sorted(pares_filtrados, key=lambda x: x[1], reverse=True)
    if not pares_orden:
        return (
            f"Para <b>{descriptor}</b> no se reportaron modalidades con porcentaje. "
            "Se sugiere recolectar esta información para focalizar acciones."
        )
    top_txt = "; ".join([f"<b>{l}</b> ({p:.1f}%)" for l, p in pares_orden[:2]])
    return (
        f"En <b>{descriptor}</b> destacan: {top_txt}. "
        "Esto orienta intervenciones específicas sobre las variantes de mayor peso."
    )


def _modalidades_png(title: str, data_pairs: List[Tuple[str, float]], kind: str = "barh") -> bytes:
    labels = [l for l, p in data_pairs if str(l).strip()]
    vals = [float(p or 0) for l, p in data_pairs if str(l).strip()]
    if not labels:
        labels, vals = ["Sin datos"], [100.0]

    order = np.argsort(vals)[::-1]
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    n = len(labels)

    import matplotlib as mpl
    from matplotlib.patches import FancyBboxPatch, Circle

    cmap = mpl.cm.get_cmap("Blues")
    colors_seq = [cmap(0.35 + 0.5 * (i / max(1, n - 1))) for i in range(n)]
    dpi = 220

    if kind == "donut":
        fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=dpi)
        wedges, _, _ = ax.pie(
            vals,
            labels=None,
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            pctdistance=0.8,
            wedgeprops=dict(width=0.4, edgecolor="white"),
            colors=colors_seq,
        )
        ax.legend(
            wedges,
            [f"{l} ({v:.1f}%)" for l, v in zip(labels, vals)],
            title="Modalidades",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
        )
        ax.set_title(title, color=TEXTO)

    elif kind == "lollipop":
        fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=dpi)
        y = np.arange(n)
        ax.hlines(y=y, xmin=0, xmax=vals, color="#94a3b8", linewidth=2)
        ax.plot(vals, y, "o", markersize=8, color=AZUL)
        ax.set_yticks(y)
        ax.set_yticklabels(_wrap_labels(labels, 35))
        ax.invert_yaxis()
        ax.set_xlabel("Porcentaje")
        ax.set_xlim(0, max(100, max(vals) * 1.05))
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10)
        ax.set_title(title, color=TEXTO)

    elif kind == "bar":
        fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=dpi)
        x = np.arange(n)
        ax.bar(x, vals, color=colors_seq)
        ax.set_xticks(x)
        ax.set_xticklabels(_wrap_labels(labels, 20), rotation=0)
        ax.set_ylabel("Porcentaje")
        ax.set_ylim(0, max(100, max(vals) * 1.15))
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.03, f"{v:.1f}%", ha="center", fontsize=10)
        ax.set_title(title, color=TEXTO)

    elif kind == "comp100":
        fig, ax = plt.subplots(figsize=(11.5, 3.0), dpi=dpi)
        left = 0.0
        for i, (lab, v) in enumerate(zip(labels, vals)):
            w = max(0.0, float(v))
            ax.barh(0, w, left=left, color=colors_seq[i])
            if w >= 7:
                ax.text(
                    left + w / 2,
                    0,
                    f"{lab}\n{v:.1f}%",
                    va="center",
                    ha="center",
                    fontsize=9,
                    color="white",
                )
            left += w
        ax.set_xlim(0, max(100, sum(vals)))
        ax.set_yticks([])
        ax.set_xlabel("Porcentaje (composición)")
        ax.set_title(title, color=TEXTO)
        ax.grid(False)

    elif kind == "pill":
        fig_height = 0.9 + n * 0.85
        fig, ax = plt.subplots(figsize=(10.8, fig_height), dpi=dpi)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, n)
        ax.axis("off")
        track_h = 0.72
        round_r = track_h / 2

        for i, (lab, v) in enumerate(zip(labels, vals)):
            y = n - 1 - i + (1 - track_h) / 2
            track = FancyBboxPatch(
                (0.8, y), 98.4, track_h,
                boxstyle=f"round,pad=0,rounding_size={round_r}",
                linewidth=1, edgecolor="#9dbbd6", facecolor="#e6f0fb",
            )
            ax.add_patch(track)

            prog_w = max(0.001, min(98.4, float(v)))
            prog = FancyBboxPatch(
                (0.8, y), prog_w, track_h,
                boxstyle=f"round,pad=0,rounding_size={round_r}",
                linewidth=0, facecolor=AZUL, alpha=0.35,
            )
            ax.add_patch(prog)

            ax.add_patch(Circle(
                (0.8 + round_r * 0.6, y + track_h / 2),
                round_r * 0.9, color=AZUL, alpha=0.9
            ))

            badge_w = 12.0
            badge_h = track_h * 0.8
            badge_x = 5.0
            badge_y = y + (track_h - badge_h) / 2
            badge = FancyBboxPatch(
                (badge_x, badge_y), badge_w, badge_h,
                boxstyle=f"round,pad=0.25,rounding_size={badge_h/2}",
                linewidth=1, edgecolor="#cfd8e3", facecolor="white",
            )
            ax.add_patch(badge)
            ax.text(
                badge_x + badge_w / 2, y + track_h / 2,
                f"{v:.1f}%", ha="center", va="center", fontsize=10
            )
            ax.text(
                badge_x + badge_w + 3.0, y + track_h / 2,
                lab, va="center", ha="left", fontsize=12, color="#0f172a"
            )

        ax.set_title(title, color=TEXTO)

    else:
        fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=dpi)
        y = np.arange(n)
        ax.barh(y, vals, color=colors_seq)
        ax.set_yticks(y)
        ax.set_yticklabels(_wrap_labels(labels, 35))
        ax.invert_yaxis()
        ax.set_xlabel("Porcentaje")
        ax.set_xlim(0, max(100, max(vals) * 1.05))
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10)
        ax.set_title(title, color=TEXTO)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


# ============================================================================
# ============================== PARTE 8/12 =================================
# ========================= Tabla PDF + generadores PDF ======================
# ============================================================================

def _tabla_resultados_flowable(df_par: pd.DataFrame, doc_width: float) -> Table:
    """
    Cuadro simplificado: Descriptor | Frecuencia | %
    Muestra solo priorizados para el PDF.
    """
    df_tab = obtener_df_priorizado(df_par)
    fracs = [0.62, 0.20, 0.18]
    col_widths = [f * doc_width for f in fracs]
    stys = _styles()

    cell_style = ParagraphStyle(
        name="CellWrap",
        parent=stys["Normal"],
        fontSize=9.6,
        leading=12,
        textColor=colors.HexColor("#111111"),
        wordWrap="CJK",
        spaceBefore=0,
        spaceAfter=0,
    )

    head = [
        Paragraph("Descriptor priorizado", stys["TableHead"]),
        Paragraph("Frecuencia", stys["TableHead"]),
        Paragraph("Porcentaje", stys["TableHead"]),
    ]
    data = [head]

    total_respuestas = int(df_tab["frecuencia"].sum()) if not df_tab.empty else 0
    for _, r in df_tab.iterrows():
        descriptor = Paragraph(str(r["descriptor"]), cell_style)
        frecuencia = int(r["frecuencia"])
        pct = f'{float(r["porcentaje"]):.2f}%'
        data.append([descriptor, frecuencia, pct])

    total_row = [Paragraph("<b>Total priorizado</b>", cell_style), total_respuestas, ""]
    data.append(total_row)
    total_index = len(data) - 1

    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3954")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10.5),

        ("FONTSIZE", (0, 1), (-1, -1), 9.6),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("ALIGN", (1, 1), (-1, -2), "RIGHT"),

        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [
            colors.HexColor("#F8FBFD"),
            colors.HexColor("#EDF5FA")
        ]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D8E6")),

        ("BACKGROUND", (0, total_index), (-1, total_index), colors.HexColor("#DCEAF6")),
        ("FONTNAME", (0, total_index), (-1, total_index), "Helvetica-Bold"),
        ("ALIGN", (1, total_index), (1, total_index), "RIGHT"),
        ("ALIGN", (2, total_index), (2, total_index), "RIGHT"),
    ]))
    return t


def generar_pdf_pareto_simple(nombre_informe: str, df_par: pd.DataFrame) -> bytes:
    if df_par.empty:
        st.warning("No hay datos válidos para generar el PDF del Pareto.")
        return b""

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="normal", frames=[frame], onPage=_page_normal))

    stys = _styles()
    story: List = []

    story.append(Paragraph(f"Diagrama de Pareto – {nombre_informe}", stys["TitleBigCenter"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(_texto_priorizados(df_par), stys["Body"]))
    story.append(Spacer(1, 0.4 * cm))

    from PIL import Image as PILImage
    pareto_png = _pareto_png(df_par, f"Pareto priorizado – {nombre_informe}", solo_priorizados=True)
    with io.BytesIO(pareto_png) as _b:
        im = PILImage.open(_b)
        w_px, h_px = im.size

    width_pts = doc.width
    proportional_height = (h_px / w_px) * width_pts
    height_pts = max(proportional_height, 7 * cm)

    story.append(RLImage(io.BytesIO(pareto_png), width=width_pts, height=height_pts))
    story.append(Spacer(1, 0.6 * cm))
    story.append(_tabla_resultados_flowable(df_par, doc.width))

    doc.build(story)
    return buf.getvalue()


def generar_pdf_informe(nombre_informe: str, df_par: pd.DataFrame, desgloses: List[Dict]) -> bytes:
    if df_par.empty:
        st.warning("No hay datos válidos para generar el informe.")
        return b""

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm
    )
    frame_std = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    frame_last = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="last")

    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[frame_std], onPage=_page_cover),
        PageTemplate(id="Normal", frames=[frame_std], onPage=_page_normal),
        PageTemplate(id="Last", frames=[frame_last], onPage=_page_last),
    ])

    stys = _styles()
    story: List = []
    df_pri = obtener_df_priorizado(df_par)

    logo_1 = Path("001.png")
    logo_2 = Path("002.png")

    # -------------------------
    # PORTADA
    # -------------------------
    story += [NextPageTemplate("Normal")]
    story += [Spacer(1, 0.8 * cm)]

    if logo_1.exists():
        story += [RLImage(str(logo_1), width=6.0 * cm, height=6.0 * cm, hAlign="CENTER")]
        story += [Spacer(1, 0.18 * cm)]

    if logo_2.exists():
        from PIL import Image as PILImage

        with PILImage.open(logo_2) as im2:
          w2, h2 = im2.size

        ancho_logo2 = 10.5 * cm
        alto_logo2 = (h2 / w2) * ancho_logo2

        story += [RLImage(str(logo_2), width=ancho_logo2, height=alto_logo2, hAlign="CENTER")]
        story += [Spacer(1, 0.45 * cm)]

    story += [Paragraph("Informe de Resultados", stys["CoverTitle"])]
    story += [Paragraph("Análisis de Problemáticas Priorizadas en Seguridad Ciudadana", stys["CoverSubtitle"])]
    story += [Spacer(1, 0.20 * cm)]
    story += [Paragraph(f"<b>{nombre_informe}</b>", stys["CoverSubtitle"])]
    story += [Spacer(1, 0.35 * cm)]
    story += [Paragraph("Estrategia Integral de Prevención para la Seguridad Pública", stys["CoverDate"])]
    story += [Paragraph("“Sembremos Seguridad”", stys["CoverDate"])]
    story += [Spacer(1, 0.15 * cm)]
    story += [Paragraph(datetime.now().strftime("Fecha de emisión: %d/%m/%Y"), stys["CoverDate"])]
    story += [Spacer(1, 0.65 * cm)]
    story += [Paragraph(
        "Documento técnico generado a partir del procesamiento y priorización de datos obtenidos mediante encuestas.",
        stys["SectionNote"]
    )]
    story += [PageBreak()]

    # -------------------------
    # INTRODUCCIÓN
    # -------------------------
    story += [Paragraph("Introducción", stys["TitleBig"]), Spacer(1, 0.2 * cm)]
    story += [Paragraph(
        "Este informe presenta los resultados del análisis de la información recopilada mediante encuestas, "
        "procesada a través de un enfoque metodológico que permite identificar, agrupar y priorizar las "
        "principales problemáticas en materia de seguridad ciudadana.",
        stys["Body"]
    )]
    story += [Spacer(1, 0.22 * cm)]
    story += [Paragraph(
        "En el marco de la Estrategia Integral de Prevención para la Seguridad Pública "
        "“Sembremos Seguridad”, los datos fueron tratados por el equipo especializado utilizando "
        "herramientas como ArcGIS, lo que permitió una clasificación precisa de las variables según su "
        "relevancia y tipología.",
        stys["Body"]
    )]
    story += [Spacer(1, 0.22 * cm)]
    story += [Paragraph(
        "El objetivo es ofrecer una visión clara de las áreas prioritarias, facilitando la comprensión "
        "de los principales problemas y aportando insumos para el diseño e implementación de acciones "
        "efectivas que fortalezcan la seguridad y el bienestar de las comunidades.",
        stys["Body"]
    )]
    story += [Spacer(1, 0.45 * cm)]

    # -------------------------
    # RESULTADOS GENERALES
    # -------------------------
    story += [Paragraph("Resultados generales", stys["TitleBig"]), Spacer(1, 0.15 * cm)]
    story += [Paragraph(_resumen_texto(df_par), stys["Body"]), Spacer(1, 0.22 * cm)]
    story += [Paragraph(
        "El diagrama incluido a continuación se presenta de forma completa para una mejor lectura integral del comportamiento observado. "
        "No obstante, la tabla resumen muestra únicamente los descriptores priorizados, con el fin de enfocar la interpretación en los temas de mayor relevancia.",
        stys["SectionNote"]
    )]
    story += [Spacer(1, 0.32 * cm)]

    # -------------------------
    # DIAGRAMA COMPLETO
    # -------------------------
    from PIL import Image as PILImage
    pareto_png = _pareto_png(df_par, "Diagrama de Pareto completo", solo_priorizados=False)

    with io.BytesIO(pareto_png) as _b:
        im = PILImage.open(_b)
        w_px, h_px = im.size

    width_pts = doc.width
    height_pts = (h_px / w_px) * width_pts

    story.append(KeepTogether([
        RLImage(io.BytesIO(pareto_png), width=width_pts, height=height_pts),
        Spacer(1, 0.22 * cm),
        Paragraph(
            "Figura 1. Diagrama de Pareto completo de las problemáticas registradas.",
            stys["Small"]
        ),
    ]))

    # -------------------------
    # TABLA SOLO PRIORIZADOS
    # -------------------------
    story.append(KeepTogether([
        Spacer(1, 0.35 * cm),
        Paragraph("Tabla de descriptores priorizados", stys["TitleBig"]),
        Spacer(1, 0.18 * cm),
        _tabla_resultados_flowable(df_par, doc.width),
    ]))

    # -------------------------
    # MODALIDADES
    # -------------------------
    prior_desc = set(df_pri["descriptor"].tolist())
    for sec in desgloses:
        descriptor = sec.get("descriptor", "").strip()
        if descriptor not in prior_desc:
            continue

        rows = sec.get("rows", [])
        chart_kind = sec.get("chart", "barh")
        pares = [(r.get("Etiqueta", ""), float(r.get("%", 0) or 0)) for r in rows]

        bloque = [
            Spacer(1, 0.45 * cm),
            Paragraph(f"Modalidades — {descriptor}", stys["TitleBig"]),
            Spacer(1, 0.12 * cm),
            Paragraph(_texto_modalidades(descriptor, pares), stys["Small"]),
            Spacer(1, 0.20 * cm),
            RLImage(
                io.BytesIO(_modalidades_png(descriptor or "Modalidades", pares, kind=chart_kind)),
                width=doc.width,
                height=8.5 * cm
            ),
        ]
        story.append(KeepTogether(bloque))

    # -------------------------
    # CONCLUSIONES
    # -------------------------
    story += [PageBreak(), NextPageTemplate("Last")]
    story += [
        Paragraph("Conclusiones y recomendaciones", stys["TitleBigCenter"]),
        Spacer(1, 0.2 * cm),
    ]

    bullets = [
        "Priorizar intervenciones sobre los descriptores que conforman el tramo priorizado del análisis Pareto.",
        "Coordinar acciones interinstitucionales enfocadas en las modalidades con mayor porcentaje.",
        "Fortalecer la participación comunitaria con presencia territorial.",
        "Monitorear y actualizar los indicadores periódicamente.",
    ]
    for b in bullets:
        story += [Paragraph(b, stys["BulletList"], bulletText="•")]

    story += [
        Spacer(1, 0.8 * cm),
        Paragraph("Dirección de Programas Policiales Preventivos – MSP", stys["H1Center"]),
        Paragraph("Sembremos Seguridad", stys["H1Center"]),
    ]

    doc.build(story)
    return buf.getvalue()


def ui_desgloses(descriptor_list: List[str], key_prefix: str) -> List[Dict]:
    st.caption("Opcional: agrega secciones de ‘Modalidades’. Cada sección admite hasta 10 filas (Etiqueta + %).")
    max_secs = max(0, len(descriptor_list))
    default_val = 1 if max_secs > 0 else 0
    n_secs = st.number_input(
        "Cantidad de secciones de Modalidades",
        min_value=0,
        max_value=max_secs,
        value=default_val,
        step=1,
        key=f"{key_prefix}_nsecs"
    )
    desgloses: List[Dict] = []
    for i in range(n_secs):
        with st.expander(f"Sección Modalidades #{i+1}", expanded=(i == 0)):
            dsel = st.selectbox(
                f"Descriptor para la sección #{i+1}",
                options=["(elegir)"] + descriptor_list,
                index=0,
                key=f"{key_prefix}_desc_{i}"
            )

            chart_kind = st.selectbox(
                "Tipo de gráfico",
                options=[
                    ("Barras horizontales", "barh"),
                    ("Barras verticales", "bar"),
                    ("Lollipop (palo+punto)", "lollipop"),
                    ("Dona / Pie", "donut"),
                    ("Barra 100% (composición)", "comp100"),
                    ("Píldora (progreso redondeado)", "pill")
                ],
                index=0,
                format_func=lambda x: x[0],
                key=f"{key_prefix}_chart_{i}"
            )[1]

            rows = [{"Etiqueta": "", "%": 0.0} for _ in range(10)]
            df_rows = pd.DataFrame(rows)
            de = st.data_editor(
                df_rows,
                key=f"{key_prefix}_rows_{i}",
                use_container_width=True,
                column_config={
                    "Etiqueta": st.column_config.TextColumn("Etiqueta / Modalidad", width="large"),
                    "%": st.column_config.NumberColumn("Porcentaje", min_value=0.0, max_value=100.0, step=0.1),
                },
                num_rows="fixed",
            )
            total_pct = float(pd.to_numeric(de["%"], errors="coerce").fillna(0).sum())
            st.caption(f"Suma actual: {total_pct:.1f}% (recomendado ≈100%)")

            if dsel != "(elegir)":
                desgloses.append({
                    "descriptor": dsel,
                    "rows": de.to_dict(orient="records"),
                    "chart": chart_kind
                })

    return desgloses

# ============================================================================
# ============================== PARTE 9/12 =================================
# ===================== MIC MAC por Excel (plantilla + carga) ===============
# ============================================================================

def clasificar_variable_micmac(descriptor: str) -> str:
    d = str(descriptor).lower().strip()

    if any(x in d for x in ["droga", "búnker", "bunker", "narco", "venta de drogas"]):
        return "drogas"

    if any(x in d for x in ["alcohol", "licor", "ebriedad"]):
        return "alcohol"

    if any(x in d for x in ["homicidio", "lesiones", "violencia", "riñas", "disturbios", "delitos sexuales", "violación"]):
        return "violencia"

    if any(x in d for x in ["robo", "hurto", "asalto", "tacha", "bajonazo", "receptación", "estafa", "defraudación"]):
        return "patrimonial"

    if any(x in d for x in ["situación de calle", "desempleo", "inversión social", "familias disfuncionales", "desvinculación", "analfabetismo"]):
        return "exclusion_social"

    if any(x in d for x in ["infraestructura", "alumbrado", "lotes baldíos", "salubridad", "vial"]):
        return "entorno"

    if any(x in d for x in ["presencia policial", "corrupción policial", "inefectividad en el servicio de policía", "falta de capacitación policial"]):
        return "institucional"

    if any(x in d for x in ["personas en situación migratoria irregular", "xenofobia", "presencia multicultural"]):
        return "movilidad_social"

    return "otros"


def peso_relacion_micmac(origen: str, destino: str) -> int:
    """
    Devuelve un peso sugerido 0-3 para la relación origen -> destino.
    0 = nula, 1 = débil, 2 = media, 3 = fuerte
    """
    o = str(origen).lower().strip()
    d = str(destino).lower().strip()

    if o == d:
        return 0

    to = clasificar_variable_micmac(o)
    td = clasificar_variable_micmac(d)

    reglas_fuertes = [
        ("consumo de drogas", "venta de drogas"),
        ("consumo de alcohol en vía pública", "disturbios (riñas)"),
        ("consumo de alcohol en vía pública", "violencia intrafamiliar"),
        ("venta de drogas", "delincuencia organizada"),
        ("personas en situación de calle", "consumo de drogas"),
        ("falta de inversión social", "personas en situación de calle"),
        ("falta de inversión social", "consumo de drogas"),
        ("falta de presencia policial", "percepción de inseguridad"),
    ]
    for a, b in reglas_fuertes:
        if a in o and b in d:
            return 3

    reglas_medias = [
        ("consumo de drogas", "robo a personas"),
        ("consumo de drogas", "hurto"),
        ("venta de drogas", "robo a personas"),
        ("venta de drogas", "homicidio"),
        ("alcohol", "violencia"),
        ("exclusion_social", "drogas"),
        ("exclusion_social", "patrimonial"),
        ("entorno", "patrimonial"),
        ("institucional", "drogas"),
        ("institucional", "violencia"),
        ("institucional", "patrimonial"),
    ]

    for a, b in reglas_medias:
        if a in ["alcohol", "exclusion_social", "entorno", "institucional"]:
            if to == a and td == b:
                return 2
        else:
            if a in o and b in d:
                return 2

    if to == td:
        if to in ["drogas", "violencia", "patrimonial"]:
            return 2
        return 1

    cruces_2 = {
        ("drogas", "violencia"),
        ("drogas", "patrimonial"),
        ("alcohol", "violencia"),
        ("exclusion_social", "drogas"),
        ("exclusion_social", "violencia"),
        ("exclusion_social", "patrimonial"),
        ("entorno", "patrimonial"),
        ("institucional", "drogas"),
        ("institucional", "violencia"),
        ("institucional", "patrimonial"),
    }
    if (to, td) in cruces_2:
        return 2

    cruces_1 = {
        ("entorno", "violencia"),
        ("entorno", "drogas"),
        ("movilidad_social", "drogas"),
        ("movilidad_social", "patrimonial"),
        ("otros", "patrimonial"),
        ("otros", "violencia"),
    }
    if (to, td) in cruces_1:
        return 1

    return 0


def generar_matriz_micmac_sugerida(variables: List[str]) -> pd.DataFrame:
    """
    Genera una matriz MIC MAC sugerida automáticamente con base en reglas lógicas.
    """
    if not variables:
        return pd.DataFrame()

    matriz = pd.DataFrame(0, index=variables, columns=variables, dtype=int)

    for origen in variables:
        for destino in variables:
            if origen == destino:
                matriz.loc[origen, destino] = 0
            else:
                matriz.loc[origen, destino] = peso_relacion_micmac(origen, destino)

    return matriz


def construir_matriz_micmac(variables: List[str], edited_df: pd.DataFrame | None = None) -> pd.DataFrame:
    n = len(variables)
    if n == 0:
        return pd.DataFrame()

    if edited_df is not None and not edited_df.empty:
        df = edited_df.copy()
        df.index = variables
        df.columns = variables
    else:
        df = generar_matriz_micmac_sugerida(variables)

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    df = df.clip(lower=0, upper=3)

    for i in range(n):
        df.iat[i, i] = 0
    return df


def calcular_micmac(matriz: pd.DataFrame) -> pd.DataFrame:
    if matriz.empty:
        return pd.DataFrame(columns=["variable", "influencia", "dependencia", "zona"])

    infl = matriz.sum(axis=1)
    dep = matriz.sum(axis=0)

    df = pd.DataFrame({
        "variable": matriz.index,
        "influencia": infl.values,
        "dependencia": dep.values
    })

    med_infl = df["influencia"].mean() if not df.empty else 0
    med_dep = df["dependencia"].mean() if not df.empty else 0

    def zona(row):
        if row["influencia"] == 0 and row["dependencia"] == 0:
            return "Sin relación"
        if row["influencia"] >= med_infl and row["dependencia"] >= med_dep:
            return "Conflicto"
        elif row["influencia"] >= med_infl and row["dependencia"] < med_dep:
            return "Poder"
        elif row["influencia"] < med_infl and row["dependencia"] >= med_dep:
            return "Resultado"
        else:
            return "Autónoma"

    df["zona"] = df.apply(zona, axis=1)
    return df.sort_values(["influencia", "dependencia"], ascending=[False, False]).reset_index(drop=True)


def _micmac_matrix_png(matriz: pd.DataFrame, titulo: str = "Matriz MIC MAC") -> bytes:
    if matriz.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
        ax.axis("off")
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return buf.getvalue()

    n = len(matriz)
    fig_w = max(8, n * 0.48)
    fig_h = max(6, n * 0.42)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    arr = matriz.to_numpy()
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=3)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(_wrap_labels(matriz.columns.tolist(), 16), fontsize=7)
    ax.set_yticklabels(_wrap_labels(matriz.index.tolist(), 18), fontsize=7)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(n):
        for j in range(n):
            val = int(arr[i, j])
            txt_color = "white" if val >= 2 else "#0f172a"
            ax.text(j, i, str(val), ha="center", va="center", color=txt_color, fontsize=6.5)

    ax.set_title(titulo, color=TEXTO, fontsize=16)
    ax.set_xlabel("Dependencia")
    ax.set_ylabel("Influencia")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="PNG", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


def _micmac_map_png(df_mic: pd.DataFrame, titulo: str = "Mapa MIC MAC") -> bytes:
    if df_mic.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
        ax.axis("off")
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return buf.getvalue()

    if df_mic["influencia"].sum() == 0:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=220)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "La matriz MIC MAC no tiene relaciones definidas.\nDebe asignar o revisar los cruces para generar el mapa.",
            ha="center", va="center", fontsize=13
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=220, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return buf.getvalue()

    fig, ax = plt.subplots(figsize=(12, 8), dpi=220)

    zone_colors = {
        "Conflicto": ROJO,
        "Poder": VERDE,
        "Resultado": AZUL,
        "Autónoma": NARANJA,
        "Sin relación": GRIS,
    }

    x = df_mic["dependencia"].to_numpy(dtype=float)
    y = df_mic["influencia"].to_numpy(dtype=float)

    jitter = np.linspace(-0.12, 0.12, len(df_mic)) if len(df_mic) > 1 else np.array([0.0])

    for idx, (_, r) in enumerate(df_mic.iterrows()):
        xx = r["dependencia"] + jitter[idx]
        yy = r["influencia"] + jitter[::-1][idx]

        ax.scatter(
            xx, yy,
            s=140,
            alpha=0.90,
            color=zone_colors.get(r["zona"], AZUL),
            edgecolors="black",
            linewidths=0.6
        )

        etiqueta = "\n".join(wrap(str(r["variable"]), width=22))
        ax.text(
            xx + 0.15,
            yy + 0.15,
            etiqueta,
            fontsize=8,
            ha="left",
            va="bottom"
        )

    mean_x = x.mean() if len(x) else 0
    mean_y = y.mean() if len(y) else 0

    ax.axvline(mean_x, linestyle="--", color="#666666", linewidth=1)
    ax.axhline(mean_y, linestyle="--", color="#666666", linewidth=1)

    ax.set_title(titulo, color=TEXTO, fontsize=18)
    ax.set_xlabel("Dependencia")
    ax.set_ylabel("Influencia")
    ax.grid(True, alpha=0.25)

    ax.set_xlim(left=min(x.min() - 1, 0), right=x.max() + 2)
    ax.set_ylim(bottom=min(y.min() - 1, 0), top=y.max() + 2)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="PNG", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


def _micmac_validar_valores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    out = out.clip(lower=0, upper=3)
    return out


def _micmac_crear_plantilla_excel(
    variables: List[str],
    nombre_fuente: str = "MICMAC",
    usar_sugerida: bool = False
) -> bytes:
    if not variables:
        return b""

    matriz = (
        generar_matriz_micmac_sugerida(variables)
        if usar_sugerida else
        pd.DataFrame(0, index=variables, columns=variables, dtype=int)
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # ---------------- Hoja matriz ----------------
        sheet_m = "Matriz_MICMAC"
        matriz_export = matriz.copy()
        matriz_export.insert(0, "Variable", matriz_export.index)
        matriz_export.to_excel(writer, sheet_name=sheet_m, index=False)

        wb = writer.book
        ws = writer.sheets[sheet_m]

        hdr_fmt = wb.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "bg_color": "#0B3954",
            "font_color": "white",
            "border": 1,
            "text_wrap": True,
            "locked": True
        })
        row_hdr_fmt = wb.add_format({
            "bold": True,
            "bg_color": "#DCEAF6",
            "border": 1,
            "text_wrap": True,
            "locked": True
        })
        cell_fmt = wb.add_format({
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "locked": False
        })
        diag_fmt = wb.add_format({
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "bg_color": "#BFBFBF",
            "font_color": "#000000",
            "bold": True,
            "locked": True
        })
        note_fmt = wb.add_format({
            "text_wrap": True,
            "valign": "top",
            "border": 1
        })

        n = len(variables)

        ws.set_row(0, 36)
        ws.set_column(0, 0, 34)
        for c in range(1, n + 1):
            ws.set_column(c, c, 16)

        for c in range(0, n + 1):
            ws.write(0, c, matriz_export.columns[c], hdr_fmt)

        for r in range(1, n + 1):
            ws.write(r, 0, variables[r - 1], row_hdr_fmt)

        for r in range(n):
            for c in range(n):
                val = int(matriz.iat[r, c])
                fmt = diag_fmt if r == c else cell_fmt
                ws.write(r + 1, c + 1, val, fmt)

        ws.freeze_panes(1, 1)

        # Validación SOLO para celdas fuera de diagonal
        for r in range(1, n + 1):
            for c in range(1, n + 1):
                if r != c:
                    ws.data_validation(r, c, r, c, {
                        "validate": "integer",
                        "criteria": "between",
                        "minimum": 0,
                        "maximum": 3,
                        "input_title": "Valor MIC MAC",
                        "input_message": "Use únicamente: 0, 1, 2 o 3",
                        "error_title": "Valor inválido",
                        "error_message": "Solo se permiten valores enteros entre 0 y 3"
                    })

        # Protección de hoja:
        # encabezados y diagonal bloqueados; resto editable
        ws.protect("micmac")
        ws.protection_format_cells = False
        ws.protection_format_columns = False
        ws.protection_format_rows = False
        ws.protection_insert_columns = False
        ws.protection_insert_rows = False
        ws.protection_insert_hyperlinks = False
        ws.protection_delete_columns = False
        ws.protection_delete_rows = False
        ws.protection_sort = False
        ws.protection_autofilter = False
        ws.protection_pivot_tables = False
        ws.protection_select_locked_cells = True
        ws.protection_select_unlocked_cells = True

        # ---------------- Hoja instrucciones ----------------
        instrucciones = pd.DataFrame({
            "Instrucción": [
                "Complete únicamente la hoja Matriz_MICMAC.",
                "No cambie nombres de variables ni encabezados.",
                "La diagonal principal está bloqueada y debe permanecer en 0.",
                "Valores permitidos fuera de la diagonal: 0 = nula, 1 = débil, 2 = media, 3 = fuerte.",
                "Luego cargue este mismo archivo en la app para generar el análisis MIC MAC."
            ]
        })
        instrucciones.to_excel(writer, sheet_name="Instrucciones", index=False)
        ws_i = writer.sheets["Instrucciones"]
        ws_i.set_column(0, 0, 95, note_fmt)

        # ---------------- Hoja variables ----------------
        pd.DataFrame({"Variables priorizadas": variables}).to_excel(
            writer, sheet_name="Variables", index=False
        )
        ws_v = writer.sheets["Variables"]
        ws_v.set_column(0, 0, 55)

        # ---------------- Hoja metadata ----------------
        meta = pd.DataFrame({
            "Campo": ["Fuente", "Fecha de generación", "Cantidad de variables", "Tipo plantilla"],
            "Valor": [
                nombre_fuente,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(variables),
                "Sugerida" if usar_sugerida else "En blanco"
            ]
        })
        meta.to_excel(writer, sheet_name="Metadata", index=False)
        ws_meta = writer.sheets["Metadata"]
        ws_meta.set_column(0, 1, 30)

    return output.getvalue()


def _micmac_leer_excel_subido(file, variables_esperadas: List[str]) -> Tuple[pd.DataFrame, str]:
    if file is None:
        return pd.DataFrame(), "No se cargó ningún archivo."

    try:
        xls = pd.ExcelFile(file)
    except Exception as e:
        return pd.DataFrame(), f"No se pudo abrir el Excel: {e}"

    if "Matriz_MICMAC" not in xls.sheet_names:
        return pd.DataFrame(), "El archivo no contiene la hoja 'Matriz_MICMAC'."

    try:
        df = pd.read_excel(file, sheet_name="Matriz_MICMAC")
    except Exception as e:
        return pd.DataFrame(), f"No se pudo leer la hoja 'Matriz_MICMAC': {e}"

    if df.empty:
        return pd.DataFrame(), "La hoja 'Matriz_MICMAC' está vacía."

    if "Variable" not in df.columns:
        return pd.DataFrame(), "La hoja 'Matriz_MICMAC' debe tener una columna llamada 'Variable'."

    filas = [str(x).strip() for x in df["Variable"].tolist()]
    cols = [str(c).strip() for c in df.columns.tolist()[1:]]
    esperadas = [str(v).strip() for v in variables_esperadas]

    if filas != esperadas:
        return pd.DataFrame(), "Las variables de las filas no coinciden exactamente con las priorizadas esperadas."

    if cols != esperadas:
        return pd.DataFrame(), "Las variables de las columnas no coinciden exactamente con las priorizadas esperadas."

    mat = df.set_index("Variable").copy()
    mat = _micmac_validar_valores(mat)

    if mat.shape[0] != mat.shape[1]:
        return pd.DataFrame(), "La matriz no es cuadrada."

    for i in range(len(mat)):
        mat.iat[i, i] = 0

    return mat, ""


def _micmac_exportar_resultados_excel(
    matriz: pd.DataFrame,
    df_mic: pd.DataFrame,
    source_name: str
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        # Hoja matriz
        matriz_out = matriz.copy()
        matriz_out.insert(0, "Variable", matriz_out.index)
        matriz_out.to_excel(writer, sheet_name="Matriz_MICMAC", index=False)
        ws_m = writer.sheets["Matriz_MICMAC"]
        ws_m.set_column(0, 0, 34)
        for c in range(1, len(matriz.columns) + 1):
            ws_m.set_column(c, c, 16)

        # Hoja análisis
        df_mic.to_excel(writer, sheet_name="Analisis_MICMAC", index=False)
        ws_a = writer.sheets["Analisis_MICMAC"]
        ws_a.set_column(0, 0, 42)
        ws_a.set_column(1, 2, 14)
        ws_a.set_column(3, 3, 16)

        # Hoja coordenadas
        coords = df_mic[["variable", "dependencia", "influencia", "zona"]].copy()
        coords.to_excel(writer, sheet_name="Mapa_MICMAC", index=False)
        ws_c = writer.sheets["Mapa_MICMAC"]
        ws_c.set_column(0, 0, 42)
        ws_c.set_column(1, 2, 14)
        ws_c.set_column(3, 3, 16)

        # Hoja resumen
        resumen = pd.DataFrame({
            "Campo": [
                "Fuente",
                "Fecha procesamiento",
                "Cantidad variables",
                "Suma influencia",
                "Suma dependencia"
            ],
            "Valor": [
                source_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(matriz.index),
                int(df_mic["influencia"].sum()) if not df_mic.empty else 0,
                int(df_mic["dependencia"].sum()) if not df_mic.empty else 0
            ]
        })
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        ws_r = writer.sheets["Resumen"]
        ws_r.set_column(0, 1, 30)

    return output.getvalue()


def ui_micmac(source_name: str, df_par: pd.DataFrame, key_prefix: str):
    st.subheader(f"🧩 MIC MAC — {source_name}")

    df_pri = obtener_df_priorizado(df_par)
    if df_pri.empty:
        st.info("No hay descriptores priorizados para construir MIC MAC.")
        return

    st.caption(
        "Ahora el MIC MAC trabaja por plantilla Excel. "
        "La app toma todas las priorizadas que selecciones, te genera un Excel, "
        "tú lo completas fuera de la app y luego lo subes para generar el análisis."
    )

    opciones = df_pri["descriptor"].tolist()

    variables = st.multiselect(
        "Variables priorizadas que entrarán al MIC MAC",
        options=opciones,
        default=opciones,
        key=f"{key_prefix}_variables_micmac_excel"
    )

    if len(variables) < 2:
        st.warning("Selecciona al menos 2 variables para construir el MIC MAC.")
        return

    st.markdown("**Variables seleccionadas**")
    df_sel = df_pri[df_pri["descriptor"].isin(variables)][["descriptor", "frecuencia", "porcentaje"]].copy()
    df_sel = df_sel.rename(columns={
        "descriptor": "Variable",
        "frecuencia": "Frecuencia",
        "porcentaje": "%"
    })
    st.dataframe(df_sel, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 1) Descargar plantilla Excel para llenar cruces")

    ctpl1, ctpl2 = st.columns(2)

    with ctpl1:
        plantilla_blanca = _micmac_crear_plantilla_excel(
            variables=variables,
            nombre_fuente=source_name,
            usar_sugerida=False
        )
        st.download_button(
            "📥 Descargar plantilla MIC MAC en blanco",
            data=plantilla_blanca,
            file_name=f"Plantilla_MICMAC_{source_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_plantilla_blanca"
        )

    with ctpl2:
        plantilla_sugerida = _micmac_crear_plantilla_excel(
            variables=variables,
            nombre_fuente=source_name,
            usar_sugerida=True
        )
        st.download_button(
            "📥 Descargar plantilla MIC MAC sugerida",
            data=plantilla_sugerida,
            file_name=f"Plantilla_MICMAC_Sugerida_{source_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_plantilla_sugerida"
        )

    st.divider()
    st.markdown("### 2) Subir Excel completado")

    archivo_micmac = st.file_uploader(
        "Sube aquí el archivo Excel MIC MAC ya lleno",
        type=["xlsx"],
        key=f"{key_prefix}_uploader_excel_micmac"
    )

    if archivo_micmac is None:
        st.info("Descarga la plantilla, complétala en Excel y luego vuelve a subirla aquí.")
        return

    matriz, err = _micmac_leer_excel_subido(archivo_micmac, variables)
    if err:
        st.error(err)
        return

    total_rel = int(matriz.to_numpy().sum())

    st.success("Excel MIC MAC cargado correctamente.")
    st.caption(f"Cantidad de variables: {len(variables)} | Suma total de relaciones: {total_rel}")

    st.markdown("**Vista previa de la matriz cargada**")
    st.dataframe(matriz, use_container_width=True)

    if total_rel == 0:
        st.error("⚠️ La matriz está en cero. Debes asignar relaciones para generar el MIC MAC.")
        return

    df_mic = calcular_micmac(matriz)

    st.markdown("### 3) Resultados MIC MAC")
    st.dataframe(
        df_mic[["variable", "influencia", "dependencia", "zona"]],
        use_container_width=True,
        hide_index=True
    )

    c1, c2 = st.columns(2)

    with c1:
        mat_png = _micmac_matrix_png(matriz, titulo=f"Matriz MIC MAC – {source_name}")
        st.image(mat_png, caption="Matriz MIC MAC", use_container_width=True)
        st.download_button(
            "📥 Descargar matriz MIC MAC (PNG)",
            data=mat_png,
            file_name=f"MICMAC_Matriz_{source_name.replace(' ', '_')}.png",
            mime="image/png",
            key=f"{key_prefix}_dl_mat"
        )

    with c2:
        map_png = _micmac_map_png(df_mic, titulo=f"Mapa MIC MAC – {source_name}")
        st.image(map_png, caption="Mapa MIC MAC", use_container_width=True)
        st.download_button(
            "📥 Descargar mapa MIC MAC (PNG)",
            data=map_png,
            file_name=f"MICMAC_Mapa_{source_name.replace(' ', '_')}.png",
            mime="image/png",
            key=f"{key_prefix}_dl_map"
        )

    st.divider()
    st.markdown("### 4) Descargar resultados MIC MAC en Excel")

    excel_resultados = _micmac_exportar_resultados_excel(matriz, df_mic, source_name)
    st.download_button(
        "📥 Descargar resultados MIC MAC (Excel)",
        data=excel_resultados,
        file_name=f"Resultados_MICMAC_{source_name.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key_prefix}_dl_excel_resultados"
    )
# ============================================================================
# ============================== PARTE 10/12 ================================
# ========================== UI Editor principal =============================
# ============================================================================

def fusionar_editor_y_manuales(df_catalogo: pd.DataFrame, df_manual: pd.DataFrame) -> pd.DataFrame:
    frames = []

    if df_catalogo is not None and not df_catalogo.empty:
        df1 = df_catalogo.copy()
        df1["descriptor"] = df1["descriptor"].astype(str).str.strip()
        df1["frecuencia"] = pd.to_numeric(df1["frecuencia"], errors="coerce").fillna(0).astype(int)
        frames.append(df1[["descriptor", "frecuencia"]])

    if df_manual is not None and not df_manual.empty:
        df2 = df_manual.copy()
        df2["descriptor"] = df2["descriptor"].astype(str).str.strip()
        df2["frecuencia"] = pd.to_numeric(df2["frecuencia"], errors="coerce").fillna(0).astype(int)
        df2 = df2[df2["descriptor"] != ""]
        frames.append(df2[["descriptor", "frecuencia"]])

    if not frames:
        return pd.DataFrame(columns=["descriptor", "frecuencia"])

    df = pd.concat(frames, ignore_index=True)
    df = df.groupby("descriptor", as_index=False)["frecuencia"].sum()
    df = df[df["frecuencia"] >= 0].sort_values("descriptor").reset_index(drop=True)
    return df


st.title("📊 Análisis Pareto 80/20 – Descriptores, Portafolio y MIC MAC")

tab_editor, tab_portafolio, tab_unificado, tab_micmac = st.tabs([
    "➕ Crear / Editar Pareto individual",
    "📁 Portafolio guardado",
    "📄 Informe PDF (unificado)",
    "🧩 MIC MAC"
])

# ---------------------------------------------------------------------------
# TAB 1 — Editor individual
# ---------------------------------------------------------------------------
with tab_editor:
    st.subheader("✏️ Editor de Pareto individual")

    nombre_pareto = st.text_input("Nombre del Pareto", "").strip()

    opts = sorted([c["descriptor"] for c in CATALOGO]) if CATALOGO else []

    # ---- CAMBIO: sin opción de select all del multiselect nativo ----
    st.markdown("**Selecciona los descriptores a incluir desde el catálogo**")
    filtro_desc = st.text_input(
        "Buscar descriptor en catálogo",
        value="",
        key="filtro_catalogo_desc"
    ).strip().lower()

    opts_filtrados = [o for o in opts if filtro_desc in o.lower()] if filtro_desc else opts

    msel = st.multiselect(
        "Descriptores",
        options=opts_filtrados,
        default=[x for x in st.session_state.get("msel", []) if x in opts_filtrados],
        key="msel_visible",
        label_visibility="collapsed"
    )

    # Sincroniza selección visible con selección real acumulada
    seleccion_previa = set(st.session_state.get("msel", []))
    visibles_previos = set([x for x in seleccion_previa if x in opts_filtrados])
    visibles_nuevos = set(msel)

    seleccion_actualizada = (seleccion_previa - visibles_previos) | visibles_nuevos
    st.session_state["msel"] = sorted(seleccion_actualizada)

    if st.session_state["msel"]:
        st.caption(f"Descriptores seleccionados: {len(st.session_state['msel'])}")

        if st.button("🧹 Limpiar selección de catálogo", key="limpiar_seleccion_catalogo"):
            st.session_state["msel"] = []
            st.session_state["last_msel"] = []
            st.session_state["editor_df"] = pd.DataFrame(columns=["descriptor", "frecuencia"])
            st.rerun()

    msel_real = st.session_state["msel"]

    if msel_real != st.session_state.get("last_msel", []):
        data = []
        freq_map_actual = st.session_state.get("freq_map", {})
        manuales_actuales = set(
            st.session_state.get(
                "manual_rows",
                pd.DataFrame(columns=["descriptor"])
            ).get("descriptor", [])
        )
        for d in msel_real:
            if d not in manuales_actuales:
                data.append({
                    "descriptor": d,
                    "frecuencia": freq_map_actual.get(d, 0)
                })
        st.session_state["editor_df"] = pd.DataFrame(data)
        st.session_state["last_msel"] = list(msel_real)

    if msel_real:
        df_edit = st.data_editor(
            st.session_state["editor_df"],
            num_rows="fixed",
            use_container_width=True,
            key="editor_freq",
            column_config={
                "descriptor": st.column_config.TextColumn("Descriptor", width="large", disabled=True),
                "frecuencia": st.column_config.NumberColumn("Frecuencia", min_value=0, step=1)
            }
        )
        st.session_state["editor_df"] = df_edit
    else:
        st.session_state["editor_df"] = pd.DataFrame(columns=["descriptor", "frecuencia"])

    with st.expander("➕ Agregar descriptores manuales / caso especial", expanded=False):
        st.caption(
            "Aquí puedes agregar descriptores que no estén en el catálogo. "
            "Se integran al Pareto, al portafolio, a Excel, a PDF y al MIC MAC."
        )
        manual_df_base = st.session_state.get(
            "manual_rows",
            pd.DataFrame(columns=["descriptor", "frecuencia"])
        )
        manual_df = st.data_editor(
            manual_df_base if not manual_df_base.empty else pd.DataFrame([{"descriptor": "", "frecuencia": 0}]),
            use_container_width=True,
            key="manual_editor",
            num_rows="dynamic",
            column_config={
                "descriptor": st.column_config.TextColumn("Descriptor manual", width="large"),
                "frecuencia": st.column_config.NumberColumn("Frecuencia", min_value=0, step=1)
            }
        )
        manual_df["descriptor"] = manual_df["descriptor"].astype(str).str.strip()
        manual_df = manual_df[
            (manual_df["descriptor"] != "") &
            (manual_df["descriptor"].str.lower() != "nan")
        ]
        st.session_state["manual_rows"] = manual_df[["descriptor", "frecuencia"]].reset_index(drop=True)

    df_total_editor = fusionar_editor_y_manuales(
        st.session_state["editor_df"],
        st.session_state["manual_rows"]
    )

    if not df_total_editor.empty:
        freq_map = dict(zip(df_total_editor["descriptor"], df_total_editor["frecuencia"]))
        st.session_state["freq_map"] = freq_map

        df_par = calcular_pareto(df_desde_freq_map(freq_map))
        df_pri = obtener_df_priorizado(df_par)

        st.divider()
        st.subheader("📊 Diagrama de Pareto (Vista previa completa)")
        dibujar_pareto(df_par, nombre_pareto or "Pareto individual")

        cimg1, cimg2, cimg3 = st.columns(3)
        with cimg1:
            st.download_button(
                "📥 Imagen Pareto completo (PNG)",
                data=_pareto_png(df_par, f"Pareto completo – {nombre_pareto or 'sin_nombre'}", solo_priorizados=False),
                file_name=f"Pareto_Completo_{(nombre_pareto or 'sin_nombre').replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
        with cimg2:
            st.download_button(
                "📥 Imagen Pareto priorizado (PNG)",
                data=_pareto_png(df_par, f"Pareto priorizado – {nombre_pareto or 'sin_nombre'}", solo_priorizados=True),
                file_name=f"Pareto_Priorizado_{(nombre_pareto or 'sin_nombre').replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
        with cimg3:
            st.download_button(
                "📥 Exportar Excel con gráfico",
                exportar_excel_con_grafico(df_par, nombre_pareto or "Pareto"),
                file_name=f"Pareto_{nombre_pareto or 'sin_nombre'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.markdown("### 🔎 Problemáticas priorizadas para PDF e MIC MAC")
        st.dataframe(
            df_pri[["descriptor", "categoria", "frecuencia", "porcentaje"]].rename(columns={
                "descriptor": "Descriptor",
                "categoria": "Categoría",
                "frecuencia": "Frecuencia",
                "porcentaje": "%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        desgloses = ui_desgloses(df_pri["descriptor"].tolist(), key_prefix="editor")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Guardar en Portafolio (y Sheets)", type="primary", use_container_width=True):
                if nombre_pareto:
                    st.session_state["portafolio"][nombre_pareto] = normalizar_freq_map(freq_map)
                    sheets_guardar_pareto(nombre_pareto, freq_map, sobrescribir=True)
                    st.success(f"Pareto '{nombre_pareto}' guardado correctamente.")
                    st.session_state["reset_after_save"] = True
                    st.rerun()
                else:
                    st.warning("Asigna un nombre al Pareto antes de guardar.")

        with col2:
            if st.button("🧾 Generar Informe PDF individual", use_container_width=True):
                if not nombre_pareto:
                    st.warning("Asigna un nombre para el informe.")
                else:
                    pdf_bytes = generar_pdf_informe(nombre_pareto, df_par, desgloses)
                    if pdf_bytes:
                        st.download_button(
                            label="📥 Descargar PDF",
                            data=pdf_bytes,
                            file_name=f"Informe_{nombre_pareto}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

    else:
        st.session_state["freq_map"] = {}
        st.info("Selecciona al menos un descriptor o agrega uno manual para comenzar.")
# ============================================================================
# ============================== PARTE 11/12 ================================
# ======================== UI Portafolio / Unificado / MIC MAC ==============
# ============================================================================

# ---------------------------------------------------------------------------
# TAB 2 — Portafolio de Paretos
# ---------------------------------------------------------------------------
with tab_portafolio:
    st.subheader("📁 Paretos almacenados en portafolio")
    port = st.session_state["portafolio"]

    if not port:
        st.info("No hay Paretos guardados todavía.")
    else:
        st.markdown("### 📊 Pareto general a partir de Paretos del portafolio")

        nombres_port = list(port.keys())
        seleccion_port = st.multiselect(
            "Selecciona los Paretos del portafolio que quieres combinar en un Pareto general",
            options=nombres_port,
            default=[],
            key="portafolio_general"
        )

        if seleccion_port:
            mapas_sel = [port[n] for n in seleccion_port]
            mapa_total = combinar_maps(mapas_sel)
            df_combo = calcular_pareto(df_desde_freq_map(mapa_total))
            df_combo_pri = obtener_df_priorizado(df_combo)

            st.subheader("📊 Pareto general (combinado)")
            dibujar_pareto(df_combo, f"Pareto general – {', '.join(seleccion_port)}")
            st.caption(f"Total de respuestas tratadas en el Pareto general: {int(df_combo['frecuencia'].sum())}")

            st.markdown("**Problemáticas priorizadas del combinado**")
            st.dataframe(
                df_combo_pri[["descriptor", "categoria", "frecuencia", "porcentaje"]],
                use_container_width=True,
                hide_index=True
            )

            col_excel, col_pdf, col_img1, col_img2 = st.columns(4)
            with col_excel:
                st.download_button(
                    "📥 Excel del Pareto general",
                    exportar_excel_con_grafico(df_combo, "Pareto general"),
                    file_name="Pareto_combinado_portafolio.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_pareto_general"
                )
            with col_pdf:
                pdf_simple = generar_pdf_pareto_simple("Pareto general portafolio", df_combo)
                if pdf_simple:
                    st.download_button(
                        "📥 PDF del Pareto general",
                        data=pdf_simple,
                        file_name="Pareto_general_portafolio.pdf",
                        mime="application/pdf",
                        key="pdf_pareto_general"
                    )
            with col_img1:
                st.download_button(
                    "📥 Imagen completo",
                    data=_pareto_png(df_combo, "Pareto general completo", solo_priorizados=False),
                    file_name="Pareto_general_completo.png",
                    mime="image/png",
                    key="png_pareto_general_full"
                )
            with col_img2:
                st.download_button(
                    "📥 Imagen priorizado",
                    data=_pareto_png(df_combo, "Pareto general priorizado", solo_priorizados=True),
                    file_name="Pareto_general_priorizado.png",
                    mime="image/png",
                    key="png_pareto_general_prior"
                )

        st.divider()
        st.markdown("### 📁 Detalle de cada Pareto almacenado")

        for nombre, mapa in list(port.items()):
            with st.expander(f"{nombre}", expanded=False):
                dfp_edit = pd.DataFrame(
                    [{"descriptor": d, "frecuencia": int(f)} for d, f in mapa.items()]
                ).sort_values("descriptor").reset_index(drop=True)

                if dfp_edit.empty:
                    st.info("Este Pareto no tiene datos.")
                    continue

                st.markdown("**Editar filas del Pareto guardado**")
                st.caption(
                    "Aquí puedes corregir descriptores mal escritos, cambiar frecuencias o eliminar filas. "
                    "Luego guarda los cambios."
                )

                dfp_edit_new = st.data_editor(
                    dfp_edit,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"editor_port_{nombre}",
                    column_config={
                        "descriptor": st.column_config.TextColumn("Descriptor", width="large"),
                        "frecuencia": st.column_config.NumberColumn("Frecuencia", min_value=0, step=1)
                    }
                )

                dfp_edit_new["descriptor"] = dfp_edit_new["descriptor"].astype(str).str.strip()
                dfp_edit_new["frecuencia"] = pd.to_numeric(
                    dfp_edit_new["frecuencia"], errors="coerce"
                ).fillna(0).astype(int)

                dfp_edit_new = dfp_edit_new[
                    (dfp_edit_new["descriptor"] != "") &
                    (dfp_edit_new["descriptor"].str.lower() != "nan") &
                    (dfp_edit_new["frecuencia"] > 0)
                ].copy()

                freq_map_editado = dict(
                    zip(dfp_edit_new["descriptor"], dfp_edit_new["frecuencia"])
                )

                col_guardar_edit, col_reset_edit = st.columns(2)

                with col_guardar_edit:
                    if st.button(f"💾 Guardar cambios en '{nombre}'", key=f"save_edit_{nombre}", use_container_width=True):
                        st.session_state["portafolio"][nombre] = normalizar_freq_map(freq_map_editado)
                        sheets_guardar_pareto(nombre, freq_map_editado, sobrescribir=True)
                        st.success(f"Se actualizaron los datos del Pareto '{nombre}'.")
                        st.rerun()

                with col_reset_edit:
                    if st.button(f"↩️ Revertir cambios de '{nombre}'", key=f"reset_edit_{nombre}", use_container_width=True):
                        st.rerun()

                dfp = calcular_pareto(df_desde_freq_map(freq_map_editado if freq_map_editado else mapa))
                dfp_pri = obtener_df_priorizado(dfp)

                st.divider()
                dibujar_pareto(dfp, nombre)
                st.caption(f"Total de respuestas tratadas: {int(dfp['frecuencia'].sum()) if not dfp.empty else 0}")

                st.markdown("**Problemáticas priorizadas**")
                st.dataframe(
                    dfp_pri[["descriptor", "categoria", "frecuencia", "porcentaje"]],
                    use_container_width=True,
                    hide_index=True
                )

                colA, colB, colC, colD, colE = st.columns([1, 1, 1, 1, 2])

                with colA:
                    st.download_button(
                        "📥 Excel",
                        exportar_excel_con_grafico(dfp, nombre),
                        file_name=f"Pareto_{nombre}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_{nombre}"
                    )

                with colB:
                    st.download_button(
                        "🖼️ Completo",
                        data=_pareto_png(dfp, f"Pareto completo – {nombre}", solo_priorizados=False),
                        file_name=f"Pareto_Completo_{nombre}.png",
                        mime="image/png",
                        key=f"png_full_{nombre}"
                    )

                with colC:
                    st.download_button(
                        "🖼️ Priorizado",
                        data=_pareto_png(dfp, f"Pareto priorizado – {nombre}", solo_priorizados=True),
                        file_name=f"Pareto_Priorizado_{nombre}.png",
                        mime="image/png",
                        key=f"png_prior_{nombre}"
                    )

                with colD:
                    if st.button(f"🗑️ Eliminar '{nombre}'", key=f"del_{nombre}"):
                        del st.session_state["portafolio"][nombre]
                        ok = sheets_eliminar_pareto(nombre)
                        if ok:
                            st.success(f"El Pareto '{nombre}' fue eliminado del sistema y de Google Sheets.")
                        else:
                            st.warning(f"El Pareto '{nombre}' se eliminó localmente, pero no pudo borrarse en Sheets.")
                        st.rerun()

                with colE:
                    try:
                        pop = st.popover("📄 Informe PDF de este Pareto")
                    except Exception:
                        pop = st.expander("📄 Informe PDF de este Pareto", expanded=False)

                    with pop:
                        nombre_inf_ind = st.text_input(
                            "Nombre del informe",
                            value=f"{nombre}",
                            key=f"inf_nom_{nombre}"
                        )
                        desgloses_ind = ui_desgloses(
                            dfp_pri["descriptor"].tolist(),
                            key_prefix=f"inf_{nombre}"
                        )
                        if st.button("Generar PDF", key=f"btn_inf_{nombre}"):
                            pdf_bytes = generar_pdf_informe(nombre_inf_ind, dfp, desgloses_ind)
                            if pdf_bytes:
                                st.download_button(
                                    "⬇️ Descargar PDF",
                                    data=pdf_bytes,
                                    file_name=f"informe_{nombre.lower().replace(' ', '_')}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_inf_{nombre}",
                                )

# ---------------------------------------------------------------------------
# TAB 3 — Informe unificado
# ---------------------------------------------------------------------------
with tab_unificado:
    st.subheader("📄 Informe PDF (unificado)")
    port = st.session_state["portafolio"]

    if not port:
        st.info("Guarda al menos un Pareto para generar el informe unificado.")
    else:
        nombres = list(port.keys())
        seleccion = st.multiselect(
            "Selecciona los Paretos a incluir en el informe unificado",
            options=nombres,
            default=nombres,
            key="multi_unificado"
        )

        if seleccion:
            mapas = [port[n] for n in seleccion]
            mapa_total = combinar_maps(mapas)
            df_uni = calcular_pareto(df_desde_freq_map(mapa_total))
            df_uni_pri = obtener_df_priorizado(df_uni)

            st.subheader("📊 Vista previa Pareto Unificado")
            dibujar_pareto(df_uni, "Pareto Unificado")
            st.caption(f"Total de respuestas tratadas: {int(df_uni['frecuencia'].sum())}")

            st.markdown("**Problemáticas priorizadas del unificado**")
            st.dataframe(
                df_uni_pri[["descriptor", "categoria", "frecuencia", "porcentaje"]],
                use_container_width=True,
                hide_index=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "📥 Imagen Pareto completo",
                    data=_pareto_png(df_uni, "Pareto Unificado completo", solo_priorizados=False),
                    file_name="Pareto_Unificado_Completo.png",
                    mime="image/png"
                )
            with c2:
                st.download_button(
                    "📥 Imagen Pareto priorizado",
                    data=_pareto_png(df_uni, "Pareto Unificado priorizado", solo_priorizados=True),
                    file_name="Pareto_Unificado_Priorizado.png",
                    mime="image/png"
                )
            with c3:
                st.download_button(
                    "📥 Excel con gráfico",
                    exportar_excel_con_grafico(df_uni, "Pareto Unificado"),
                    file_name="Pareto_Unificado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            desgloses_uni = ui_desgloses(df_uni_pri["descriptor"].tolist(), key_prefix="uni")

            if st.button("📄 Generar Informe PDF (Unificado)", type="primary"):
                pdf_bytes = generar_pdf_informe("Pareto Unificado", df_uni, desgloses_uni)
                if pdf_bytes:
                    st.download_button(
                        label="📥 Descargar Informe PDF (Unificado)",
                        data=pdf_bytes,
                        file_name="Informe_Pareto_Unificado.pdf",
                        mime="application/pdf"
                    )

# ---------------------------------------------------------------------------
# TAB 4 — MIC MAC
# ---------------------------------------------------------------------------
with tab_micmac:
    st.subheader("🧩 Generador MIC MAC (no se incluye en los PDF)")
    st.caption("Puedes construir MIC MAC desde el Pareto actual, desde un Pareto guardado o desde un unificado.")

    fuentes = ["Pareto actual en edición"]
    if st.session_state.get("portafolio"):
        fuentes += [f"Portafolio: {k}" for k in st.session_state["portafolio"].keys()]
        fuentes.append("Unificado desde portafolio")

    fuente_sel = st.selectbox("Fuente de datos para MIC MAC", options=fuentes, index=0)

    df_source = pd.DataFrame()
    source_name = fuente_sel

    if fuente_sel == "Pareto actual en edición":
        freq_map_actual = st.session_state.get("freq_map", {})
        df_source = calcular_pareto(df_desde_freq_map(freq_map_actual))

    elif fuente_sel.startswith("Portafolio: "):
        nom = fuente_sel.replace("Portafolio: ", "", 1)
        mapa = st.session_state["portafolio"].get(nom, {})
        df_source = calcular_pareto(df_desde_freq_map(mapa))
        source_name = nom

    elif fuente_sel == "Unificado desde portafolio":
        port = st.session_state.get("portafolio", {})
        if port:
            nombres = list(port.keys())
            seleccion_um = st.multiselect(
                "Selecciona los Paretos a combinar para MIC MAC unificado",
                options=nombres,
                default=nombres,
                key="micmac_unificado_select"
            )
            if seleccion_um:
                mapa_total = combinar_maps([port[n] for n in seleccion_um])
                df_source = calcular_pareto(df_desde_freq_map(mapa_total))
                source_name = " + ".join(seleccion_um)
            else:
                st.info("Selecciona al menos un Pareto del portafolio.")
                df_source = pd.DataFrame()

    if df_source.empty:
        st.info("No hay datos disponibles para construir MIC MAC.")
    else:
        ui_micmac(source_name, df_source, key_prefix=f"mic_{source_name.replace(' ', '_')}")

# ============================================================================
# ============================== PARTE 12/12 ================================
# ======================== Créditos y limpieza final =========================
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align:center; font-size:14px; color:gray;">
Desarrollado para la Estrategia <b>Sembremos Seguridad</b><br>
Aplicación de análisis Pareto 80/20 con Google Sheets + ReportLab + MIC MAC<br>
Versión 2026 ⚙️
</div>
""", unsafe_allow_html=True)

for key in ["sheet_url_loaded", "reset_after_save"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.toast("✅ App lista. Puedes generar, guardar y eliminar Paretos, exportar imágenes y construir MIC MAC.", icon="✅")






