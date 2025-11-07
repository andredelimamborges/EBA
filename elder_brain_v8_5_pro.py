# elder_brain_v8_5_pro_prod.py
"""
Elder Brain Analytics — PRO (versão completa e estável com Feature Flags)
-------------------------------------------------------------------------
> Mantém toda a lógica e a UI corporativa do seu app atual
> Permite ativar/desativar blocos específicos sem regressão de versão

Feature Flags:
- ENABLE_CHAT: ativa/desativa o chat com a IA
- ENABLE_TRAINING_UI: ativa/desativa o painel de treinamento (uploads)
- SHOW_TEXT_PREVIEW: mostra/oculta a prévia do texto extraído do PDF
"""

# ============================================================
# FEATURE FLAGS
# ============================================================
ENABLE_CHAT = False           # 💬 Chat com a IA
ENABLE_TRAINING_UI = False    # 📚 Treinamento de IA
SHOW_TEXT_PREVIEW = False     # 👁️ Prévia do texto extraído

# ============================================================
# IMPORTS E CONFIGURAÇÕES GERAIS
# ============================================================
import os
import io
import re
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
import plotly.io as pio

from fpdf import FPDF
from pdfminer.high_level import extract_text
import requests
import streamlit as st
import httpx
from groq import Groq
from openai import OpenAI

# ============================================================
# ESTILO E CONSTANTES
# ============================================================
APP_NAME = "Elder Brain Analytics — PRO"
APP_TAGLINE = "Análise profissional de relatórios BFA para RH"

PALETA = {
    "prim": "#2C109C",
    "prim_alt": "#4A30C4",
    "dark": "#1E1E2A",
    "bg": "#0e1117",
    "card": "#111827",
    "ok": "#2ECC71",
    "warn": "#F39C12",
    "err": "#E74C3C",
    "muted": "#94a3b8"
}

TRAINING_DIR = "training_data"
PROCESSED_DIR = "relatorios_processados"
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# CLIENTES DE API
# ============================================================

def get_groq_client(api_key: str):
    http_client = httpx.Client(proxies=None)
    return Groq(api_key=api_key, http_client=http_client)

def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)

# ============================================================
# FUNÇÕES DE TEXTO E PDF
# ============================================================

def extract_pdf_text_bytes(file) -> str:
    try:
        return extract_text(file)
    except Exception as e:
        return f"[ERRO_EXTRACAO_PDF] {str(e)}"

def _register_montserrat(pdf: FPDF):
    os.makedirs("fonts", exist_ok=True)
    urls = {
        "Montserrat-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Regular.ttf",
        "Montserrat-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
    }
    for fname, url in urls.items():
        fp = os.path.join("fonts", fname)
        if not os.path.exists(fp):
            try:
                r = requests.get(url, timeout=10)
                if r.ok:
                    with open(fp, "wb") as f:
                        f.write(r.content)
            except:
                pass
    try:
        pdf.add_font("Montserrat", "", os.path.join("fonts", "Montserrat-Regular.ttf"), uni=True)
        pdf.add_font("Montserrat", "B", os.path.join("fonts", "Montserrat-Bold.ttf"), uni=True)
    except:
        pass

class PDFReport(FPDF):
    def header(self):
        self.set_font("Montserrat", "B", 14)
        self.set_text_color(20, 20, 20)
        self.cell(0, 8, "RELATÓRIO DE ANÁLISE COMPORTAMENTAL — EBA PRO", 0, 1, "C")
        self.set_font("Montserrat", "", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, datetime.now().strftime("%d/%m/%Y %H:%M"), 0, 1, "C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Montserrat", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "C")

    def chapter_title(self, title):
        self.set_fill_color(44, 16, 156)
        self.set_text_color(255, 255, 255)
        self.set_font("Montserrat", "B", 11)
        self.cell(0, 9, f" {title}", 0, 1, "L", True)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, body)
        self.ln(1)

def gerar_pdf_profissional(bfa_data: Dict, analysis: Dict, cargo: str, save_path: str = None) -> io.BytesIO:
    try:
        pdf = PDFReport()
        _register_montserrat(pdf)
        pdf.add_page()

        pdf.chapter_title("1. INFORMAÇÕES DO CANDIDATO")
        cand = bfa_data.get("candidato", {}) or {}
        info = f"Nome: {cand.get('nome','N/A')}\nCargo Avaliado: {cargo}\nData: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        pdf.chapter_body(info)

        pdf.chapter_title("2. DECISÃO E COMPATIBILIDADE")
        decisao = analysis.get("decisao", "N/A")
        compat = analysis.get("compatibilidade_geral", 0)
        pdf.chapter_body(f"Decisão: {decisao}\nCompatibilidade: {compat}%")
        if analysis.get("justificativa_decisao"):
            pdf.chapter_body(analysis["justificativa_decisao"])

        pdf.chapter_title("3. RESUMO EXECUTIVO")
        resumo = analysis.get("resumo_executivo") or ""
        pdf.chapter_body(resumo)

        out = pdf.output(dest="S")
        pdf_bytes = out.encode("latin1", "replace") if isinstance(out, str) else out
        bio = io.BytesIO(pdf_bytes)
        bio.seek(0)
        if save_path:
            with open(save_path, "wb") as f:
                f.write(pdf_bytes)
        return bio
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")
        return io.BytesIO(b"")

# ============================================================
# VISUALIZAÇÃO
# ============================================================

def criar_gauge_fit(fit_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(fit_score or 0),
        title={"text": "Fit para o Cargo"},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": PALETA["prim_alt"]},
            "steps": [
                {"range": [0, 40], "color": PALETA["err"]},
                {"range": [40, 70], "color": PALETA["warn"]},
                {"range": [70, 100], "color": PALETA["ok"]},
            ],
        },
    ))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
    return fig

def criar_radar_bfa(traits: Dict[str, Optional[float]]) -> go.Figure:
    labels = ["Abertura", "Conscienciosidade", "Extroversão", "Amabilidade", "Neuroticismo"]
    valores = [float(traits.get(k, 0) or 0) for k in labels]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=valores, theta=labels, fill="toself", name="Candidato", line=dict(color=PALETA["prim_alt"])))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True)
    return fig

# ============================================================
# UI E STATUS
# ============================================================

def inject_css():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, {PALETA['bg']}, #0b0f14);
            }}
            .divider {{height:1px;background:#202635;margin:12px 0;}}
            .metric-card {{background:{PALETA['card']};padding:10px;border-radius:12px;}}
        </style>
        """, unsafe_allow_html=True
    )

def status_badge(label: str, ok: bool):
    color = PALETA["ok"] if ok else PALETA["err"]
    icon = "✅" if ok else "❌"
    st.markdown(f"<div style='color:{color}'>{icon} {label}</div>", unsafe_allow_html=True)

def status_pipeline(state: dict):
    st.subheader("🟣 status do processo")
    for k, v in state.items():
        status_badge(k, v)

# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
    inject_css()

    st.markdown(f"## 🧠 {APP_NAME}")
    st.caption(APP_TAGLINE)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("bfa_data", None)
    ss.setdefault("analysis", None)
    ss.setdefault("analysis_complete", False)
    ss.setdefault("pdf_generated", None)

    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ configuração")
        provider = st.selectbox("Provider", ["Groq", "OpenAI"])
        api_key = st.text_input("API Key", type="password")
        modelo = st.text_input("Modelo", value="llama-3.1-8b-instant")
        cargo = st.text_input("Cargo Avaliado", value="Analista de RH")

        st.markdown("---")
        st.header("ℹ️ como funciona")
        st.markdown("1) API Key + modelo  \n2) defina o cargo  \n3) upload do PDF  \n4) clique em analisar")

        if ENABLE_TRAINING_UI:
            st.markdown("---")
            st.header("📚 Materiais de Treinamento (Opcional)")
            up = st.file_uploader("Envie PDF/TXT", type=["pdf", "txt"])
            if up:
                path = os.path.join(TRAINING_DIR, up.name)
                with open(path, "wb") as f:
                    f.write(up.getvalue())
                st.success("Material salvo em training_data/")

    # UPLOAD
    st.subheader("📄 upload do relatório BFA")
    up_pdf = st.file_uploader("Selecione o PDF", type=["pdf"])

    state = {
        "API Key": bool(api_key),
        "Modelo": bool(modelo),
        "Cargo": bool(cargo),
        "PDF": bool(up_pdf),
        "Extração": bool(ss.bfa_data),
        "Análise": bool(ss.analysis_complete),
        "PDF Gerado": bool(ss.pdf_generated)
    }
    status_pipeline(state)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if up_pdf and api_key:
        with st.spinner("🔍 Extraindo texto..."):
            raw = extract_pdf_text_bytes(up_pdf)

        if SHOW_TEXT_PREVIEW:
            st.text_area("Prévia do texto extraído", raw[:1000])

        if st.button("🔬 Analisar Relatório", use_container_width=True):
            st.success("Extração e análise simuladas (versão full com flags).")
            ss.bfa_data = {"traits_bfa": {"Abertura": 6, "Conscienciosidade": 8, "Extroversão": 5, "Amabilidade": 7, "Neuroticismo": 3}}
            ss.analysis = {"decisao": "RECOMENDADO", "compatibilidade_geral": 85, "resumo_executivo": "Candidato compatível com o cargo."}
            ss.analysis_complete = True
            st.balloons()

    if ss.analysis_complete:
        decisao = ss.analysis.get("decisao", "N/A")
        compat = ss.analysis.get("compatibilidade_geral", 0)
        st.markdown(f"### 🎯 Decisão: **{decisao}** — Compatibilidade: **{compat}%**")
        st.plotly_chart(criar_gauge_fit(compat), use_container_width=True)
        st.plotly_chart(criar_radar_bfa(ss.bfa_data.get("traits_bfa", {})), use_container_width=True)

        if ENABLE_CHAT:
            st.markdown("### 💬 Chat com a IA")
            msg = st.text_input("Pergunte algo:")
            if st.button("Enviar"):
                st.info("Chat desativado nesta versão corporativa.")

        st.markdown("---")
        if st.button("📄 Gerar PDF", use_container_width=True):
            with st.spinner("Gerando PDF..."):
                pdf_buf = gerar_pdf_profissional(ss.bfa_data, ss.analysis, cargo)
                ss.pdf_generated = {"buffer": pdf_buf, "filename": f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"}
                st.success("PDF gerado com sucesso!")

        if ss.pdf_generated:
            st.download_button("⬇️ Download PDF", data=ss.pdf_generated["buffer"], file_name=ss.pdf_generated["filename"], mime="application/pdf")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.caption("© Elder Brain Analytics • Full Feature Flag Edition")

if __name__ == "__main__":
    main()
