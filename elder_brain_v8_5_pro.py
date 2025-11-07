# elder_brain_v8_5_pro.py
"""
Elder Brain Analytics — PRO (UI Corporativa • sem Chat/Preview/Treinamento)
- Melhorias gráficas e layout corporativo
- Removido Chat com IA
- Removido Preview de texto
- Removido upload de materiais de treinamento
- Adicionado 'Como funciona'
- Status realmente funcional (pipeline)
"""

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
from fpdf import FPDF
import requests

from pdfminer.high_level import extract_text
import streamlit as st
from groq import Groq

# =============================================================================
# CONFIG & CONSTANTES
# =============================================================================

APP_NAME = "Elder Brain Analytics — PRO"
APP_TAGLINE = "Análise profissional de relatórios BFA para RH"
PALETA = {
    "prim": "#2C109C",      # Roxo corporativo
    "prim_alt": "#4A30C4",
    "dark": "#1E1E2A",
    "bg": "#0e1117",
    "card": "#111827",
    "ok": "#2ECC71",
    "warn": "#F39C12",
    "err": "#E74C3C",
    "muted": "#94a3b8"
}

TRAINING_DIR = "training_data"          # mantido, mas não exposto no UI
PROCESSED_DIR = "relatorios_processados"
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Sugeridos (campo é livre, mas ajudamos)
MODELOS_SUGERIDOS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
]

# =============================================================================
# PROMPTS
# =============================================================================

EXTRACTION_PROMPT = """Você é um especialista em análise de relatórios BFA (Big Five Analysis) para seleção de talentos.

Sua tarefa: extrair dados do relatório abaixo e retornar APENAS um JSON válido, sem texto adicional.

ESTRUTURA OBRIGATÓRIA:
{
  "candidato": {
    "nome": "string ou null",
    "cargo_avaliado": "string ou null"
  },
  "traits_bfa": {
    "Abertura": número 0-10 ou null,
    "Conscienciosidade": número 0-10 ou null,
    "Extroversao": número 0-10 ou null,
    "Amabilidade": número 0-10 ou null,
    "Neuroticismo": número 0-10 ou null
  },
  "competencias_ms": [
    {"nome": "string", "nota": número, "classificacao": "string"}
  ],
  "facetas_relevantes": [
    {"nome": "string", "percentil": número, "interpretacao": "string resumida"}
  ],
  "indicadores_saude_emocional": {
    "ansiedade": número 0-100 ou null,
    "irritabilidade": número 0-100 ou null,
    "estado_animo": número 0-100 ou null,
    "impulsividade": número 0-100 ou null
  },
  "potencial_lideranca": "BAIXO" | "MÉDIO" | "ALTO" ou null,
  "integridade_fgi": número 0-100 ou null,
  "resumo_qualitativo": "texto do resumo presente no relatório",
  "pontos_fortes": ["lista de 3-5 pontos"],
  "pontos_atencao": ["lista de 2-4 pontos"],
  "fit_geral_cargo": número 0-100
}

REGRAS:
1) Converta percentis Big Five para 0-10 quando necessário (ex.: 60 -> 6.0).
2) Extraia TODAS as competências mencionadas.
3) Use null quando não houver dado confiável.
4) 'resumo_qualitativo' é o trecho original do relatório.
5) 'fit_geral_cargo' (0-100) baseado no cargo: {cargo}.

RELATÓRIO (texto):
\"\"\"{text}\"\"\"

Retorne APENAS o JSON, sem markdown, sem explicações.
"""

ANALYSIS_PROMPT = """Você é um consultor sênior de RH especializado em análise comportamental e fit cultural.

Baseado nos dados extraídos do BFA, faça uma análise profissional para o cargo: {cargo}

DADOS DO CANDIDATO:
{json_data}

PERFIL IDEAL (gerado dinamicamente):
{perfil_cargo}

Retorne **apenas JSON**:
{
  "compatibilidade_geral": número 0-100,
  "decisao": "RECOMENDADO" | "RECOMENDADO COM RESSALVAS" | "NÃO RECOMENDADO",
  "justificativa_decisao": "parágrafo explicativo",
  "analise_tracos": {
    "Abertura": "texto",
    "Conscienciosidade": "texto",
    "Extroversao": "texto",
    "Amabilidade": "texto",
    "Neuroticismo": "texto"
  },
  "competencias_criticas": [
    {"competencia": "nome", "avaliacao": "texto", "status": "ATENDE" | "PARCIAL" | "NÃO ATENDE"}
  ],
  "saude_emocional_contexto": "parágrafo",
  "recomendacoes_desenvolvimento": ["itens..."],
  "cargos_alternativos": [{"cargo": "nome", "justificativa": "texto"}],
  "resumo_executivo": "100-150 palavras"
}
"""

# =============================================================================
# FONTE & PDF
# =============================================================================

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
                r = requests.get(url, timeout=12)
                if r.ok:
                    with open(fp, "wb") as f:
                        f.write(r.content)
            except Exception:
                pass
    try:
        pdf.add_font("Montserrat", "", os.path.join("fonts", "Montserrat-Regular.ttf"), uni=True)
        pdf.add_font("Montserrat", "B", os.path.join("fonts", "Montserrat-Bold.ttf"), uni=True)
    except Exception:
        pass

class PDFReport(FPDF):
    def header(self):
        self.set_font('Montserrat', 'B', 14)
        self.set_text_color(20, 20, 20)
        self.cell(0, 8, 'RELATÓRIO DE ANÁLISE COMPORTAMENTAL — EBA PRO', 0, 1, 'C')
        self.set_font('Montserrat', '', 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, datetime.now().strftime('%d/%m/%Y %H:%M'), 0, 1, 'C')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Montserrat', '', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_fill_color(44, 16, 156)  # roxo corporativo
        self.set_text_color(255, 255, 255)
        self.set_font('Montserrat', 'B', 11)
        self.cell(0, 9, f" {title}", 0, 1, 'L', True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln(1)

def gerar_pdf_profissional(bfa_data: Dict, analysis: Dict, cargo: str, save_path: str = None) -> io.BytesIO:
    try:
        pdf = PDFReport()
        _register_montserrat(pdf)
        pdf.add_page()

        # 1. CANDIDATO
        pdf.chapter_title('1. INFORMACOES DO CANDIDATO')
        cand = bfa_data.get('candidato', {}) or {}
        info = f"Nome: {cand.get('nome', 'Nao informado')}\nCargo Avaliado: {cargo}\nData da Análise: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        pdf.chapter_body(info)

        # 2. DECISÃO
        pdf.chapter_title('2. DECISAO E COMPATIBILIDADE')
        decisao = analysis.get('decisao', 'N/A')
        compat = float(analysis.get('compatibilidade_geral', 0) or 0)
        blocos = {
            'RECOMENDADO': (46, 204, 113),
            'RECOMENDADO COM RESSALVAS': (241, 196, 15),
            'NÃO RECOMENDADO': (231, 76, 60)
        }
        r, g, b = blocos.get(decisao, (150, 150, 150))
        pdf.set_fill_color(r, g, b)
        pdf.set_font('Montserrat', 'B', 13)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, f'  {decisao}  ', 0, 1, 'C', True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Montserrat', 'B', 11)
        pdf.cell(0, 7, f'Compatibilidade: {compat:.0f}%', 0, 1, 'C')
        just = analysis.get('justificativa_decisao', '')
        if just:
            pdf.chapter_body(just)

        # 3. RESUMO EXECUTIVO
        pdf.chapter_title('3. RESUMO EXECUTIVO')
        resumo = analysis.get('resumo_executivo') or analysis.get('justificativa_decisao', '')
        if resumo:
            pdf.chapter_body(resumo)

        # 4. BIG FIVE
        pdf.chapter_title('4. TRACOS DE PERSONALIDADE (BIG FIVE)')
        traits = bfa_data.get('traits_bfa', {}) or {}
        for k, v in traits.items():
            if v is not None:
                pdf.set_font('Montserrat', 'B', 10)
                pdf.cell(60, 6, f'{k}:', 0, 0)
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, f'{float(v):.1f}/10', 0, 1)
        pdf.ln(1)
        a_tracos = analysis.get('analise_tracos', {}) or {}
        for k, txt in a_tracos.items():
            if txt:
                pdf.set_font('Montserrat', '', 9)
                pdf.multi_cell(0, 5, f'{k}: {txt}')
        pdf.ln(1)

        # 5. COMPETÊNCIAS CRÍTICAS
        pdf.chapter_title('5. COMPETENCIAS CRITICAS')
        comp_crit = analysis.get('competencias_criticas', []) or []
        for c in comp_crit:
            status = c.get('status', '')
            simbolo = 'OK' if status == 'ATENDE' else ('PARC' if status == 'PARCIAL' else 'NAO')
            pdf.set_font('Montserrat', 'B', 10)
            pdf.cell(0, 6, f"[{simbolo}] {c.get('competencia','')}: {status}", 0, 1)
            if c.get('avaliacao'):
                pdf.set_font('Arial', '', 9)
                pdf.multi_cell(0, 5, "   " + c['avaliacao'])
                pdf.ln(0.5)

        # 6. SAÚDE EMOCIONAL
        pdf.chapter_title('6. SAUDE EMOCIONAL E RESILIENCIA')
        ctx = analysis.get('saude_emocional_contexto', '')
        if ctx:
            pdf.chapter_body(ctx)
        ind = bfa_data.get('indicadores_saude_emocional', {}) or {}
        for nome, valor in ind.items():
            if valor is not None:
                pdf.set_font('Arial', '', 9)
                pdf.cell(70, 5, f'{nome.replace("_"," ").title()}:', 0, 0)
                pdf.cell(0, 5, f'{float(valor):.0f}/100', 0, 1)
        pdf.ln(1)

        # 7. PONTOS
        pf = bfa_data.get('pontos_fortes', []) or []
        if pf:
            pdf.chapter_title('7. PONTOS FORTES')
            for t in pf:
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 5, f'+ {t}')
        pa = bfa_data.get('pontos_atencao', []) or []
        if pa:
            pdf.chapter_title('8. PONTOS DE ATENCAO')
            for t in pa:
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 5, f'! {t}')

        # 8. RECOMENDAÇÕES
        pdf.add_page()
        pdf.chapter_title('9. RECOMENDACOES DE DESENVOLVIMENTO')
        recs = analysis.get('recomendacoes_desenvolvimento', []) or []
        for i, rtxt in enumerate(recs, 1):
            pdf.set_font('Montserrat', 'B', 10)
            pdf.cell(10, 6, f'{i}.', 0, 0)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, rtxt)
            pdf.ln(0.5)

        # 9. CARGOS ALTERNATIVOS
        alt = analysis.get('cargos_alternativos', []) or []
        if alt:
            pdf.chapter_title('10. CARGOS ALTERNATIVOS')
            for c in alt:
                nome = c.get('cargo', '')
                j = c.get('justificativa', '')
                if nome:
                    pdf.set_font('Montserrat', 'B', 10)
                    pdf.cell(0, 6, f"- {nome}", 0, 1)
                    if j:
                        pdf.set_font('Arial', '', 9)
                        pdf.multi_cell(0, 5, "   " + j)
                        pdf.ln(0.5)

        # Saída
        out = pdf.output(dest='S')
        if out is None:
            raise ValueError("PDF output None")
        pdf_bytes = out.encode('latin1', 'replace') if isinstance(out, str) else out

        bio = io.BytesIO(pdf_bytes)
        bio.seek(0)
        if save_path:
            with open(save_path, "wb") as f:
                f.write(pdf_bytes)
        return bio
    except Exception as e:
        st.error(f"Erro crítico na geração do PDF: {e}")
        return io.BytesIO(b'')

# =============================================================================
# AUXILIARES DE IA & VISUALIZAÇÃO
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_groq_client_cached(token: str):
    if not token:
        raise RuntimeError("Informe a Groq API Key")
    try:
        return Groq(api_key=token)
    except Exception as e:
        raise RuntimeError(f"Erro ao criar cliente Groq: {e}")

def extract_pdf_text_bytes(file) -> str:
    try:
        return extract_text(file)
    except Exception as e:
        return f"[ERRO_EXTRACAO_PDF] {str(e)}"

def gerar_perfil_cargo_dinamico(cargo: str) -> Dict:
    return {
        "traits_ideais": {
            "Abertura": (5, 8),
            "Conscienciosidade": (6, 9),
            "Extroversao": (4, 8),
            "Amabilidade": (5, 8),
            "Neuroticismo": (0, 5)
        },
        "competencias_criticas": ["Adaptabilidade", "Comunicação", "Trabalho em Equipe", "Resolução de Problemas"],
        "descricao": f"Perfil profissional adequado para {cargo}."
    }

def extract_bfa_data(text: str, cargo: str, model_id: str, token: str, temperature: float, max_tokens: int) -> Tuple[Optional[Dict], str]:
    if not token:
        return None, "Groq API Key não informada"
    if not model_id.strip():
        return None, "Modelo não informado"

    try:
        client = get_groq_client_cached(token)
    except Exception as e:
        return None, f"[Erro cliente] {e}"

    prompt = EXTRACTION_PROMPT.format(text=text[:10000], cargo=cargo)
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"},
            model=model_id.strip(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        raw = resp.choices[0].message.content.strip()

        # Tenta achar JSON
        m = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return parsed, raw
            except json.JSONDecodeError:
                for cand in re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', raw, re.DOTALL):
                    try:
                        return json.loads(cand), raw
                    except json.JSONDecodeError:
                        continue
        return None, f"Nenhum JSON válido encontrado: {raw[:500]}..."
    except Exception as e:
        err = f"[Erro Groq] {str(e)}"
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            try:
                j = json.loads(e.response.text)
                if 'error' in j and 'message' in j['error']:
                    err += f" - {j['error']['message']}"
            except:
                err += f" - Resposta: {e.response.text}"
        return None, err

def analyze_bfa_data(
    bfa_data: Dict,
    cargo: str,
    model_id: str,
    token: str,
    temperature: float,
    max_tokens: int
) -> Tuple[Optional[Dict], str]:
    if not token:
        return None, "Groq API Key não informada"
    if not model_id.strip():
        return None, "Modelo não informado"

    try:
        client = get_groq_client_cached(token)
    except Exception as e:
        return None, f"[Erro cliente] {e}"

    perfil = gerar_perfil_cargo_dinamico(cargo)
    prompt = ANALYSIS_PROMPT.format(
        cargo=cargo,
        json_data=json.dumps(bfa_data, ensure_ascii=False, indent=2),
        perfil_cargo=json.dumps(perfil, ensure_ascii=False, indent=2),
    )

    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"},
            model=model_id.strip(),
            messages=[
                {"role": "system", "content": "Responda ESTRITAMENTE com JSON. Sem markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        raw = resp.choices[0].message.content.strip()

        # Ajuste se vier com texto fora do JSON
        if not raw.startswith("{"):
            raw = re.sub(r'.*?(\{.*\})', r'\1', raw, flags=re.DOTALL).strip()

        parsed = json.loads(raw)
        return parsed, raw
    except Exception as e:
        return None, f"[Erro análise] {e}"

def criar_radar_bfa(traits: Dict[str, Optional[float]], traits_ideais: Dict = None) -> go.Figure:
    labels = ["Abertura", "Conscienciosidade", "Extroversão", "Amabilidade", "Neuroticismo"]
    valores = [float(traits.get(k, 0) or 0) for k in labels]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores, theta=labels, fill='toself', name='Candidato', line=dict(color=PALETA["prim_alt"])
    ))
    if traits_ideais:
        vmin = [traits_ideais.get(k, (0, 10))[0] for k in labels]
        vmax = [traits_ideais.get(k, (0, 10))[1] for k in labels]
        fig.add_trace(go.Scatterpolar(r=vmax, theta=labels, fill='toself', name='Faixa Ideal (Máx)',
                                      line=dict(color='rgba(46,213,115,0.35)'), fillcolor='rgba(46,213,115,0.20)'))
        fig.add_trace(go.Scatterpolar(r=vmin, theta=labels, fill='tonext', name='Faixa Ideal (Mín)',
                                      line=dict(color='rgba(46,213,115,0.25)'), fillcolor='rgba(46,213,115,0.10)'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True, height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=None
    )
    return fig

def criar_grafico_competencias(competencias: List[Dict]) -> Optional[go.Figure]:
    if not competencias:
        return None
    df = pd.DataFrame(competencias)
    df = df.sort_values('nota', ascending=True).tail(15)
    cores = [PALETA["err"] if n < 45 else PALETA["warn"] if n < 55 else PALETA["ok"] for n in df['nota']]
    fig = go.Figure(go.Bar(
        x=df['nota'], y=df['nome'], orientation='h',
        marker=dict(color=cores), text=df['nota'].round(0).astype(int), textposition='outside'
    ))
    fig.update_layout(
        height=560, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=15, t=10, b=10), xaxis_title="Nota", yaxis_title=""
    )
    fig.add_vline(x=45, line_dash="dash", line_color=PALETA["warn"])
    fig.add_vline(x=55, line_dash="dash", line_color=PALETA["ok"])
    return fig

def criar_gauge_fit(fit_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(fit_score or 0),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fit para o Cargo"},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': PALETA["prim_alt"]},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 40], 'color': PALETA["err"]},
                {'range': [40, 70], 'color': PALETA["warn"]},
                {'range': [70, 100], 'color': PALETA["ok"]}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# =============================================================================
# UI HELPERS
# =============================================================================

def badge(label: str, ok: bool, hint: str = ""):
    color = PALETA["ok"] if ok else PALETA["err"]
    icon = "✅" if ok else "⛔"
    st.markdown(
        f"""
        <div style="background:{PALETA['card']};border:1px solid #222;border-radius:10px;padding:10px 12px;margin-bottom:6px;">
            <span style="font-size:14px">{icon} <b>{label}</b></span>
            <span style="color:{PALETA['muted']};font-size:12px;">{ " — " + hint if hint else ""}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def status_pipeline(state: dict):
    st.subheader("🟣 status do processo")
    badge("API Key informada", state.get("has_token", False), "necessária para as chamadas Groq")
    badge("Modelo válido", state.get("has_model", False), "ex.: llama-3.1-8b-instant")
    badge("Cargo definido", state.get("has_role", False))
    badge("PDF carregado", state.get("has_pdf", False))
    badge("Extração concluída", state.get("extracted", False))
    badge("Análise concluída", state.get("analyzed", False))
    badge("PDF gerado", state.get("pdf_ready", False))

def inject_css():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, {PALETA['bg']}, #0b0f14);
            }}
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 3rem;
            }}
            .stTabs [data-baseweb="tab-list"] button {{
                font-weight: 600;
            }}
            .metric-card {{
                background: {PALETA['card']};
                border: 1px solid #222;
                border-radius: 14px;
                padding: 14px;
            }}
            .headline {{
                font-size: 1.7rem;
                font-weight: 800;
                color: #fff;
            }}
            .subtle {{
                color: {PALETA['muted']};
            }}
            .divider {{
                height: 1px; background: #202635; margin: 10px 0 14px 0;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# APP
# =============================================================================

def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    # ---------- HEADER ----------
    st.markdown(f"<div class='headline'>🧠 {APP_NAME}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtle'>{APP_TAGLINE}</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Session state base
    ss = st.session_state
    if "analysis_complete" not in ss: ss.analysis_complete = False
    if "bfa_data" not in ss: ss.bfa_data = None
    if "analysis" not in ss: ss.analysis = None
    if "cargo_final" not in ss: ss.cargo_final = ""
    if "pdf_generated" not in ss: ss.pdf_generated = None
    if "modelo_selecionado" not in ss: ss.modelo_selecionado = "llama-3.1-8b-instant"

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("⚙️ configuração")
        modelo_input = st.text_input(
            "Modelo Groq",
            value=ss.modelo_selecionado,
            placeholder="ex.: llama-3.1-8b-instant",
            help="Sugestões: " + ", ".join(MODELOS_SUGERIDOS)
        )
        if modelo_input:
            ss.modelo_selecionado = modelo_input

        groq_token = st.text_input("Groq API Key", type="password", placeholder="gsk_xxx...")
        temp = st.slider("Temperatura", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.slider("Tokens máximos", 512, 3072, 1500, 128)

        st.markdown("---")
        st.header("🎯 análise")
        cargo_input = st.text_input("Cargo para análise", value=ss.cargo_final, placeholder="ex.: Gerente Comercial")
        if cargo_input:
            ss.cargo_final = cargo_input
            with st.expander("perfil gerado para o cargo"):
                st.json(gerar_perfil_cargo_dinamico(cargo_input))

        st.markdown("---")
        st.header("ℹ️ como funciona")
        st.markdown(
            """
            1) informe a **API Key** e o **modelo** groq  
            2) defina o **cargo** da avaliação  
            3) faça **upload do PDF BFA**  
            4) clique em **ANALISAR RELATÓRIO**  
            5) confira **gráficos e decisão**  
            6) gere o **PDF executivo**
            """.strip()
        )

    # ---------- UPLOAD ----------
    st.subheader("📄 upload do relatório BFA")
    uploaded_file = st.file_uploader("Carregue o PDF do relatório BFA", type=["pdf"])

    # ---------- STATUS (antes de rodar) ----------
    state_flags = {
        "has_token": bool(groq_token),
        "has_model": bool(ss.modelo_selecionado and ss.modelo_selecionado.strip()),
        "has_role": bool(ss.cargo_final),
        "has_pdf": bool(uploaded_file),
        "extracted": bool(ss.bfa_data),
        "analyzed": bool(ss.analysis_complete and ss.analysis),
        "pdf_ready": bool(ss.pdf_generated),
    }
    status_pipeline(state_flags)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ---------- PROCESSAMENTO ----------
    if uploaded_file:
        if not ss.cargo_final:
            st.error("⚠️ Insira o cargo na sidebar.")
            st.stop()
        if not groq_token:
            st.error("⚠️ Insira a Groq API Key na sidebar.")
            st.stop()
        if not ss.modelo_selecionado.strip():
            st.error("⚠️ Informe um modelo Groq válido.")
            st.stop()

        # Extração
        with st.spinner("🔍 extraindo texto do PDF..."):
            raw_text = extract_pdf_text_bytes(uploaded_file)
        if raw_text.startswith("[ERRO"):
            st.error(raw_text)
            st.stop()

        # Etapa 1
        if st.button("🔬 ANALISAR RELATÓRIO", type="primary", use_container_width=True):
            with st.spinner("Etapa 1/2: Extraindo dados estruturados..."):
                bfa_data, raw_extraction = extract_bfa_data(
                    raw_text, ss.cargo_final, ss.modelo_selecionado, groq_token, temp, max_tokens
                )

            if not bfa_data:
                st.error("❌ Falha na extração de dados")
                with st.expander("ver resposta bruta"):
                    st.code(raw_extraction)
                st.stop()

            st.success("✓ Dados extraídos")
            ss.bfa_data = bfa_data

            # Etapa 2
            with st.spinner("Etapa 2/2: Analisando compatibilidade..."):
                analysis, raw_analysis = analyze_bfa_data(
                    ss.bfa_data, ss.cargo_final, ss.modelo_selecionado, groq_token, temp, max_tokens
                )

            if not analysis:
                st.error("❌ Falha na análise")
                with st.expander("ver resposta bruta"):
                    st.code(raw_analysis)
                st.stop()

            ss.analysis = analysis
            ss.analysis_complete = True
            st.success("✓ Análise concluída!")
            st.rerun()

    # ---------- RESULTADOS ----------
    if ss.analysis_complete and ss.bfa_data and ss.analysis:
        decisao = ss.analysis.get('decisao', 'N/A')
        compat = float(ss.analysis.get('compatibilidade_geral', 0) or 0)
        potencial = ss.bfa_data.get('potencial_lideranca', 'N/A')

        # KPIs
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            col = {"RECOMENDADO": PALETA["ok"], "RECOMENDADO COM RESSALVAS": PALETA["warn"], "NÃO RECOMENDADO": PALETA["err"]}.get(decisao, "#6b7280")
            st.markdown(
                f"<div class='metric-card'><div style='font-weight:700;color:{col};font-size:1.2rem'>{decisao}</div><div class='subtle'>decisão de contratação</div></div>",
                unsafe_allow_html=True
            )
        with c2:
            st.metric("Compatibilidade", f"{compat:.0f}%")
        with c3:
            st.metric("Liderança", potencial)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Big Five", "💼 Competências", "🧘 Saúde Emocional", "📈 Desenvolvimento", "📄 JSON Bruto"])

        with tab1:
            st.subheader("Traços de Personalidade (Big Five)")
            traits = ss.bfa_data.get('traits_bfa', {}) or {}
            perfil = gerar_perfil_cargo_dinamico(ss.cargo_final)
            fig_radar = criar_radar_bfa(traits, perfil.get('traits_ideais', {}))
            st.plotly_chart(fig_radar, use_container_width=True)

            analise_tracos = ss.analysis.get('analise_tracos', {}) or {}
            if analise_tracos:
                with st.expander("análises por traço"):
                    for k, v in analise_tracos.items():
                        st.markdown(f"**{k}** — {v}")

        with tab2:
            st.subheader("Competências")
            competencias = ss.bfa_data.get('competencias_ms', []) or []
            figc = criar_grafico_competencias(competencias)
            if figc:
                st.plotly_chart(figc, use_container_width=True)

            crit = ss.analysis.get('competencias_criticas', []) or []
            if crit:
                st.markdown("**Competências Críticas**")
                for c in crit:
                    status = c.get('status', '')
                    ic = "✅" if status == "ATENDE" else ("⚠️" if status == "PARCIAL" else "❌")
                    st.markdown(f"- {ic} **{c.get('competencia','')}** — {status}")
                    if c.get('avaliacao'):
                        st.caption(c['avaliacao'])

        with tab3:
            st.subheader("Saúde Emocional e Resiliência")
            st.markdown(ss.analysis.get('saude_emocional_contexto', ''))
            indicadores = ss.bfa_data.get('indicadores_saude_emocional', {}) or {}
            if any(v is not None for v in indicadores.values()):
                cols = st.columns(2)
                for i, (nome, valor) in enumerate(indicadores.items()):
                    if valor is not None:
                        with cols[i % 2]:
                            limite = 55
                            delta = (limite - float(valor)) if float(valor) <= limite else (float(valor) - limite)
                            color = 'inverse' if float(valor) <= limite else 'normal'
                            st.metric(nome.replace("_", " ").title(), f"{float(valor):.0f}", f"{delta:.0f}", delta_color=color)

        with tab4:
            st.subheader("Plano de Desenvolvimento")
            recs = ss.analysis.get('recomendacoes_desenvolvimento', []) or []
            if recs:
                for i, r in enumerate(recs, 1):
                    st.markdown(f"**{i}.** {r}")

            pf = ss.bfa_data.get('pontos_fortes', []) or []
            if pf:
                st.markdown("**✅ Pontos Fortes**")
                for t in pf: st.success(f"• {t}")
            pa = ss.bfa_data.get('pontos_atencao', []) or []
            if pa:
                st.markdown("**⚠️ Pontos de Atenção**")
                for t in pa: st.warning(f"• {t}")

        with tab5:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Dados Extraídos**")
                st.json(ss.bfa_data)
            with c2:
                st.markdown("**Análise Completa**")
                st.json(ss.analysis)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Gauge
        st.subheader("🎯 Compatibilidade Geral")
        st.plotly_chart(criar_gauge_fit(compat), use_container_width=True)

        # PDF
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("📄 Relatório em PDF")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Gere um relatório executivo em PDF com a análise completa.")
        with col2:
            if st.button("🔨 Gerar PDF", use_container_width=True, key="btn_pdf"):
                with st.spinner("Gerando PDF..."):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome = ss.bfa_data.get('candidato', {}).get('nome')
                        if not isinstance(nome, str) or not nome.strip():
                            nome = "candidato"
                        nome = re.sub(r'[^\w\s-]', '', nome).strip().replace(' ', '_')
                        pdf_filename = f"relatorio_{nome}_{timestamp}.pdf"
                        pdf_path = os.path.join(PROCESSED_DIR, pdf_filename)
                        buf = gerar_pdf_profissional(ss.bfa_data, ss.analysis, ss.cargo_final, save_path=pdf_path)

                        if buf.getbuffer().nbytes > 100:
                            ss.pdf_generated = {'buffer': buf, 'filename': pdf_filename}
                            st.success(f"✓ PDF gerado: {pdf_filename}")
                            st.rerun()
                        else:
                            st.error("❌ PDF vazio. Tente novamente.")
                    except Exception as e:
                        st.error(f"Erro na geração do PDF: {e}")

        if ss.pdf_generated:
            st.download_button(
                "⬇️ Download do Relatório PDF",
                data=ss.pdf_generated['buffer'].getvalue(),
                file_name=ss.pdf_generated['filename'],
                mime="application/pdf",
                use_container_width=True
            )

    # ---------- FOOTER ----------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.caption("© Elder Brain Analytics • UI corporativa • build sem chat/preview/treinamento")

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
