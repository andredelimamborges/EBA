# elder_brain_v8_5_pro_prod_full.py
"""
Elder Brain Analytics — PRO (Full) • PROD
- TODAS as funcionalidades: Extração → Análise → Gráficos → PDF Deluxe → Chat → Treinamento
- API keys lidas via st.secrets (não aparecem para o usuário)
- Painel Administrativo protegido por senha (ADMIN_PASSWORD)
- Custos e tokens visíveis somente ao Admin
- Compatível com groq==0.8.0 (corrige erro de proxies)
Autor: André de Lima
"""

import os, io, re, json, time, tempfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
from pdfminer.high_level import extract_text
import streamlit as st
import httpx

# ======== LLM Clients ========
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ======== Token helpers (estimativa caso SDK não retorne usage) ========
try:
    import tiktoken
except Exception:
    tiktoken = None

# ======== Diretórios / Constantes ========
TRAINING_DIR = "training_data"
PROCESSED_DIR = "relatorios_processados"
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MODELOS_SUGERIDOS_GROQ = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
]
MODELOS_SUGERIDOS_OPENAI = ["gpt-4o-mini", "gpt-4o"]

MAX_TOKENS_FIXED = 4096
TEMP_FIXED = 0.3
# preços de referência (apenas para estimativa)
GPT_PRICE_INPUT_PER_1K = 0.005
GPT_PRICE_OUTPUT_PER_1K = 0.015

# ======== Tema / CSS ========
DARK_CSS = """
<style>
:root{
  --bg:#20152b; --panel:#2a1f39; --panel-2:#332447; --accent:#9b6bff;
  --text:#EAE6F5; --muted:#B9A8D9; --success:#2ECC71; --warn:#F39C12; --danger:#E74C3C;
}
html, body, .stApp { background: var(--bg); color: var(--text) !important; }
section[data-testid="stSidebar"] { background: #1b1c25; border-right: 1px solid #3b3d4b; }
header[data-testid="stHeader"] { display:none !important; }
.kpi-card{background:var(--panel); border:1px solid #3f4151; border-radius:14px; padding:14px; box-shadow:0 8px 24px rgba(0,0,0,.22)}
.small{color:var(--muted);font-size:.9rem}
.badge{display:inline-block;background:#2a2b36;color:var(--muted);padding:.25rem .55rem;border-radius:999px;border:1px solid #3f4151;margin-right:.35rem}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,var(--accent),#7c69d4); color:white; border:0; padding:.55rem 1rem; border-radius:12px; font-weight:700; box-shadow:0 10px 22px rgba(96,81,155,.25)}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.06)}
</style>
"""

# ======== Fontes (Montserrat opcional no PDF) ========
def _download_font(dst: str, url: str) -> bool:
    try:
        import requests
        r = requests.get(url, timeout=15)
        if r.ok:
            with open(dst, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def _register_montserrat(pdf: FPDF) -> bool:
    os.makedirs("fonts", exist_ok=True)
    font_map = {
        "Montserrat-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Regular.ttf",
        "Montserrat-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
        "Montserrat-Italic.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Italic.ttf",
    }
    ok = True
    for fname, url in font_map.items():
        path = os.path.join("fonts", fname)
        if not os.path.exists(path):
            if not _download_font(path, url):
                ok = False
    if not ok:
        return False
    try:
        pdf.add_font("Montserrat", "", os.path.join("fonts", "Montserrat-Regular.ttf"), uni=True)
        pdf.add_font("Montserrat", "B", os.path.join("fonts", "Montserrat-Bold.ttf"), uni=True)
        pdf.add_font("Montserrat", "I", os.path.join("fonts", "Montserrat-Italic.ttf"), uni=True)
        return True
    except Exception:
        return False

# ======== Token Accounting ========
@dataclass
class TokenStep:
    prompt: int = 0
    completion: int = 0
    @property
    def total(self): return self.prompt + self.completion

@dataclass
class TokenTracker:
    steps: Dict[str, TokenStep] = field(default_factory=lambda: {
        "extracao": TokenStep(),
        "analise": TokenStep(),
        "chat": TokenStep(),
        "pdf": TokenStep()  # lógico, sem custo
    })
    model: str = ""
    provider: str = ""

    def add(self, step: str, prompt_tokens: int, completion_tokens: int):
        if step not in self.steps:
            self.steps[step] = TokenStep()
        self.steps[step].prompt += int(prompt_tokens or 0)
        self.steps[step].completion += int(completion_tokens or 0)

    def dict(self):
        return {k: {"prompt": v.prompt, "completion": v.completion, "total": v.total} for k, v in self.steps.items()}

    @property
    def total_prompt(self): return sum(s.prompt for s in self.steps.values())
    @property
    def total_completion(self): return sum(s.completion for s in self.steps.values())
    @property
    def total_tokens(self): return self.total_prompt + self.total_completion

    def cost_usd_gpt(self) -> float:
        return (self.total_prompt/1000.0)*GPT_PRICE_INPUT_PER_1K + (self.total_completion/1000.0)*GPT_PRICE_OUTPUT_PER_1K

def _estimate_tokens(text: str) -> int:
    if not text: return 0
    try:
        if tiktoken:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
    except Exception:
        pass
    return max(1, int(len(text) / 4))  # heurística

# ======== Cliente seguro (conserta erro de proxies) ========
@st.cache_resource(show_spinner=False)
def get_llm_client_cached(provider: str, api_key: str):
    """Cria cliente LLM seguro e compatível, evitando o bug de proxies no Streamlit Cloud."""
    if not api_key:
        raise RuntimeError("Chave da API não configurada. Defina nos Secrets do Streamlit.")
    pv = (provider or "Groq").lower()

    # ---- implementação robusta ----
    if pv == "groq":
        try:
            from groq import Groq
            # Cria com api_key e http_client sem proxies
            client = Groq(
                api_key=api_key,
                http_client=httpx.Client(proxies=None)
            )
            # Garante que o ambiente não interfira
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)
            return client
        except Exception as e:
            raise RuntimeError(f"[Erro cliente] Groq SDK falhou ({e})")

    elif pv == "openai":
        try:
            from openai import OpenAI
            # Cria com api_key e http_client sem proxies
            client = OpenAI(
                api_key=api_key,
                http_client=httpx.Client(proxies=None)
            )
            # Garante que o ambiente não interfira
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)
            return client
        except Exception as e:
            raise RuntimeError(f"[Erro cliente] OpenAI SDK falhou ({e})")

    else:
        raise RuntimeError(f"Provedor não suportado: {provider}")
    
# ======== Prompts =========
EXTRACTION_PROMPT = """Você é um especialista em análise de relatórios BFA (Big Five Analysis) para seleção de talentos.
Sua tarefa: extrair dados do relatório abaixo e retornar APENAS um JSON válido, sem texto adicional.

ESTRUTURA OBRIGATÓRIA:
{
  "candidato": {"nome": "string ou null","cargo_avaliado": "string ou null"},
  "traits_bfa": {
    "Abertura": número 0-10 ou null,
    "Conscienciosidade": número 0-10 ou null,
    "Extroversao": número 0-10 ou null,
    "Amabilidade": número 0-10 ou null,
    "Neuroticismo": número 0-10 ou null
  },
  "competencias_ms": [{"nome": "string","nota": número,"classificacao": "string"}],
  "facetas_relevantes": [{"nome": "string","percentil": número,"interpretacao": "string resumida"}],
  "indicadores_saude_emocional": {"ansiedade": 0-100 ou null,"irritabilidade": 0-100 ou null,"estado_animo": 0-100 ou null,"impulsividade": 0-100 ou null},
  "potencial_lideranca": "BAIXO" | "MÉDIO" | "ALTO" ou null,
  "integridade_fgi": 0-100 ou null,
  "resumo_qualitativo": "texto original do relatório",
  "pontos_fortes": ["3-5 itens"],
  "pontos_atencao": ["2-4 itens"],
  "fit_geral_cargo": 0-100
}

REGRAS:
1) Normalize percentis para escalas; 2) Big Five: percentil 60 -> 6.0/10; 3) Extraia TODAS as competências;
4) Use null quando não houver evidência; 5) resumo_qualitativo = texto original;
6) pontos_fortes (3-5) e pontos_atencao (2-4); 7) fit_geral_cargo 0-100 baseado no cargo: {cargo}.

RELATÓRIO:
\"\"\"{text}\"\"\"

MATERIAIS (opcional):
\"\"\"{training_context}\"\"\"

Retorne apenas o JSON puro.
"""

ANALYSIS_PROMPT = """Você é um consultor sênior de RH especializado em análise comportamental.

Cargo avaliado: {cargo}

DADOS (JSON extraído):
{json_data}

PERFIL IDEAL DO CARGO:
{perfil_cargo}

Responda em JSON:
{
  "compatibilidade_geral": 0-100,
  "decisao": "RECOMENDADO" | "RECOMENDADO COM RESSALVAS" | "NÃO RECOMENDADO",
  "justificativa_decisao": "texto",
  "analise_tracos": {
    "Abertura": "texto","Conscienciosidade": "texto","Extroversao": "texto","Amabilidade": "texto","Neuroticismo": "texto"
  },
  "competencias_criticas": [{"competencia":"nome","avaliacao":"texto","status":"ATENDE" | "PARCIAL" | "NÃO ATENDE"}],
  "saude_emocional_contexto": "texto",
  "recomendacoes_desenvolvimento": ["a","b","c"],
  "cargos_alternativos": [{"cargo":"nome","justificativa":"texto"}],
  "resumo_executivo": "100-150 palavras"
}"""

# ======== Helpers I/O ========
def extract_pdf_text_bytes(file) -> str:
    try:
        return extract_text(file)
    except Exception as e:
        return f"[ERRO_EXTRACAO_PDF] {e}"

def load_all_training_texts() -> str:
    """Carrega todos os PDFs e TXTs da pasta training_data e junta em um contexto."""
    texts = []
    for fname in sorted(os.listdir("training_data")):
        path = os.path.join("training_data", fname)
        try:
            if fname.lower().endswith(".pdf"):
                with open(path, "rb") as f:
                    txt = extract_text(f)
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            texts.append(f"--- {fname} ---\n{txt[:2000]}\n")
        except Exception:
            continue
    return "\n".join(texts)

def gerar_perfil_cargo_dinamico(cargo: str) -> Dict:
    return {
        "traits_ideais": {"Abertura": (5,8), "Conscienciosidade": (6,9), "Extroversão": (4,8), "Amabilidade": (5,8), "Neuroticismo": (0,5)},
        "competencias_criticas": ["Adaptabilidade","Comunicação","Trabalho em Equipe","Resolução de Problemas"],
        "descricao": f"Perfil para {cargo}"
    }

# ======== Chat/Completion wrappers ========
def _chat_completion_json(provider, client, model, messages, force_json=True):
    usage = None
    if (provider or "").lower() == "groq":
        kwargs = dict(model=model, messages=messages, max_tokens=MAX_TOKENS_FIXED, temperature=TEMP_FIXED)
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        if usage:
            usage = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
        return content, usage
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages if not force_json else ([{"role":"system","content":"Responda apenas com JSON válido."}] + messages),
            temperature=TEMP_FIXED,
            max_tokens=MAX_TOKENS_FIXED,
            response_format={"type":"json_object"} if force_json else None
        )
        content = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        if usage:
            usage = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
        return content, usage

def _estimate_and_add(tracker, step, messages, content, usage):
    if usage:
        tracker.add(step, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        return
    prompt_text = "\n".join([m.get("content","") for m in messages])
    tracker.add(step, _estimate_tokens(prompt_text), _estimate_tokens(content))

# ======== Core IA ========
def extract_bfa_data(text: str, cargo: str, training_context: str,
                     provider: str, model_id: str, token: str, tracker: TokenTracker
                     ) -> Tuple[Optional[Dict], str]:
    try:
        client = get_llm_client_cached(provider, token)
    except Exception as e:
        return None, f"[Erro cliente] {e}"

    prompt = (EXTRACTION_PROMPT
              .replace("{text}", text[:10000])
              .replace("{training_context}", training_context[:3000])
              .replace("{cargo}", cargo))

    try:
        content, usage = _chat_completion_json(provider, client, model_id.strip(), [{"role": "user", "content": prompt}], True)
        _estimate_and_add(tracker, "extracao", [{"role":"user","content":prompt}], content, usage)

        try:
            return json.loads(content), content
        except Exception:
            m = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group(0)), content
            return None, f"Nenhum JSON válido encontrado: {content[:800]}..."
    except Exception as e:
        msg = f"[Erro LLM] {e}"
        if hasattr(e, "response") and getattr(e.response, "text", None):
            msg += f" - Resposta: {e.response.text}"
        return None, msg

def analyze_bfa_data(bfa_data: Dict, cargo: str, perfil_cargo: Dict,
                     provider: str, model_id: str, token: str, tracker: TokenTracker
                     ) -> Tuple[Optional[Dict], str]:
    try:
        client = get_llm_client_cached(provider, token)
    except Exception as e:
        return None, f"[Erro cliente] {e}"

    prompt = (ANALYSIS_PROMPT
              .replace("{cargo}", cargo)
              .replace("{json_data}", json.dumps(bfa_data, ensure_ascii=False, indent=2))
              .replace("{perfil_cargo}", json.dumps(perfil_cargo, ensure_ascii=False, indent=2)))

    try:
        content, usage = _chat_completion_json(provider, client, model_id.strip(), [{"role": "user", "content": prompt}], True)
        _estimate_and_add(tracker, "analise", [{"role":"user","content":prompt}], content, usage)

        try:
            return json.loads(content), content
        except Exception:
            m = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group(0)), content
            return None, f"Nenhum JSON válido encontrado: {content[:800]}..."
    except Exception as e:
        msg = f"[Erro LLM] {e}"
        if hasattr(e, "response") and getattr(e.response, "text", None):
            msg += f" - Resposta: {e.response.text}"
        return None, msg

# ======== Gráficos ========
def criar_radar_bfa(traits: Dict, ideais: Dict) -> go.Figure:
    try:
        categories = list(traits.keys())
        values = [float(traits.get(c, 0)) for c in categories]
        ideal_min = [float(ideais.get(c, (0,10))[0]) for c in categories]
        ideal_max = [float(ideais.get(c, (0,10))[1]) for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Candidato'))
        fig.add_trace(go.Scatterpolar(r=ideal_min, theta=categories, fill=None, name='Ideal Min', line_color='green', dash='dash'))
        fig.add_trace(go.Scatterpolar(r=ideal_max, theta=categories, fill=None, name='Ideal Max', line_color='green', dash='dash'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=True)
        return fig
    except Exception:
        return None

def criar_grafico_competencias(comps: List[Dict]) -> go.Figure:
    try:
        if not comps: return None
        df = pd.DataFrame(comps).sort_values('nota', ascending=False)[:10]
        fig = go.Figure(go.Bar(x=df['nota'], y=df['nome'], orientation='h', marker_color='purple'))
        fig.update_layout(title="Top Competências", xaxis_title="Nota")
        return fig
    except Exception:
        return None

def criar_gauge_fit(fit: float) -> go.Figure:
    try:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=fit, domain={'x': [0,1], 'y': [0,1]},
                                    title={'text': "Fit Geral"}, gauge={'axis': {'range': [0,100]},
                                    'bar': {'color': "darkblue"}, 'steps': [{'range': [0,50], 'color': "lightgray"},
                                    {'range': [50,80], 'color': "gray"}, {'range': [80,100], 'color': "green"}]}))
        return fig
    except Exception:
        return None

# ======== PDF Deluxe ========
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self._family = "Arial"
        self._main_font = True

    def set_main_family(self, family: str, montserrat_ok: bool):
        self._family = family if montserrat_ok else "Arial"
        self._main_font = montserrat_ok

    def _safe(self, text: str) -> str:
        return str(text or "").encode('latin-1', 'replace').decode('latin-1')

    def header(self):
        if self.page_no() > 1:
            self.set_font(self._family, 'I', 8)
            self.cell(0, 10, 'Elder Brain Analytics — Relatório Confidencial', 0, 1, 'C')
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._family, 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def gerar_pdf_corporativo(bfa_data: Dict, analysis: Dict, cargo: str, save_path: str = None, logo_path: str = None) -> io.BytesIO:
    try:
        pdf = PDFReport()
        montserrat_ok = _register_montserrat(pdf)
        pdf.set_main_family("Montserrat" if montserrat_ok else "Arial", montserrat_ok)
        pdf.add_page()

        # Cabeçalho
        if logo_path and os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=30)
        pdf.set_font(pdf._family, 'B', 16)
        pdf.cell(0, 10, pdf._safe('RELATÓRIO DE ANÁLISE COMPORTAMENTAL'), ln=1, align='C')
        pdf.set_font(pdf._family, '', 12)
        pdf.cell(0, 10, pdf._safe(f"Cargo: {cargo}"), ln=1, align='C')
        pdf.ln(5)

        # Resumo Executivo
        pdf.set_font(pdf._family, 'B', 12)
        pdf.cell(0, 8, pdf._safe('Resumo Executivo'), ln=1)
        pdf.set_font(pdf._family, '', 10)
        pdf.multi_cell(0, 5, pdf._safe(analysis.get('resumo_executivo', '')))
        pdf.ln(5)

        # Decisão e Justificativa
        pdf.set_font(pdf._family, 'B', 12)
        pdf.cell(0, 8, pdf._safe('Decisão Recomendada'), ln=1)
        pdf.set_font(pdf._family, '', 10)
        decisao = analysis.get('decisao', 'N/A')
        pdf.cell(0, 5, pdf._safe(f"{decisao} - {analysis.get('justificativa_decisao', '')}"), ln=1)
        pdf.ln(5)

        # Big Five Traits
        pdf.set_font(pdf._family, 'B', 12)
        pdf.cell(0, 8, pdf._safe('Análise dos Traços Big Five'), ln=1)
        traits = bfa_data.get('traits_bfa', {})
        for trait, value in traits.items():
            if value is not None:
                pdf.set_font(pdf._family, 'B', 10)
                pdf.cell(0, 6, pdf._safe(f"{trait}: {value:.1f}/10"), ln=1)
                pdf.set_font(pdf._family, '', 9)
                pdf.multi_cell(0, 5, pdf._safe(analysis.get('analise_tracos', {}).get(trait, '')))
        pdf.ln(5)

        # Competências
        pdf.set_font(pdf._family, 'B', 12)
        pdf.cell(0, 8, pdf._safe('Competências Críticas'), ln=1)
        for comp in analysis.get('competencias_criticas', []):
            pdf.set_font(pdf._family, 'B', 10)
            pdf.cell(0, 6, pdf._safe(f"{comp.get('competencia')}: {comp.get('status')}"), ln=1)
            pdf.set_font(pdf._family, '', 9)
            pdf.multi_cell(0, 5, pdf._safe(comp.get('avaliacao', '')))
        pdf.ln(5)

        # Saúde Emocional
        pdf.set_font(pdf._family, 'B', 12)
        pdf.cell(0, 8, pdf._safe('Saúde Emocional'), ln=1)
        pdf.set_font(pdf._family, '', 10)
        pdf.multi_cell(0, 5, pdf._safe(analysis.get('saude_emocional_contexto', '')))
        pdf.ln(5)

        # Recomendações
        pdf.set_font(pdf._family, 'B', 12)
        pdf.cell(0, 8, pdf._safe('Recomendações de Desenvolvimento'), ln=1)
        pdf.set_font(pdf._family, '', 10)
        for rec in analysis.get('recomendacoes_desenvolvimento', []):
            pdf.cell(0, 5, pdf._safe(f"- {rec}"), ln=1)
        pdf.ln(5)

        # Cargos Alternativos
        pdf.set_font(pdf._family, 'B', 12)
        pdf.cell(0, 8, pdf._safe('Cargos Alternativos'), ln=1)
        for alt in analysis.get('cargos_alternativos', []):
            pdf.set_font(pdf._family, 'B', 10)
            pdf.cell(0, 6, pdf._safe(alt.get('cargo', '')), ln=1)
            pdf.set_font(pdf._family, '', 9)
            pdf.multi_cell(0, 5, pdf._safe(alt.get('justificativa', '')))

        pdf.ln(2); pdf.set_font(pdf._family,'I',8)
        pdf.multi_cell(0,4,pdf._safe("Este relatório auxilia a decisão e não substitui avaliação profissional. Uso interno — Elder Brain Analytics PRO (Versão Deluxe)."))

        try:
            out_bytes = pdf.output(dest='S')
            if isinstance(out_bytes, str): out_bytes = out_bytes.encode('latin-1','replace')
        except Exception:
            fb = PDFReport(); fb.set_main_family("Helvetica", False); fb.add_page()
            fb.set_font(fb._family,'B',14); fb.cell(0,10,fb._safe('RELATÓRIO DE ANÁLISE COMPORTAMENTAL'), ln=1, align='C')
            fb.set_font(fb._family,'',11); fb.multi_cell(0,8,fb._safe(f"Relatório gerado para: {cargo}\nData: {datetime.now():%d/%m/%Y %H:%M}"))
            out_bytes = fb.output(dest='S')
            if isinstance(out_bytes,str): out_bytes = out_bytes.encode('latin-1','replace')

        buf = io.BytesIO(out_bytes); buf.seek(0)
        if save_path:
            try:
                with open(save_path,'wb') as f: f.write(buf.getbuffer())
            except Exception as e:
                st.error(f"Erro ao salvar PDF: {e}")
        return buf
    except Exception as e:
        st.error(f"Erro crítico na geração do PDF: {e}")
        return io.BytesIO(b'%PDF-1.4\n%EOF\n')

# ======== Chat com Elder Brain ========
CHAT_PROMPT = """Você é o Elder Brain, especialista em análise comportamental.

Dados extraídos: {bfa_data}

Análise: {analysis}

Cargo: {cargo}

Pergunta do usuário: {query}

Responda de forma concisa e profissional."""

def chat_with_elder_brain(query: str, bfa_data: Dict, analysis: Dict, cargo: str,
                          provider: str, model_id: str, token: str, tracker: TokenTracker) -> str:
    try:
        client = get_llm_client_cached(provider, token)
    except Exception as e:
        return f"[Erro cliente] {e}"

    prompt = (CHAT_PROMPT
              .replace("{bfa_data}", json.dumps(bfa_data, ensure_ascii=False))
              .replace("{analysis}", json.dumps(analysis, ensure_ascii=False))
              .replace("{cargo}", cargo)
              .replace("{query}", query))

    try:
        content, usage = _chat_completion_json(provider, client, model_id.strip(), [{"role": "user", "content": prompt}], False)
        _estimate_and_add(tracker, "chat", [{"role":"user","content":prompt}], content, usage)
        return content
    except Exception as e:
        return f"[Erro no chat] {e}"

# ======== UI helpers ========
def kpi_card(title, value, sub=None):
    st.markdown(
        f'<div class="kpi-card"><div style="font-weight:700;font-size:1.02rem">{title}</div>'
        f'<div style="font-size:1.9rem;margin:.2rem 0 .25rem 0">{value}</div>'
        f'<div class="small">{sub or ""}</div></div>', unsafe_allow_html=True
    )

# ======== Imports Adicionais ========
import uuid
import gspread
from google.oauth2.service_account import Credentials

# ======== Função para Conectar à Sheet ========
@st.cache_resource(show_spinner=False)
def get_gsheet_client():
    """Conecta à Google Sheets usando service account de st.secrets."""
    try:
        creds_json = json.loads(st.secrets.get("GOOGLE_SERVICE_ACCOUNT", "{}"))
        creds = Credentials.from_service_account_info(creds_json, scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
        client = gspread.authorize(creds)
        sheet = client.open_by_id("1dW3Lvn9LrV6FgUshk4x2TwnJA1grM2iETThn3T3rAVw")
        worksheet = sheet.worksheet("LogsUso")  # Assume aba "LogsUso" existe; crie manualmente se necessário
        return worksheet
    except Exception as e:
        st.error(f"Erro ao conectar com Google Sheets: {e}")
        return None

# ======== Função para Logar Tokens ========
def log_tokens_to_sheet(tracker: TokenTracker, step: str, cargo: str):
    """Appenda log de tokens na Google Sheet."""
    worksheet = get_gsheet_client()
    if not worksheet:
        return  # Silencia erro para não quebrar o app

    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_data = tracker.steps.get(step, TokenStep())
        row = [
            now,
            st.session_state['session_id'],
            tracker.provider,
            tracker.model,
            cargo,
            step,
            step_data.prompt,
            step_data.completion,
            step_data.total,
            tracker.cost_usd_gpt()  # Custo total da sessão até agora
        ]
        worksheet.append_row(row)
    except Exception as e:
        pass  # Silencia para não impactar UX

# ======== APP ========
def main():
    st.set_page_config(page_title="EBA — Corporate PROD (Full)", page_icon="🧠", layout="wide")
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault('provider', "Groq")
    ss.setdefault('modelo', "llama-3.1-8b-instant")
    ss.setdefault('cargo', "")
    ss.setdefault('analysis_complete', False)
    ss.setdefault('bfa_data', None)
    ss.setdefault('analysis', None)
    ss.setdefault('pdf_generated', None)
    ss.setdefault('tracker', TokenTracker())
    ss.setdefault('admin_mode', False)   # sempre inicia como usuário comum
    ss.setdefault('session_id', str(uuid.uuid4()))  # ID único por sessão

    # ===== Topo
    st.markdown("## 🧠 Elder Brain Analytics — Corporate (PROD • Full)")
    st.markdown('<span class="badge">PDF Deluxe</span> <span class="badge">Seguro</span> <span class="badge">Streamlit Cloud</span>', unsafe_allow_html=True)

    # ===== Sidebar (Config + Admin)
    with st.sidebar:
        st.header("⚙️ Configuração")
        provider = st.radio("Provedor", ["Groq","OpenAI"], index=0, key="provider")
        modelo = st.text_input("Modelo", value=ss['modelo'],
                               help=("Sugestões: " + ", ".join(MODELOS_SUGERIDOS_GROQ if provider=="Groq" else MODELOS_SUGERIDOS_OPENAI)))
        ss['modelo'] = modelo

        # 🔐 SEM CAMPO DE API KEY PARA O USUÁRIO — usa st.secrets
        token = st.secrets.get("GROQ_API_KEY","") if provider=="Groq" else st.secrets.get("OPENAI_API_KEY","")

        st.caption("Temperatura fixa: 0.3 · Máx tokens: 4096")
        ss['cargo'] = st.text_input("Cargo para análise", value=ss['cargo'])
        if ss['cargo']:
            with st.expander("Perfil gerado (dinâmico)"):
                st.json(gerar_perfil_cargo_dinamico(ss['cargo']))

        st.markdown("---")
        st.subheader("🔒 Painel Administrativo")
        admin_pwd = st.text_input("Senha do Admin", type="password", placeholder="somente administradores")
        if admin_pwd:
            if admin_pwd == st.secrets.get("ADMIN_PASSWORD",""):
                ss['admin_mode'] = True
                st.success("Acesso administrativo concedido")
            else:
                ss['admin_mode'] = False
                st.error("Senha incorreta")
        else:
            ss['admin_mode'] = False

        # 📈 Token Log — SOMENTE admin
        if ss['admin_mode']:
            st.markdown("---")
            st.header("📈 Token Log")
            td = ss['tracker'].dict()
            for step in ["extracao","analise","chat","pdf"]:
                d = td.get(step, {"prompt":0,"completion":0,"total":0})
                st.write(f"- **{step.capitalize()}**: {d['total']}  (prompt {d['prompt']} / output {d['completion']})")
            st.write(f"**Total:** {ss['tracker'].total_tokens} tokens")
            st.write(f"**Custo (estimado):** ${ss['tracker'].cost_usd_gpt():.4f}")
            st.markdown("### 📊 Logs de Uso (Todos os Usuários)")
            worksheet = get_gsheet_client()
            if worksheet:
                data = worksheet.get_all_values()
                if len(data) > 1:  # Ignora header se vazio
                    df_logs = pd.DataFrame(data[1:], columns=data[0])
                    st.dataframe(df_logs, use_container_width=True, hide_index=True)
                else:
                    st.info("Nenhum log registrado ainda.")
            else:
                st.warning("Não foi possível carregar logs da sheet.")
        else:
            st.caption("modo usuário — sem métricas financeiras visíveis")

    # ===== KPIs (cliente NUNCA vê custo/tokens)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("Status", "Pronto", "Aguardando PDF")
    if ss['admin_mode']:
        with c2: kpi_card("Tokens (Total)", f"{ss['tracker'].total_tokens}", "desde o início")
        with c3: kpi_card("Prompt/Output", f"{ss['tracker'].total_prompt}/{ss['tracker'].total_completion}", "tokens")
        with c4: kpi_card("Custo Estimado", f"${ss['tracker'].cost_usd_gpt():.4f}", "apenas admin")
    else:
        with c2: kpi_card("Relatórios", "—", "em sessão")
        with c3: kpi_card("Andamento", "—", "")
        with c4: kpi_card("Disponibilidade", "Online", "")

    # ===== Upload & Treinamento
    st.markdown("### 📄 Upload do Relatório BFA")
    uploaded_file = st.file_uploader("Carregue o PDF do relatório BFA", type=["pdf"])

    with st.expander("📚 Materiais de Treinamento (Opcional)"):
        training_files = st.file_uploader("Arraste PDFs/TXTs", accept_multiple_files=True, key="training")
        if training_files:
            for f in training_files:
                save_path = os.path.join(TRAINING_DIR, f"{int(time.time())}_{f.name}")
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
            st.success(f"{len(training_files)} arquivo(s) salvos")

    # ===== Processamento
    if uploaded_file:
        if not ss['cargo']: st.error("Informe o cargo na sidebar"); st.stop()
        if not token:
            st.error("Chave da API não configurada nos Secrets do Streamlit. Defina GROQ_API_KEY/OPENAI_API_KEY.")
            st.stop()
        if not (ss['modelo'] and ss['modelo'].strip()): st.error("Informe o modelo"); st.stop()

        with st.spinner("Extraindo texto do PDF..."):
            raw_text = extract_pdf_text_bytes(uploaded_file)
        if raw_text.startswith("[ERRO"): st.error(raw_text); st.stop()
        st.success("✓ Texto extraído")
        st.text_area("Prévia do texto (início)", raw_text[:1500], height=180)

        if st.button("🔬 ANALISAR RELATÓRIO", type="primary", use_container_width=True):
            training_context = load_all_training_texts()
            tracker: TokenTracker = ss['tracker']
            tracker.model, tracker.provider = ss['modelo'], ss['provider']

            with st.spinner("Etapa 1/2: Extraindo dados estruturados..."):
                bfa_data, raw1 = extract_bfa_data(raw_text, ss['cargo'], training_context, ss['provider'], ss['modelo'], token, tracker)
                log_tokens_to_sheet(tracker, "extracao", ss['cargo'])
            if not bfa_data:
                st.error("Falha na extração"); 
                with st.expander("Resposta bruta da IA"):
                    st.code(raw1)
                st.stop()

            perfil = gerar_perfil_cargo_dinamico(ss['cargo'])
            with st.spinner("Etapa 2/2: Analisando compatibilidade..."):
                analysis, raw2 = analyze_bfa_data(bfa_data, ss['cargo'], perfil, ss['provider'], ss['modelo'], token, tracker)
                log_tokens_to_sheet(tracker, "analise", ss['cargo'])
            if not analysis:
                st.error("Falha na análise");
                with st.expander("Resposta bruta da IA"):
                    st.code(raw2)
                st.stop()

            ss['bfa_data'], ss['analysis'], ss['analysis_complete'] = bfa_data, analysis, True
            st.success("✓ Análise concluída!"); st.rerun()

    # ===== Resultados
    if ss.get('analysis_complete') and ss.get('bfa_data') and ss.get('analysis'):
        st.markdown("## 📊 Resultados")
        decisao = ss['analysis'].get('decisao','N/A')
        compat = float(ss['analysis'].get('compatibilidade_geral',0) or 0)

        c1,c2,c3 = st.columns([2,1,1])
        with c1: st.markdown(f"### 🏷️ Decisão: **{decisao}**")
        with c2: st.metric("Compatibilidade", f"{compat:.0f}%")
        with c3: st.metric("Liderança", ss['bfa_data'].get('potencial_lideranca','N/A'))

        with st.expander("📋 Resumo Executivo", expanded=True):
            st.write(ss['analysis'].get('resumo_executivo',''))
        st.info(ss['analysis'].get('justificativa_decisao',''))

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Big Five","💼 Competências","🧘 Saúde Emocional","📈 Desenvolvimento","📄 Dados Brutos"])
        with tab1:
            traits = ss['bfa_data'].get('traits_bfa',{})
            fig_radar = criar_radar_bfa(traits, gerar_perfil_cargo_dinamico(ss['cargo']).get('traits_ideais',{}))
            st.plotly_chart(fig_radar, use_container_width=True)
            # tabela
            traits_ideais = gerar_perfil_cargo_dinamico(ss['cargo']).get('traits_ideais',{})
            df_traits = pd.DataFrame([
                {'Traço': k, 'Valor': f"{(traits.get(k) if traits.get(k) is not None else 0):.1f}/10" if traits.get(k) is not None else "N/A",
                 'Faixa Ideal': f"{traits_ideais.get(k,(0,10))[0]:.0f}-{traits_ideais.get(k,(0,10))[1]:.0f}"}
                for k in ["Abertura","Conscienciosidade","Extroversão","Amabilidade","Neuroticismo"]
            ])
            st.dataframe(df_traits, use_container_width=True, hide_index=True)
            # análise por traço
            st.markdown("##### Análise Detalhada")
            for trait, txt in (ss['analysis'].get('analise_tracos',{}) or {}).items():
                with st.expander(f"**{trait}**"):
                    st.write(txt)

        with tab2:
            comps = ss['bfa_data'].get('competencias_ms',[])
            figc = criar_grafico_competencias(comps)
            if figc: st.plotly_chart(figc, use_container_width=True)
            st.markdown("##### Competências Críticas")
            for comp in ss['analysis'].get('competencias_criticas',[]):
                status = comp.get('status'); compn = comp.get('competencia'); txt = comp.get('avaliacao','')
                if status == 'ATENDE': st.success(f"✓ {compn} — {status}"); st.caption(txt)
                elif status == 'PARCIAL': st.warning(f"⚠ {compn} — {status}"); st.caption(txt)
                else: st.error(f"✗ {compn} — {status}"); st.caption(txt)
            if comps:
                with st.expander("Ver todas as competências"):
                    df_comp = pd.DataFrame(comps).sort_values('nota', ascending=False)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
            else:
                st.warning("Nenhuma competência extraída.")

        with tab3:
            st.subheader("Saúde Emocional e Resiliência")
            st.write(ss['analysis'].get('saude_emocional_contexto',''))
            indicadores = ss['bfa_data'].get('indicadores_saude_emocional',{})
            if any(v is not None for v in indicadores.values()):
                st.markdown("##### Indicadores (0-100, menor melhor)")
                cols = st.columns(2)
                for i,(k,v) in enumerate(indicadores.items()):
                    if v is None: continue
                    with cols[i%2]: st.metric(k.replace('_',' ').title(), f"{float(v):.0f}")
            facetas = ss['bfa_data'].get('facetas_relevantes', [])
            if facetas:
                with st.expander("Facetas detalhadas"):
                    for f in facetas:
                        st.markdown(f"**{f.get('nome','')}** (Percentil: {f.get('percentil',0):.0f})")
                        st.caption(f.get('interpretacao','')); st.markdown("---")

        with tab4:
            st.subheader("Plano de Desenvolvimento")
            recs = ss['analysis'].get('recomendacoes_desenvolvimento',[])
            if recs:
                for i,r in enumerate(recs,1): st.markdown(f"**{i}.** {r}")
            pf = ss['bfa_data'].get('pontos_fortes',[])
            if pf:
                st.markdown("##### ✅ Pontos Fortes")
                for x in pf: st.success(f"• {x}")
            pa = ss['bfa_data'].get('pontos_atencao',[])
            if pa:
                st.markdown("##### ⚠️ Pontos de Atenção")
                for x in pa: st.warning(f"• {x}")
            alt = ss['analysis'].get('cargos_alternativos',[])
            if alt:
                st.markdown("##### 🔄 Cargos Alternativos")
                for c in alt:
                    with st.expander(f"**{c.get('cargo','')}**"):
                        st.write(c.get('justificativa',''))

        with tab5:
            c1,c2 = st.columns(2)
            with c1: st.json(ss['bfa_data'])
            with c2: st.json(ss['analysis'])

        st.markdown("### 🎯 Compatibilidade")
        st.plotly_chart(criar_gauge_fit(compat), use_container_width=True)

        st.markdown("### 📄 Gerar PDF")
        logo_path = st.text_input("Caminho para logo (opcional)", value="")
        if st.button("🔨 Gerar PDF", key="gen_pdf"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome = ((ss['bfa_data'].get('candidato',{}) or {}).get('nome') or 'candidato')
            nome = re.sub(r'[^\w\s-]', '', str(nome)).strip().replace(' ', '_')
            fname = f"relatorio_{nome}_{ts}.pdf"
            path = os.path.join(PROCESSED_DIR, fname)
            buf = gerar_pdf_corporativo(ss['bfa_data'], ss['analysis'], ss['cargo'], save_path=path, logo_path=logo_path if logo_path else None)
            ss['tracker'].add("pdf", 0, 0)
            log_tokens_to_sheet(ss['tracker'], "pdf", ss['cargo'])
            if buf.getbuffer().nbytes > 100:
                ss['pdf_generated'] = {'buffer': buf, 'filename': fname}
                st.success(f"✓ PDF gerado: {fname}")
            else:
                st.error("Arquivo PDF vazio (erro na geração).")
        if ss.get('pdf_generated'):
            st.download_button("⬇️ Download do PDF", data=ss['pdf_generated']['buffer'].getvalue(),
                               file_name=ss['pdf_generated']['filename'], mime="application/pdf", use_container_width=True)

        st.markdown("### 💬 Chat com o Elder Brain")
        q = st.text_input("Pergunte sobre este relatório", placeholder="Ex.: Principais riscos para este cargo?")
        if q and st.button("Enviar", key="ask"):
            with st.spinner("Pensando..."):
                ans = chat_with_elder_brain(q, ss['bfa_data'], ss['analysis'], ss['cargo'],
                                            ss['provider'], ss['modelo'], token, ss['tracker'])
                log_tokens_to_sheet(ss['tracker'], "chat", ss['cargo'])
            st.markdown(f"**Você:** {q}")
            st.markdown(f"**Elder Brain:** {ans}")

    st.caption(f"📁 Relatórios salvos em: `{PROCESSED_DIR}`")

if __name__ == "__main__":
    main()