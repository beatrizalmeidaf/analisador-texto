"""
Geração de relatório de análise em HTML e PDF.

HTML: puro Python (template f-string), sem dependências extras.
PDF : gerado via pdfkit (wkhtmltopdf) com fallback para weasyprint
      se wkhtmltopdf não estiver instalado.
"""

from __future__ import annotations
import base64
import datetime
import io
import re
import tempfile
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str | None:
    """Converte figura matplotlib para string base64 PNG."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=96)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return None


def _img_tag(b64: str | None, alt: str = "gráfico") -> str:
    if not b64:
        return f"<p><em>[{alt} não disponível]</em></p>"
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;border-radius:8px;"/>'


# ---------------------------------------------------------------------------
# Construção do HTML
# ---------------------------------------------------------------------------

def gerar_html_relatorio(
    texto_original: str,
    metrics: dict,
    word_counts: dict,
    df_tfidf=None,
    resumo: list[str] | None = None,
    sentimentos: list[dict] | None = None,
    sentimento_geral: dict | None = None,
    keyphrases: list[dict] | None = None,
    classificacao: list[dict] | None = None,
    figs: dict | None = None,          # {"wordcloud": fig, "tfidf": fig, ...}
) -> str:
    """Gera string HTML completa do relatório."""

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    preview = texto_original[:500].replace("<", "&lt;").replace(">", "&gt;")
    if len(texto_original) > 500:
        preview += "…"

    # --- Métricas ---
    media = metrics.get("media", 0)
    mediana = metrics.get("mediana", 0)
    desvio = metrics.get("desvio_padrao", 0)
    total_tokens = sum(word_counts.values()) if word_counts else 0
    vocab_size = len(word_counts) if word_counts else 0

    # --- Top palavras ---
    top_words_rows = ""
    if word_counts:
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1])[:15]:
            top_words_rows += f"<tr><td>{word}</td><td>{count}</td></tr>"

    # --- TF-IDF ---
    tfidf_rows = ""
    if df_tfidf is not None and not df_tfidf.empty:
        df = df_tfidf.reset_index().iloc[:, 1:] if "index" in df_tfidf.reset_index().columns else df_tfidf
        for _, row in df.head(10).iterrows():
            cols_vals = "".join(f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>" for v in row)
            tfidf_rows += f"<tr>{cols_vals}</tr>"
        tfidf_header = "".join(f"<th>{c}</th>" for c in df.head(1).columns)
    else:
        tfidf_header = "<th>Termo</th><th>TF-IDF</th>"

    # --- Resumo ---
    resumo_html = ""
    if resumo:
        items = "".join(f"<li>{s}</li>" for s in resumo)
        resumo_html = f"<ol>{items}</ol>"
    else:
        resumo_html = "<p><em>Resumo não gerado.</em></p>"

    # --- Sentimento ---
    sent_geral_html = ""
    sent_chunks_html = ""
    if sentimento_geral:
        cor = {"Positivo": "#22c55e", "Negativo": "#ef4444", "Neutro": "#a3a3a3"}.get(
            sentimento_geral.get("label", "Neutro"), "#a3a3a3"
        )
        sent_geral_html = f"""
        <div style="display:inline-block;padding:12px 24px;border-radius:12px;
                    background:{cor}22;border:2px solid {cor};margin-bottom:16px;">
          <span style="font-size:2rem;">{sentimento_geral.get('emoji','')}</span>
          <strong style="color:{cor};font-size:1.2rem;margin-left:8px;">
            {sentimento_geral.get('label','')}
          </strong>
          <span style="color:#666;margin-left:8px;">
            ({sentimento_geral.get('score',0):.1%})
          </span>
        </div>"""

    if sentimentos:
        for s in sentimentos:
            cor = {"Positivo": "#22c55e", "Negativo": "#ef4444", "Neutro": "#a3a3a3"}.get(s.get("label"), "#a3a3a3")
            trecho = s.get("texto", "")[:120].replace("<","&lt;").replace(">","&gt;")
            sent_chunks_html += f"""
            <tr>
              <td style="border-left:4px solid {cor};padding-left:8px;">{trecho}{"…" if len(s.get("texto",""))>120 else ""}</td>
              <td style="color:{cor};font-weight:600;">{s.get('emoji','')} {s.get('label','')}</td>
              <td>{s.get('score',0):.1%}</td>
            </tr>"""

    # --- Frases-chave ---
    kp_html = ""
    if keyphrases:
        kp_html = "".join(
            f'<span style="display:inline-block;margin:4px;padding:6px 12px;'
            f'border-radius:20px;background:#6366f122;border:1px solid #6366f1;'
            f'font-size:0.9rem;">'
            f'{kp.get("frase","")} <em style="color:#6366f1;">({kp.get("score",0):.2f})</em></span>'
            for kp in keyphrases
        )

    # --- Classificação ---
    class_html = ""
    if classificacao and len(classificacao) > 0:
        top = classificacao[0]
        class_html = f"<p><strong>Gênero:</strong> {top.get('label','')} — {top.get('score',0):.1%} de confiança</p>"

    # --- Imagens ---
    img_wordcloud = _img_tag(_fig_to_b64(figs.get("wordcloud")) if figs else None, "Nuvem de Palavras")
    img_tfidf     = _img_tag(_fig_to_b64(figs.get("tfidf"))     if figs else None, "TF-IDF")
    img_freq      = _img_tag(_fig_to_b64(figs.get("freq"))      if figs else None, "Frequência de Palavras")

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Relatório de Análise de Texto</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f8fafc;color:#1e293b;padding:32px}}
    h1{{font-size:2rem;color:#4f46e5;margin-bottom:4px}}
    h2{{font-size:1.3rem;color:#4f46e5;margin:32px 0 12px;border-bottom:2px solid #e2e8f0;padding-bottom:6px}}
    h3{{font-size:1rem;color:#64748b;margin:16px 0 8px}}
    .meta{{color:#64748b;font-size:0.85rem;margin-bottom:32px}}
    .card{{background:#fff;border-radius:12px;padding:24px;margin-bottom:24px;
           box-shadow:0 1px 4px #0001}}
    .metrics{{display:flex;gap:16px;flex-wrap:wrap}}
    .metric{{background:#f1f5f9;border-radius:8px;padding:16px 24px;text-align:center;flex:1;min-width:120px}}
    .metric .val{{font-size:1.8rem;font-weight:700;color:#4f46e5}}
    .metric .lbl{{font-size:0.8rem;color:#64748b;margin-top:4px}}
    table{{width:100%;border-collapse:collapse;font-size:0.9rem}}
    th{{background:#f1f5f9;padding:8px 12px;text-align:left;font-weight:600}}
    td{{padding:8px 12px;border-bottom:1px solid #f1f5f9}}
    ol li{{margin:6px 0;line-height:1.6}}
    .preview{{background:#f8fafc;border-left:4px solid #6366f1;padding:12px 16px;
              border-radius:0 8px 8px 0;font-size:0.9rem;color:#475569;
              white-space:pre-wrap;word-break:break-word}}
    footer{{text-align:center;color:#94a3b8;font-size:0.8rem;margin-top:32px}}
  </style>
</head>
<body>
  <h1>📊 Relatório de Análise de Texto</h1>
  <p class="meta">Gerado em {now} · Analisador de Texto</p>

  <!-- PRÉVIA -->
  <div class="card">
    <h2>📄 Prévia do Texto</h2>
    <p class="preview">{preview}</p>
  </div>

  <!-- MÉTRICAS -->
  <div class="card">
    <h2>📏 Métricas Estatísticas</h2>
    <div class="metrics">
      <div class="metric"><div class="val">{total_tokens:,}</div><div class="lbl">Tokens</div></div>
      <div class="metric"><div class="val">{vocab_size:,}</div><div class="lbl">Vocabulário único</div></div>
      <div class="metric"><div class="val">{media:.2f}</div><div class="lbl">Média (caracteres/palavra)</div></div>
      <div class="metric"><div class="val">{mediana:.2f}</div><div class="lbl">Mediana</div></div>
      <div class="metric"><div class="val">{desvio:.2f}</div><div class="lbl">Desvio padrão</div></div>
    </div>
  </div>

  <!-- CLASSIFICAÇÃO -->
  {"<div class='card'><h2>🏷️ Classificação de Gênero</h2>" + class_html + "</div>" if class_html else ""}

  <!-- SENTIMENTO -->
  <div class="card">
    <h2>💬 Análise de Sentimento</h2>
    {sent_geral_html}
    {"<table><thead><tr><th>Trecho</th><th>Sentimento</th><th>Confiança</th></tr></thead><tbody>" + sent_chunks_html + "</tbody></table>" if sent_chunks_html else "<p><em>Análise de sentimento não realizada.</em></p>"}
  </div>

  <!-- FRASES-CHAVE -->
  <div class="card">
    <h2>🔑 Frases-Chave</h2>
    {kp_html if kp_html else "<p><em>Extração de frases-chave não realizada.</em></p>"}
  </div>

  <!-- RESUMO -->
  <div class="card">
    <h2>📝 Resumo Automático</h2>
    {resumo_html}
  </div>

  <!-- TOP PALAVRAS -->
  <div class="card">
    <h2>📊 Top 15 Palavras Mais Frequentes</h2>
    <table>
      <thead><tr><th>Palavra</th><th>Ocorrências</th></tr></thead>
      <tbody>{top_words_rows}</tbody>
    </table>
  </div>

  <!-- TF-IDF -->
  <div class="card">
    <h2>🔎 Top 10 Termos TF-IDF</h2>
    <table>
      <thead><tr>{tfidf_header}</tr></thead>
      <tbody>{tfidf_rows}</tbody>
    </table>
  </div>

  <!-- VISUALIZAÇÕES -->
  <div class="card">
    <h2>🖼️ Visualizações</h2>
    <h3>Nuvem de Palavras</h3>{img_wordcloud}
    <h3 style="margin-top:20px;">Frequência de Palavras</h3>{img_freq}
    <h3 style="margin-top:20px;">TF-IDF</h3>{img_tfidf}
  </div>

  <footer>Gerado pelo Analisador de Texto · {now}</footer>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Exportação PDF
# ---------------------------------------------------------------------------

def _html_to_pdf_pdfkit(html: str, output_path: str) -> bool:
    """Tenta gerar PDF via pdfkit (wkhtmltopdf)."""
    try:
        import pdfkit  # type: ignore
        pdfkit.from_string(html, output_path)
        return True
    except Exception:
        return False


def _html_to_pdf_weasyprint(html: str, output_path: str) -> bool:
    """Fallback: geração via WeasyPrint."""
    try:
        from weasyprint import HTML  # type: ignore
        HTML(string=html).write_pdf(output_path)
        return True
    except Exception:
        return False


def exportar_pdf(html: str, output_path: str | None = None) -> bytes | None:
    """
    Gera o PDF a partir do HTML e retorna os bytes.
    Tenta pdfkit primeiro, depois WeasyPrint.
    Retorna None se nenhum backend estiver disponível.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if _html_to_pdf_pdfkit(html, tmp_path) or _html_to_pdf_weasyprint(html, tmp_path):
            with open(tmp_path, "rb") as f:
                return f.read()
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
