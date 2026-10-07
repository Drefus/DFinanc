"""
Módulo de GERAÇÃO DE RELATÓRIOS.

Produz três saídas a partir dos resultados calculados:
1. relatorio_markowitz.html -> relatório completo, com CSS embutido e
   gráficos embutidos em base64 (um único arquivo que abre em qualquer
   navegador, mesmo se for enviado sozinho por e-mail).
2. carteiras_otimas.csv     -> pesos e métricas das carteiras (abre no Excel).
3. Resumo impresso no terminal.

A seção de INTERPRETAÇÃO do relatório é gerada automaticamente a partir dos
números: o texto muda conforme os dados (ex.: cita o par de ações mais
correlacionado, quanto risco a diversificação eliminou etc.).
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .graficos import CORES_CARTEIRAS, MAPA_DIVERGENTE

# ---------------------------------------------------------------------------
# Estrutura com todos os resultados da análise
# ---------------------------------------------------------------------------


@dataclass
class ResultadosAnalise:
    """Agrupa todos os números que o relatório precisa exibir."""

    parametros: dict                 # tickers, período, simulações, semente...
    dados: object                    # DadosMercado (coleta_dados.py)
    estatisticas: pd.DataFrame       # métricas por ativo
    correlacao: pd.DataFrame         # matriz de correlação
    covariancia: pd.DataFrame        # matriz de covariância anualizada
    pesos: pd.DataFrame              # linhas = ativos, colunas = carteiras
    metricas_carteiras: pd.DataFrame # retorno/vol/Sharpe esperados de cada carteira
    desempenho: pd.DataFrame         # métricas do backtest (inclui Ibovespa)
    diversificacao: dict             # efeito da diversificação por carteira
    simulacao: pd.DataFrame          # carteiras de Monte Carlo
    graficos: dict                   # nome -> caminho do PNG


# ---------------------------------------------------------------------------
# Formatação de números no padrão brasileiro
# ---------------------------------------------------------------------------

def pct(valor: float, casas: int = 2) -> str:
    """0.1234 -> '12,34%' (vírgula como separador decimal)."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "-"
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def num(valor: float, casas: int = 2) -> str:
    """1.2345 -> '1,23'."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "-"
    return f"{valor:.{casas}f}".replace(".", ",")


def curto(ticker: str) -> str:
    """'PETR4.SA' -> 'PETR4'."""
    return ticker.replace(".SA", "")


def _imagem_base64(caminho: Path) -> str:
    """
    Converte um PNG em texto base64 para embutir no HTML.

    Assim a imagem fica DENTRO do arquivo HTML (<img src="data:image/png;base64,...">)
    e o relatório não depende dos arquivos PNG estarem na mesma pasta.
    """
    codificado = base64.b64encode(Path(caminho).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{codificado}"


def _cor_correlacao(valor: float) -> tuple[str, str]:
    """Cor de fundo (mesma escala do heatmap) e cor do texto para a tabela HTML."""
    r, g, b, _ = MAPA_DIVERGENTE((valor + 1) / 2)  # escala -1..1 -> 0..1
    fundo = f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
    texto = "#ffffff" if abs(valor) > 0.6 else "#0b0b0b"
    return fundo, texto


# ---------------------------------------------------------------------------
# Interpretação automática dos resultados
# ---------------------------------------------------------------------------

def gerar_interpretacao(r: ResultadosAnalise) -> list[tuple[str, str]]:
    """
    Gera parágrafos explicativos em linguagem simples.

    Retorna uma lista de (título, texto_html). Cada parágrafo é construído com
    os números reais da análise, para que a explicação corresponda aos dados.
    """
    est = r.estatisticas
    selic = r.dados.selic_anual
    corr = r.correlacao
    m = r.metricas_carteiras
    blocos = []

    # --- 1) Diversificação ---------------------------------------------------
    d_iguais = r.diversificacao["Pesos Iguais"]
    d_minvar = r.diversificacao["Mínima Variância"]
    menos_arriscado = est["volatilidade"].idxmin()
    vol_menor_ativo = est["volatilidade"].min()
    texto = (
        f"Se as ações se movessem sempre juntas (correlação +1), uma carteira com pesos iguais "
        f"teria risco igual à média dos riscos individuais: <b>{pct(d_iguais['vol_ponderada'])}</b> ao ano. "
        f"Como as correlações são menores que 1, as quedas de umas são parcialmente compensadas "
        f"pelas altas de outras, e o risco real dessa carteira é de apenas "
        f"<b>{pct(d_iguais['vol_real'])}</b>, uma redução de <b>{pct(d_iguais['reducao'], 1)}</b> "
        f"obtida só pela diversificação, sem abrir mão de retorno médio. "
    )
    if d_minvar["vol_real"] < vol_menor_ativo:
        texto += (
            f"Mais ainda: a carteira de mínima variância tem volatilidade de "
            f"<b>{pct(d_minvar['vol_real'])}</b>, <b>menor que a do ativo menos arriscado</b> "
            f"isoladamente ({curto(menos_arriscado)}, {pct(vol_menor_ativo)}). "
            "Combinar ativos arriscados pode resultar em algo menos arriscado do que cada um deles."
        )
    blocos.append(("Por que a diversificação reduziu o risco?", texto))

    # --- 2) Correlações ------------------------------------------------------
    # Pega apenas o triângulo acima da diagonal (cada par uma única vez).
    mascara = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pares = corr.where(mascara).stack()
    par_max, par_min = pares.idxmax(), pares.idxmin()
    texto = (
        f"A correlação média entre os pares de ações foi de <b>{num(pares.mean())}</b>. "
        f"O par mais correlacionado é <b>{curto(par_max[0])} x {curto(par_max[1])}</b> "
        f"({num(pares.max())}): juntos, oferecem pouca proteção um ao outro. "
        f"O par menos correlacionado é <b>{curto(par_min[0])} x {curto(par_min[1])}</b> "
        f"({num(pares.min())}), a combinação que mais contribui para diluir o risco. "
        "Ações do mesmo setor (por exemplo, dois bancos) costumam ter correlação alta, "
        "porque reagem aos mesmos fatores econômicos."
    )
    blocos.append(("O que a matriz de correlação mostra?", texto))

    # --- 3) Beta -------------------------------------------------------------
    agressivas = [curto(t) for t, b in est["beta"].items() if b > 1]
    defensivas = [curto(t) for t, b in est["beta"].items() if b <= 1]
    texto = (
        "O beta mede quanto cada ação tende a oscilar quando o Ibovespa oscila 1%. "
        + (f"Com beta acima de 1 (amplificam o mercado): <b>{', '.join(agressivas)}</b>. " if agressivas else "")
        + (f"Com beta até 1 (mais defensivas): <b>{', '.join(defensivas)}</b>. " if defensivas else "")
        + f"A ação de maior beta é {curto(est['beta'].idxmax())} ({num(est['beta'].max())}) e a de menor é "
        f"{curto(est['beta'].idxmin())} ({num(est['beta'].min())}). O beta representa o risco "
        "<i>sistemático</i> (de mercado), que não desaparece com a diversificação; o que a diversificação "
        "elimina é o risco <i>específico</i> de cada empresa."
    )
    blocos.append(("Como cada ação reage ao mercado (beta)?", texto))

    # --- 4) Sharpe e carteira tangente --------------------------------------
    ms = m.loc["Máximo Sharpe"]
    melhor_ativo = est["sharpe"].idxmax()
    abaixo_selic = [curto(t) for t, v in est["retorno_anual"].items() if v < selic]
    pesos_ms = r.pesos["Máximo Sharpe"]
    principais = pesos_ms[pesos_ms > 0.005].sort_values(ascending=False)
    zerados = [curto(t) for t, v in pesos_ms.items() if v <= 0.005]
    texto = (
        f"Usando a Selic de <b>{pct(selic)}</b> ao ano como taxa livre de risco, a carteira de "
        f"máximo Sharpe atinge índice <b>{num(ms['sharpe'])}</b>, contra {num(est['sharpe'].max())} "
        f"do melhor ativo isolado ({curto(melhor_ativo)}). Ela é composta por "
        + ", ".join(f"{curto(t)} ({pct(v, 1)})" for t, v in principais.items())
        + ". "
    )
    # Escolhe singular ou plural conforme a quantidade de ações citadas.
    if zerados:
        texto += (
            (f"A ação {zerados[0]} ficou com peso zero" if len(zerados) == 1
             else f"As ações {', '.join(zerados)} ficaram com peso zero")
            + ": no período analisado, não acrescentaram retorno suficiente para compensar "
            "o risco que trariam à carteira. "
        )
    if abaixo_selic:
        texto += (
            (f"Note que {abaixo_selic[0]} teve" if len(abaixo_selic) == 1
             else f"Note que {', '.join(abaixo_selic)} tiveram")
            + " retorno médio abaixo da Selic, ou seja, Sharpe negativo: o investidor teria "
            "corrido risco de ações para ganhar menos que um título público. "
        )
    if ms["sharpe"] < 0:
        texto += ("Como nem a melhor carteira superou a Selic, o resultado indica que, nesse período, "
                  "a renda fixa pós-fixada foi mais vantajosa que essas ações. ")
    blocos.append(("Qual a melhor relação risco-retorno (Sharpe)?", texto))

    # --- 5) Monte Carlo x otimização ----------------------------------------
    sim = r.simulacao
    melhor_sim = sim["sharpe"].max()
    # f"{n:,}" usa vírgula como separador de milhar (padrão americano);
    # trocamos por ponto para seguir o padrão brasileiro (10.000).
    qtd = f"{len(sim):,}".replace(",", ".")
    texto = (
        f"Das {qtd} carteiras sorteadas aleatoriamente, a melhor obteve Sharpe de "
        f"{num(melhor_sim)}, enquanto a otimização matemática encontrou {num(ms['sharpe'])}. "
        "Todas as carteiras sorteadas ficam abaixo ou sobre a fronteira eficiente (linha preta): "
        "isso confirma visualmente que a fronteira representa o melhor retorno possível para cada "
        "nível de risco. A simulação ajuda a visualizar o espaço de possibilidades, mas a otimização "
        "encontra o ponto ótimo de forma exata."
    )
    blocos.append(("Simulação de Monte Carlo x otimização", texto))

    # --- 6) Desempenho histórico --------------------------------------------
    des = r.desempenho
    vencedora = des["retorno_total"].idxmax()
    ibov = des.loc["Ibovespa"]
    texto = (
        f"No período, o Ibovespa rendeu <b>{pct(ibov['retorno_total'], 1)}</b> no total "
        f"({pct(ibov['retorno_anualizado'], 1)} ao ano). A carteira de pesos iguais rendeu "
        f"{pct(des.loc['Pesos Iguais', 'retorno_total'], 1)}, a de mínima variância "
        f"{pct(des.loc['Mínima Variância', 'retorno_total'], 1)} (com volatilidade de "
        f"{pct(des.loc['Mínima Variância', 'volatilidade'], 1)}) e a de máximo Sharpe "
        f"{pct(des.loc['Máximo Sharpe', 'retorno_total'], 1)}. O maior retorno foi de "
        f"<b>{vencedora}</b>. "
        "<b>Atenção:</b> as carteiras ótimas foram calculadas com os mesmos dados usados nesse teste "
        "(análise <i>dentro da amostra</i>). Por isso seu desempenho tende a parecer melhor do que seria "
        "no futuro: o modelo \"conhece\" o passado. Retornos passados não garantem retornos futuros."
    )
    blocos.append(("Como as carteiras se saíram contra o mercado?", texto))
    return blocos


# ---------------------------------------------------------------------------
# Montagem do HTML
# ---------------------------------------------------------------------------

CSS = """
:root { --fundo:#f6f5f2; --cartao:#ffffff; --texto:#1d1d1b; --sec:#5b5a56; --borda:#e3e1db;
        --destaque:#1c5cab; --aviso:#fff4d6; --aviso-borda:#e0b440; }
* { box-sizing:border-box; }
body { margin:0; background:var(--fundo); color:var(--texto);
       font-family:"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height:1.55; }
.pagina { max-width:1100px; margin:0 auto; padding:32px 20px 60px; }
header { background:linear-gradient(135deg,#0d366b,#2a78d6); color:#fff; border-radius:14px;
         padding:28px 32px; margin-bottom:24px; }
header h1 { margin:0 0 6px; font-size:1.8rem; }
header p { margin:2px 0; opacity:.92; }
section { background:var(--cartao); border:1px solid var(--borda); border-radius:12px;
          padding:22px 26px; margin-bottom:20px; }
h2 { margin:0 0 14px; font-size:1.3rem; color:var(--destaque); }
h3 { margin:18px 0 6px; font-size:1.05rem; }
p { margin:8px 0; }
.tabela-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.92rem; margin:10px 0; }
th, td { padding:8px 10px; text-align:right; border-bottom:1px solid var(--borda); white-space:nowrap; }
th { background:#f0efea; color:var(--sec); font-weight:600; }
th:first-child, td:first-child { text-align:left; }
tr:hover td { background:#fafaf7; }
td.correl { text-align:center; font-weight:600; border:2px solid #fff; }
img { max-width:100%; height:auto; display:block; margin:12px auto; border-radius:8px; }
.grade-param { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
.param { background:#f7f6f2; border-radius:10px; padding:10px 14px; }
.param span { display:block; color:var(--sec); font-size:.82rem; text-transform:uppercase; letter-spacing:.03em; }
.param b { font-size:1.02rem; }
.aviso { background:var(--aviso); border:1px solid var(--aviso-borda); border-radius:10px;
         padding:12px 16px; margin-bottom:20px; }
.aviso ul { margin:6px 0 0 18px; padding:0; }
.conceito { border-left:4px solid var(--destaque); padding:4px 14px; margin:10px 0; background:#f7f9fd; }
.formula { font-family:Consolas, "Courier New", monospace; background:#f0efea; padding:2px 6px; border-radius:4px; }
.bolinha { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.nota { color:var(--sec); font-size:.88rem; }
footer { text-align:center; color:var(--sec); font-size:.85rem; margin-top:30px; }
"""


def _tabela_html(df: pd.DataFrame, formatos: dict, rotulo_indice: str,
                 nomes_colunas: dict, marcar_carteira: bool = False) -> str:
    """
    Converte um DataFrame em tabela HTML com formatação brasileira.

    formatos: coluna -> função de formatação (ex.: pct, num)
    nomes_colunas: coluna -> título legível
    marcar_carteira: se True, coloca uma bolinha colorida antes do nome da linha
    """
    cab = "".join(f"<th>{html.escape(nomes_colunas.get(c, c))}</th>" for c in df.columns)
    linhas = []
    for indice, linha in df.iterrows():
        nome = html.escape(curto(str(indice)))
        if marcar_carteira and indice in CORES_CARTEIRAS:
            nome = f'<span class="bolinha" style="background:{CORES_CARTEIRAS[indice]}"></span>{nome}'
        celulas = "".join(f"<td>{formatos.get(c, num)(linha[c])}</td>" for c in df.columns)
        linhas.append(f"<tr><td>{nome}</td>{celulas}</tr>")
    return (f'<div class="tabela-wrap"><table><thead><tr><th>{html.escape(rotulo_indice)}</th>{cab}'
            f'</tr></thead><tbody>{"".join(linhas)}</tbody></table></div>')


def _tabela_correlacao_html(corr: pd.DataFrame) -> str:
    """Tabela da matriz de correlação com células coloridas (mesma escala do heatmap)."""
    cab = "".join(f"<th style='text-align:center'>{curto(c)}</th>" for c in corr.columns)
    linhas = []
    for i in corr.index:
        celulas = []
        for j in corr.columns:
            fundo, texto = _cor_correlacao(corr.loc[i, j])
            celulas.append(f'<td class="correl" style="background:{fundo};color:{texto}">'
                           f'{num(corr.loc[i, j])}</td>')
        linhas.append(f"<tr><td><b>{curto(i)}</b></td>{''.join(celulas)}</tr>")
    return (f'<div class="tabela-wrap"><table><thead><tr><th></th>{cab}</tr></thead>'
            f'<tbody>{"".join(linhas)}</tbody></table></div>')


def gerar_html(r: ResultadosAnalise, caminho: Path) -> Path:
    """Monta o relatório HTML completo e salva em 'caminho'."""
    p = r.parametros
    d = r.dados
    img = {nome: _imagem_base64(arq) for nome, arq in r.graficos.items()}

    # ----- Avisos (fallback, ativos sem dados etc.) -----
    avisos_html = ""
    if d.avisos:
        itens = "".join(f"<li>{html.escape(a)}</li>" for a in d.avisos)
        avisos_html = f'<div class="aviso"><b>Avisos da execução</b><ul>{itens}</ul></div>'

    # ----- Parâmetros -----
    qtd_sim = f"{p['simulacoes']:,}".replace(",", ".")  # 10000 -> "10.000"
    parametros = [
        ("Ativos analisados", ", ".join(curto(t) for t in r.estatisticas.index)),
        ("Referência de mercado", "Ibovespa (^BVSP)"),
        ("Período", f"{p['inicio_efetivo']} a {p['fim_efetivo']}"),
        ("Pregões utilizados", f"{p['n_pregoes']}"),
        ("Data da coleta", d.data_coleta),
        ("Selic meta (taxa livre de risco)", f"{pct(d.selic_anual)} a.a. (vigente em {d.data_selic})"),
        ("Fonte das cotações", d.fonte_precos),
        ("Fonte da Selic", d.fonte_selic),
        ("Carteiras simuladas (Monte Carlo)", qtd_sim),
        ("Semente aleatória", str(p["semente"])),
        ("Pontos da fronteira eficiente", str(p["pontos_fronteira"])),
        ("Base de anualização", "252 dias úteis"),
    ]
    grade_param = "".join(f'<div class="param"><span>{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>'
                          for k, v in parametros)

    # ----- Tabelas -----
    tab_ativos = _tabela_html(
        r.estatisticas,
        {"retorno_anual": pct, "volatilidade": pct, "sharpe": num, "beta": num,
         "retorno_periodo": lambda v: pct(v, 1), "max_drawdown": lambda v: pct(v, 1)},
        "Ativo",
        {"retorno_anual": "Retorno esperado a.a.", "volatilidade": "Volatilidade a.a.",
         "sharpe": "Sharpe", "beta": "Beta", "retorno_periodo": "Retorno no período",
         "max_drawdown": "Máx. drawdown"},
    )
    tab_pesos = _tabela_html(
        r.pesos.T,  # transposta: uma linha por carteira, uma coluna por ativo
        {c: (lambda v: pct(v, 1)) for c in r.pesos.index},
        "Carteira", {c: curto(c) for c in r.pesos.index}, marcar_carteira=True,
    )
    tab_metricas = _tabela_html(
        r.metricas_carteiras,
        {"retorno": pct, "volatilidade": pct, "sharpe": num},
        "Carteira", {"retorno": "Retorno esperado a.a.", "volatilidade": "Volatilidade a.a.",
                     "sharpe": "Sharpe"}, marcar_carteira=True,
    )
    tab_desempenho = _tabela_html(
        r.desempenho,
        {"retorno_total": lambda v: pct(v, 1), "retorno_anualizado": lambda v: pct(v, 1),
         "volatilidade": pct, "sharpe": num, "max_drawdown": lambda v: pct(v, 1)},
        "Carteira", {"retorno_total": "Retorno total", "retorno_anualizado": "Retorno anualizado (composto)",
                     "volatilidade": "Volatilidade a.a.", "sharpe": "Sharpe",
                     "max_drawdown": "Máx. drawdown"}, marcar_carteira=True,
    )

    interpretacao = "".join(f"<h3>{html.escape(t)}</h3><p>{texto}</p>" for t, texto in gerar_interpretacao(r))

    documento = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório - Carteiras de Markowitz (B3)</title>
<style>{CSS}</style>
</head>
<body>
<div class="pagina">

<header>
  <h1>Risco e Retorno: Teoria Moderna de Carteiras</h1>
  <p>Aplicação prática do modelo de Markowitz com ações da B3</p>
  <p>Gerado em {html.escape(d.data_coleta)} &middot; Selic utilizada: {pct(d.selic_anual)} a.a.</p>
</header>

{avisos_html}

<section>
  <h2>1. Parâmetros da análise</h2>
  <div class="grade-param">{grade_param}</div>
</section>

<section>
  <h2>2. Conceitos utilizados</h2>
  <div class="conceito"><b>Retorno:</b> variação percentual diária do preço ajustado,
    <span class="formula">r = P<sub>t</sub>/P<sub>t-1</sub> - 1</span>. O retorno esperado é a média histórica anualizada (x 252).</div>
  <div class="conceito"><b>Risco (volatilidade):</b> desvio-padrão dos retornos, anualizado por
    <span class="formula">&sigma;<sub>anual</sub> = &sigma;<sub>diário</sub> x &radic;252</span>. Mede o quanto o retorno oscila.</div>
  <div class="conceito"><b>Covariância e correlação:</b> medem se dois ativos se movem juntos. O risco da carteira é
    <span class="formula">&sigma;<sub>p</sub><sup>2</sup> = w<sup>T</sup> &Sigma; w</span>; quanto menor a correlação, maior o benefício da diversificação.</div>
  <div class="conceito"><b>Beta:</b> sensibilidade ao Ibovespa,
    <span class="formula">&beta; = Cov(r<sub>i</sub>, r<sub>m</sub>) / Var(r<sub>m</sub>)</span>. Mede o risco sistemático, que não se elimina diversificando.</div>
  <div class="conceito"><b>Índice de Sharpe:</b> retorno excedente por unidade de risco,
    <span class="formula">(R<sub>p</sub> - R<sub>f</sub>) / &sigma;<sub>p</sub></span>, com R<sub>f</sub> = Selic.</div>
  <div class="conceito"><b>Fronteira eficiente:</b> conjunto de carteiras com o maior retorno possível para cada nível de risco.
    Restrições: pesos entre 0% e 100% (sem venda a descoberto) e soma igual a 100%.</div>
</section>

<section>
  <h2>3. Estatísticas por ativo</h2>
  {tab_ativos}
  <p class="nota">Retorno e volatilidade anualizados a partir dos retornos diários. Sharpe calculado com a Selic de {pct(d.selic_anual)}.
  Máx. drawdown = maior queda do pico ao vale no período.</p>
  <img src="{img['precos']}" alt="Gráfico de preços normalizados em base 100">
</section>

<section>
  <h2>4. Matriz de correlação</h2>
  {_tabela_correlacao_html(r.correlacao)}
  <img src="{img['correlacao']}" alt="Mapa de calor da correlação" style="max-width:640px">
</section>

<section>
  <h2>5. Simulação de Monte Carlo e fronteira eficiente</h2>
  <p>Foram sorteadas {qtd_sim} carteiras com pesos aleatórios (semente {p['semente']}). Cada ponto é uma carteira,
  colorida pelo índice de Sharpe. A linha preta é a fronteira eficiente calculada por otimização (scipy, método SLSQP).</p>
  <img src="{img['fronteira']}" alt="Nuvem de carteiras simuladas e fronteira eficiente">
</section>

<section>
  <h2>6. Carteiras ótimas</h2>
  <h3>Pesos</h3>
  {tab_pesos}
  <h3>Métricas esperadas (com base nas estatísticas históricas)</h3>
  {tab_metricas}
  <img src="{img['pesos']}" alt="Composição das carteiras ótimas">
</section>

<section>
  <h2>7. Desempenho histórico x Ibovespa</h2>
  <p>Evolução de R$ 100 investidos no início do período, mantendo os pesos constantes (rebalanceamento diário).</p>
  <img src="{img['desempenho']}" alt="Evolução acumulada das carteiras e do Ibovespa">
  {tab_desempenho}
</section>

<section>
  <h2>8. Interpretação dos resultados</h2>
  {interpretacao}
</section>

<section>
  <h2>9. Limitações do modelo</h2>
  <p>&bull; O modelo usa médias e covariâncias <b>passadas</b> como estimativa do futuro; essas estimativas mudam com o tempo,
  e a média de retornos é especialmente instável.<br>
  &bull; A Selic atual é aplicada a todo o período histórico, como simplificação.<br>
  &bull; Não são considerados custos de transação, impostos nem a necessidade de rebalancear a carteira.<br>
  &bull; A volatilidade trata altas e quedas como "risco" da mesma forma e supõe retornos aproximadamente normais.<br>
  &bull; Trabalho acadêmico: os resultados <b>não constituem recomendação de investimento</b>.</p>
</section>

<footer>Trabalho prático de Administração Financeira &middot; Sistemas de Informação &middot;
Dados: Yahoo Finance e Banco Central do Brasil (SGS 432)</footer>
</div>
</body>
</html>"""
    caminho.write_text(documento, encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# CSV de resultados
# ---------------------------------------------------------------------------

def gerar_csv_carteiras(r: ResultadosAnalise, caminho: Path) -> Path:
    """
    Salva pesos e métricas das carteiras em CSV.

    Usa ';' como separador e ',' como decimal (padrão do Excel em português),
    e codificação utf-8-sig para os acentos aparecerem corretamente no Excel.
    """
    tabela = r.pesos.T.add_prefix("peso_")
    tabela.columns = [c.replace(".SA", "") for c in tabela.columns]
    tabela = tabela.join(r.metricas_carteiras.add_suffix("_esperado"))
    tabela = tabela.join(r.desempenho.add_suffix("_historico"))
    tabela.index.name = "carteira"
    tabela.to_csv(caminho, sep=";", decimal=",", float_format="%.6f", encoding="utf-8-sig")
    return caminho


# ---------------------------------------------------------------------------
# Resumo no terminal
# ---------------------------------------------------------------------------

def imprimir_resumo(r: ResultadosAnalise, arquivos: list[Path]) -> None:
    """Imprime no terminal um resumo legível dos principais resultados."""
    d = r.dados
    linha = "=" * 78
    print("\n" + linha)
    print("  RESUMO DA ANÁLISE - TEORIA MODERNA DE CARTEIRAS (MARKOWITZ)")
    print(linha)
    print(f"  Período: {r.parametros['inicio_efetivo']} a {r.parametros['fim_efetivo']} "
          f"({r.parametros['n_pregoes']} pregões)")
    print(f"  Selic (taxa livre de risco): {pct(d.selic_anual)} a.a. (vigente em {d.data_selic})")
    print(f"  Fonte dos preços: {d.fonte_precos}")

    print("\n  ESTATÍSTICAS POR ATIVO (anualizadas)")
    print(f"  {'Ativo':<10}{'Retorno':>12}{'Volatilidade':>15}{'Sharpe':>10}{'Beta':>9}")
    for t, s in r.estatisticas.iterrows():
        print(f"  {curto(t):<10}{pct(s['retorno_anual']):>12}{pct(s['volatilidade']):>15}"
              f"{num(s['sharpe']):>10}{num(s['beta']):>9}")

    print("\n  PESOS DAS CARTEIRAS")
    cab = "".join(f"{curto(t):>9}" for t in r.pesos.index)
    print(f"  {'Carteira':<19}{cab}")
    for carteira in r.pesos.columns:
        valores = "".join(f"{pct(v, 1):>9}" for v in r.pesos[carteira])
        print(f"  {carteira:<19}{valores}")

    print("\n  MÉTRICAS ESPERADAS E DESEMPENHO HISTÓRICO")
    print(f"  {'Carteira':<19}{'Ret.esp.':>10}{'Vol.':>9}{'Sharpe':>8}{'Ret.total':>11}{'Máx.DD':>9}")
    for nome, des in r.desempenho.iterrows():
        if nome in r.metricas_carteiras.index:
            m = r.metricas_carteiras.loc[nome]
            ret_esp, vol, sh = pct(m["retorno"]), pct(m["volatilidade"]), num(m["sharpe"])
        else:  # Ibovespa: usa as métricas do próprio histórico
            ret_esp, vol, sh = "-", pct(des["volatilidade"]), num(des["sharpe"])
        print(f"  {nome:<19}{ret_esp:>10}{vol:>9}{sh:>8}{pct(des['retorno_total'], 1):>11}"
              f"{pct(des['max_drawdown'], 1):>9}")

    div = r.diversificacao["Pesos Iguais"]
    print(f"\n  Diversificação (pesos iguais): risco caiu de {pct(div['vol_ponderada'])} "
          f"para {pct(div['vol_real'])} ({pct(div['reducao'], 1)} a menos).")

    if d.avisos:
        print("\n  AVISOS:")
        for a in d.avisos:
            print(f"   - {a}")

    print("\n  ARQUIVOS GERADOS:")
    for arq in arquivos:
        print(f"   - {arq}")
    print(linha + "\n")
