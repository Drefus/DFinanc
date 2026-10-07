"""
Módulo de GRÁFICOS (matplotlib).

Gera os gráficos em PNG na pasta output/ e devolve os caminhos dos arquivos,
que depois são embutidos no relatório HTML.

Escolhas de visualização (para que os gráficos sejam fáceis de ler):
- Cores CATEGÓRICAS em ordem fixa para identificar ativos e carteiras;
  a mesma carteira tem sempre a mesma cor em todos os gráficos.
- Cor SEQUENCIAL (um só tom de azul, do claro ao escuro) para grandezas
  contínuas, como o índice de Sharpe na nuvem de carteiras.
- Cor DIVERGENTE (azul <-> cinza <-> vermelho) para a correlação, que vai de
  -1 a +1 e tem um ponto neutro (zero) no meio.
- Um único eixo Y por gráfico; grades e eixos discretos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend "Agg" desenha os gráficos direto em arquivo, sem abrir janelas.
# Isso evita erros em computadores sem interface gráfica configurada.
matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402  (import após matplotlib.use é intencional)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# ---------------------------------------------------------------------------
# Paleta de cores e estilo
# ---------------------------------------------------------------------------

SUPERFICIE = "#fcfcfb"      # fundo do gráfico
TEXTO = "#0b0b0b"           # texto principal
TEXTO_SEC = "#52514e"       # texto secundário (eixos, rótulos)
GRADE = "#e6e5e0"           # linhas de grade bem suaves
NEUTRO = "#8a8984"          # elementos neutros (ex.: ativos individuais)

# Ordem fixa de cores categóricas (testada para daltonismo).
CATEGORICAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Cor fixa de cada carteira (mesma em todos os gráficos e no relatório).
CORES_CARTEIRAS = {
    "Pesos Iguais": CATEGORICAS[0],      # azul
    "Máximo Sharpe": CATEGORICAS[1],     # laranja
    "Mínima Variância": CATEGORICAS[2],  # verde-água
    "Ibovespa": TEXTO_SEC,               # cinza escuro (referência de mercado)
}

# Escala sequencial (azul claro -> azul escuro) para o Sharpe.
MAPA_SEQUENCIAL = LinearSegmentedColormap.from_list(
    "azul_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Escala divergente (azul = correlação negativa, cinza = zero, vermelho = positiva).
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "correl", ["#104281", "#5598e7", "#f0efec", "#e66767", "#a32c2b"])

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": TEXTO,
    "axes.labelcolor": TEXTO_SEC,
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,      # grade atrás dos dados (barras, linhas)
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "xtick.color": TEXTO_SEC,
    "ytick.color": TEXTO_SEC,
    "legend.frameon": False,
    "text.color": TEXTO,
})


def _formato_pct(casas: int = 0) -> FuncFormatter:
    """Formatador de eixo: 0.12 -> '12%' (vírgula decimal, padrão brasileiro)."""
    return FuncFormatter(lambda v, _: f"{v:.{casas}f}%".replace(".", ","))


def _nome_curto(ticker: str) -> str:
    """'PETR4.SA' -> 'PETR4' (rótulos mais limpos nos gráficos)."""
    return ticker.replace(".SA", "")


def _salvar(fig, caminho: Path) -> Path:
    """Salva a figura em PNG (150 dpi) e libera a memória."""
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _espalhar_rotulos(valores: list[float], distancia_min: float) -> list[float]:
    """
    Ajusta posições verticais de rótulos para que não fiquem sobrepostos.

    Ordena os valores e empurra para cima cada rótulo que estiver mais perto
    que 'distancia_min' do anterior. Usado nos rótulos do fim das linhas.
    """
    ordem = np.argsort(valores)
    ajustados = np.array(valores, dtype=float)
    for k in range(1, len(ordem)):
        anterior, atual = ordem[k - 1], ordem[k]
        if ajustados[atual] - ajustados[anterior] < distancia_min:
            ajustados[atual] = ajustados[anterior] + distancia_min
    return ajustados.tolist()


# ---------------------------------------------------------------------------
# 1) Preços normalizados (base 100)
# ---------------------------------------------------------------------------

def grafico_precos_normalizados(precos: pd.DataFrame, mercado: pd.Series, pasta: Path) -> Path:
    """
    Evolução dos preços com base 100 na data inicial.

    Normalizar (P_t / P_0 * 100) coloca ativos de preços muito diferentes
    (ex.: R$ 10 e R$ 60) na mesma escala: o valor 150 significa +50% desde
    o início, independentemente do preço original.
    """
    base100 = precos / precos.iloc[0] * 100
    ibov = mercado / mercado.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for i, ticker in enumerate(base100.columns):
        ax.plot(base100.index, base100[ticker], lw=1.6,
                color=CATEGORICAS[i % len(CATEGORICAS)], label=_nome_curto(ticker))
    ax.plot(ibov.index, ibov, lw=2.2, ls="--", color=TEXTO, label="Ibovespa")
    ax.axhline(100, color=NEUTRO, lw=1, ls=":")

    ax.set_title("Preços ajustados normalizados (base 100)")
    ax.set_ylabel("Valor (início = 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.legend(ncol=min(7, len(base100.columns) + 1), loc="upper left",
              bbox_to_anchor=(0, -0.08))
    return _salvar(fig, pasta / "01_precos_normalizados.png")


# ---------------------------------------------------------------------------
# 2) Heatmap de correlação
# ---------------------------------------------------------------------------

def grafico_correlacao(correlacao: pd.DataFrame, pasta: Path) -> Path:
    """
    Mapa de calor da matriz de correlação.

    Tons de vermelho = correlação positiva forte (ativos andam juntos, pouca
    diversificação); tons de azul = correlação negativa; cinza = perto de zero.
    """
    n = len(correlacao)
    nomes = [_nome_curto(t) for t in correlacao.columns]
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.0 * n + 1.5))

    imagem = ax.imshow(correlacao.values, cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels=nomes, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=nomes)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)

    # Linhas brancas entre as células (pequeno espaçamento visual).
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Escreve o valor em cada célula; texto branco em células escuras.
    for i in range(n):
        for j in range(n):
            valor = correlacao.values[i, j]
            cor = "white" if abs(valor) > 0.6 else TEXTO
            ax.text(j, i, f"{valor:.2f}".replace(".", ","), ha="center",
                    va="center", color=cor, fontsize=10)

    barra = fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("Correlação")
    barra.outline.set_visible(False)
    ax.set_title("Matriz de correlação dos retornos diários")
    return _salvar(fig, pasta / "02_correlacao.png")


# ---------------------------------------------------------------------------
# 3) Nuvem de carteiras + fronteira eficiente
# ---------------------------------------------------------------------------

def grafico_fronteira(simulacao: pd.DataFrame, fronteira: pd.DataFrame,
                      estatisticas: pd.DataFrame, carteiras: dict[str, dict],
                      taxa_livre_risco: float, pasta: Path) -> Path:
    """
    Gráfico risco x retorno (o gráfico clássico de Markowitz).

    - Cada pontinho é uma carteira aleatória da simulação de Monte Carlo,
      colorida pelo índice de Sharpe (mais escuro = melhor).
    - A linha preta é a fronteira eficiente obtida por otimização: nenhuma
      carteira consegue ficar acima/à esquerda dela.
    - Pontos cinza são os ativos isolados: perceba que várias carteiras têm
      risco menor que qualquer ativo sozinho (efeito da diversificação).
    - A reta tracejada é a Linha de Mercado de Capitais (CML), que sai da
      Selic e tangencia a fronteira na carteira de máximo Sharpe.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    nuvem = ax.scatter(simulacao["volatilidade"] * 100, simulacao["retorno"] * 100,
                       c=simulacao["sharpe"], cmap=MAPA_SEQUENCIAL, s=6, alpha=0.7,
                       linewidths=0, rasterized=True)
    barra = fig.colorbar(nuvem, ax=ax, pad=0.02)
    barra.set_label("Índice de Sharpe")
    barra.outline.set_visible(False)

    ax.plot(fronteira["volatilidade"] * 100, fronteira["retorno"] * 100,
            color=TEXTO, lw=2.2, label="Fronteira eficiente", zorder=4)

    # Linha de Mercado de Capitais: R = rf + Sharpe_max * sigma
    tangente = carteiras["Máximo Sharpe"]
    x_max = max(simulacao["volatilidade"].max(), estatisticas["volatilidade"].max()) * 100
    x_cml = np.linspace(0, x_max, 50)
    y_cml = taxa_livre_risco * 100 + tangente["sharpe"] * x_cml
    ax.plot(x_cml, y_cml, ls="--", lw=1.4, color=NEUTRO,
            label="Linha de Mercado de Capitais (a partir da Selic)", zorder=3)

    # Ativos individuais
    ax.scatter(estatisticas["volatilidade"] * 100, estatisticas["retorno_anual"] * 100,
               s=55, color=NEUTRO, edgecolor=SUPERFICIE, linewidth=1.5, zorder=5,
               label="Ativos individuais")
    for ticker, linha in estatisticas.iterrows():
        ax.annotate(_nome_curto(ticker), (linha["volatilidade"] * 100, linha["retorno_anual"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=9, color=TEXTO_SEC)

    # Carteiras destacadas (formas diferentes + cor + rótulo: não dependem só da cor)
    marcadores = {"Mínima Variância": ("D", 130), "Máximo Sharpe": ("*", 380),
                  "Pesos Iguais": ("o", 110)}
    for nome, (marcador, tamanho) in marcadores.items():
        c = carteiras[nome]
        ax.scatter(c["volatilidade"] * 100, c["retorno"] * 100, marker=marcador, s=tamanho,
                   color=CORES_CARTEIRAS[nome], edgecolor=TEXTO, linewidth=1.2,
                   zorder=6, label=nome)
        ax.annotate(nome, (c["volatilidade"] * 100, c["retorno"] * 100), xytext=(10, -14),
                    textcoords="offset points", fontsize=9.5, fontweight="bold", color=TEXTO)

    # Enquadra o gráfico na região de interesse (sem esticar até risco zero).
    todas_vol = np.concatenate([simulacao["volatilidade"], estatisticas["volatilidade"]]) * 100
    todos_ret = np.concatenate([simulacao["retorno"], estatisticas["retorno_anual"]]) * 100
    margem_x = (todas_vol.max() - todas_vol.min()) * 0.08
    margem_y = (todos_ret.max() - todos_ret.min()) * 0.08
    ax.set_xlim(todas_vol.min() - margem_x, todas_vol.max() + margem_x)
    ax.set_ylim(todos_ret.min() - margem_y, todos_ret.max() + margem_y)

    ax.set_title(f"Fronteira eficiente e {len(simulacao):,} carteiras simuladas".replace(",", "."))
    ax.set_xlabel("Risco: volatilidade anualizada")
    ax.set_ylabel("Retorno esperado anualizado")
    ax.xaxis.set_major_formatter(_formato_pct())
    ax.yaxis.set_major_formatter(_formato_pct())
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), ncol=3, markerscale=0.8)
    return _salvar(fig, pasta / "03_fronteira_eficiente.png")


# ---------------------------------------------------------------------------
# 4) Pesos das carteiras ótimas
# ---------------------------------------------------------------------------

def grafico_pesos(pesos: pd.DataFrame, pasta: Path) -> Path:
    """
    Barras horizontais com a composição (pesos) de cada carteira ótima.

    'pesos' tem uma linha por ativo e uma coluna por carteira.
    """
    nomes = [_nome_curto(t) for t in pesos.index]
    n_carteiras = len(pesos.columns)
    altura = 0.8 / n_carteiras
    y = np.arange(len(pesos))

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(pesos) + 1.8))
    for k, carteira in enumerate(pesos.columns):
        posicoes = y + (k - (n_carteiras - 1) / 2) * altura
        valores = pesos[carteira].values * 100
        # A borda da cor do fundo cria um pequeno espaço entre barras vizinhas.
        ax.barh(posicoes, valores, height=altura, color=CORES_CARTEIRAS[carteira],
                edgecolor=SUPERFICIE, linewidth=2, label=carteira)
        for p, v in zip(posicoes, valores):
            if v >= 0.5:
                ax.text(v + 0.8, p, f"{v:.1f}%".replace(".", ","), va="center",
                        fontsize=8.5, color=TEXTO_SEC)

    ax.set_yticks(y, labels=nomes)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_formato_pct())
    ax.set_xlim(0, min(100, pesos.values.max() * 100 + 12))
    ax.set_xlabel("Peso na carteira")
    ax.set_title("Composição das carteiras ótimas")
    ax.legend(loc="lower right")
    return _salvar(fig, pasta / "04_pesos_carteiras.png")


# ---------------------------------------------------------------------------
# 5) Evolução acumulada das carteiras x Ibovespa
# ---------------------------------------------------------------------------

def grafico_desempenho(acumulado: pd.DataFrame, pasta: Path) -> Path:
    """
    Quanto R$ 100 investidos no início teriam se tornado em cada carteira.

    Os valores finais aparecem rotulados à direita de cada linha.
    """
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for nome in acumulado.columns:
        ax.plot(acumulado.index, acumulado[nome], lw=2.2 if nome == "Ibovespa" else 2,
                ls="--" if nome == "Ibovespa" else "-",
                color=CORES_CARTEIRAS.get(nome, NEUTRO), label=nome)
    ax.axhline(100, color=NEUTRO, lw=1, ls=":")

    # Rótulos com o valor final, espaçados para não se sobreporem.
    finais = acumulado.iloc[-1]
    amplitude = acumulado.max().max() - acumulado.min().min()
    posicoes = _espalhar_rotulos(finais.tolist(), amplitude * 0.045)
    for nome, y_rotulo in zip(finais.index, posicoes):
        ax.annotate(f"{nome}: {finais[nome]:.0f}", xy=(acumulado.index[-1], y_rotulo),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=TEXTO, annotation_clip=False)

    ax.set_title("Evolução de R$ 100 investidos: carteiras x Ibovespa")
    ax.set_ylabel("Valor da carteira (R$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.legend(loc="upper left")
    return _salvar(fig, pasta / "05_desempenho_historico.png")
