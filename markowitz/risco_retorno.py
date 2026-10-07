"""
Módulo de CÁLCULOS DE RISCO E RETORNO.

Conceitos financeiros implementados aqui:

RETORNO
    Retorno simples diário: r_t = (P_t / P_{t-1}) - 1
    Mede quanto o preço variou de um dia para o outro, em percentual.

RETORNO ESPERADO
    Na Teoria de Markowitz, o retorno esperado de um ativo é estimado pela
    MÉDIA dos retornos históricos. Para anualizar, multiplicamos a média
    diária por 252 (número aproximado de dias úteis/pregões em um ano).

RISCO (VOLATILIDADE)
    O risco é medido pelo DESVIO-PADRÃO dos retornos: quanto mais os retornos
    oscilam em torno da média, mais incerto (arriscado) é o investimento.
    Como a variância cresce proporcionalmente ao tempo, o desvio-padrão cresce
    com a RAIZ do tempo: sigma_anual = sigma_diario * sqrt(252).

COVARIÂNCIA E CORRELAÇÃO
    A covariância mede se dois ativos tendem a subir e cair juntos.
    A correlação é a covariância "padronizada", sempre entre -1 e +1:
        +1 -> movem-se exatamente juntos (diversificação nula)
         0 -> sem relação linear
        -1 -> movem-se em sentidos opostos (diversificação máxima)
    O risco de uma CARTEIRA depende da matriz de covariância:
        sigma_p^2 = w^T * Sigma * w
    Como a correlação entre ações geralmente é menor que 1, o risco da
    carteira é MENOR que a média ponderada dos riscos individuais.
    Esse é o efeito da DIVERSIFICAÇÃO.

BETA
    Mede a sensibilidade do ativo ao mercado (Ibovespa):
        beta = Cov(r_ativo, r_mercado) / Var(r_mercado)
    beta > 1: ação "agressiva" (amplifica os movimentos do mercado)
    beta < 1: ação "defensiva" (oscila menos que o mercado)
    O beta representa o RISCO SISTEMÁTICO, aquele que a diversificação
    NÃO consegue eliminar.

ÍNDICE DE SHARPE
    Retorno em excesso à taxa livre de risco por unidade de risco:
        Sharpe = (retorno_esperado - taxa_livre_de_risco) / volatilidade
    Responde à pergunta: "quanto de retorno extra eu ganho para cada
    unidade de risco que aceito correr?". Quanto maior, melhor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Número de pregões (dias úteis de negociação) em um ano no Brasil.
DIAS_UTEIS_ANO = 252


# ---------------------------------------------------------------------------
# Retornos e estatísticas básicas
# ---------------------------------------------------------------------------

def calcular_retornos(precos: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Calcula os retornos SIMPLES diários: (P_hoje / P_ontem) - 1.

    pct_change() faz exatamente essa conta para cada coluna. A primeira linha
    fica vazia (não existe "ontem" para o primeiro dia), por isso o dropna().
    """
    return precos.pct_change().dropna()


def retorno_esperado_anual(retornos: pd.DataFrame) -> pd.Series:
    """Média dos retornos diários x 252 = retorno esperado anualizado."""
    return retornos.mean() * DIAS_UTEIS_ANO


def volatilidade_anual(retornos: pd.DataFrame) -> pd.Series:
    """Desvio-padrão diário x raiz(252) = volatilidade anualizada."""
    return retornos.std() * np.sqrt(DIAS_UTEIS_ANO)


def matriz_covariancia_anual(retornos: pd.DataFrame) -> pd.DataFrame:
    """
    Matriz de covariância anualizada (covariância diária x 252).

    É uma matriz quadrada N x N: na diagonal ficam as variâncias de cada ativo
    e fora da diagonal as covariâncias entre pares de ativos.
    """
    return retornos.cov() * DIAS_UTEIS_ANO


def matriz_correlacao(retornos: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação de Pearson entre os retornos diários."""
    return retornos.corr()


def calcular_beta(retornos_ativo: pd.Series, retornos_mercado: pd.Series) -> float:
    """
    Beta = Cov(ativo, mercado) / Var(mercado).

    np.cov devolve a matriz 2x2 [[var_a, cov_am], [cov_am, var_m]];
    o elemento [0, 1] é a covariância entre o ativo e o mercado.
    """
    cov = np.cov(retornos_ativo, retornos_mercado, ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def estatisticas_por_ativo(retornos: pd.DataFrame, retornos_mercado: pd.Series,
                           precos: pd.DataFrame, taxa_livre_risco: float) -> pd.DataFrame:
    """
    Monta a tabela-resumo com as métricas de cada ação.

    Colunas:
        retorno_anual     -> retorno esperado anualizado (média x 252)
        volatilidade      -> risco anualizado (desvio-padrão x raiz(252))
        sharpe            -> (retorno - Selic) / volatilidade
        beta              -> sensibilidade ao Ibovespa
        retorno_periodo   -> variação total do preço no período analisado
        max_drawdown      -> maior queda do topo ao fundo no período
    """
    mu = retorno_esperado_anual(retornos)
    sigma = volatilidade_anual(retornos)

    tabela = pd.DataFrame({
        "retorno_anual": mu,
        "volatilidade": sigma,
        "sharpe": (mu - taxa_livre_risco) / sigma,
        "beta": {t: calcular_beta(retornos[t], retornos_mercado) for t in retornos.columns},
        "retorno_periodo": precos.iloc[-1] / precos.iloc[0] - 1,
        "max_drawdown": {t: maximo_drawdown(precos[t]) for t in precos.columns},
    })
    return tabela


# ---------------------------------------------------------------------------
# Métricas de carteira
# ---------------------------------------------------------------------------

def metricas_carteira(pesos: np.ndarray, mu: pd.Series | np.ndarray,
                      cov: pd.DataFrame | np.ndarray, taxa_livre_risco: float) -> dict:
    """
    Calcula retorno, volatilidade e Sharpe de uma carteira com os pesos dados.

    Fórmulas de Markowitz (notação matricial):
        Retorno da carteira:    R_p = w^T * mu      (média ponderada dos retornos)
        Variância da carteira:  s_p^2 = w^T * Sigma * w
    Note que o retorno é uma simples média ponderada, mas o risco NÃO:
    ele depende das covariâncias, e é aí que a diversificação atua.
    """
    w = np.asarray(pesos, dtype=float)
    retorno = float(w @ np.asarray(mu))
    volatilidade = float(np.sqrt(w @ np.asarray(cov) @ w))
    sharpe = (retorno - taxa_livre_risco) / volatilidade if volatilidade > 0 else np.nan
    return {"retorno": retorno, "volatilidade": volatilidade, "sharpe": sharpe}


def efeito_diversificacao(pesos: np.ndarray, volatilidades: pd.Series,
                          cov: pd.DataFrame) -> dict:
    """
    Quantifica o benefício da diversificação para uma carteira.

    Compara:
      - volatilidade "ingênua": média ponderada dos riscos individuais, que
        seria o risco da carteira se todas as correlações fossem +1;
      - volatilidade real: sqrt(w^T Sigma w), que considera as correlações.
    A diferença é o risco eliminado pela diversificação.
    """
    w = np.asarray(pesos, dtype=float)
    vol_ponderada = float(w @ volatilidades.values)
    vol_real = float(np.sqrt(w @ cov.values @ w))
    reducao = 1 - vol_real / vol_ponderada if vol_ponderada > 0 else 0.0
    return {"vol_ponderada": vol_ponderada, "vol_real": vol_real, "reducao": reducao}


def maximo_drawdown(serie_valores: pd.Series) -> float:
    """
    Maior queda percentual do pico ao vale (máximo drawdown).

    cummax() guarda o maior valor já atingido até cada data; a distância
    entre o valor atual e esse pico é o drawdown do dia. O pior deles é o
    máximo drawdown (um número negativo, ex.: -0.35 = queda de 35%).
    """
    pico = serie_valores.cummax()
    return float((serie_valores / pico - 1).min())


# ---------------------------------------------------------------------------
# Desempenho histórico (backtest)
# ---------------------------------------------------------------------------

def desempenho_historico(retornos: pd.DataFrame, carteiras: dict[str, np.ndarray],
                         retornos_mercado: pd.Series,
                         taxa_livre_risco: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simula como cada carteira teria se comportado no período analisado.

    Hipótese: os pesos são mantidos CONSTANTES (rebalanceamento diário), que
    é exatamente a hipótese do modelo de Markowitz. O retorno diário da
    carteira é então a média ponderada dos retornos dos ativos naquele dia:
        r_carteira,t = soma_i( w_i * r_i,t )   ->  em código: retornos @ pesos

    O valor acumulado parte de 100 e é obtido pelo produto acumulado
    (1 + r_1) * (1 + r_2) * ... -> cumprod(), ou seja, juros compostos.

    Retorna:
        acumulado : DataFrame com a evolução (base 100) de cada carteira e do Ibovespa
        metricas  : DataFrame com retorno total, retorno anualizado, volatilidade,
                    Sharpe e máximo drawdown de cada série
    """
    series_retorno = {nome: retornos @ pesos for nome, pesos in carteiras.items()}
    series_retorno["Ibovespa"] = retornos_mercado.loc[retornos.index]
    diarios = pd.DataFrame(series_retorno)

    acumulado = 100 * (1 + diarios).cumprod()
    # Acrescenta o ponto inicial (valor 100) na data anterior ao primeiro retorno,
    # para que todas as curvas comecem visualmente no mesmo ponto.
    acumulado.loc[retornos.index[0] - pd.Timedelta(days=1)] = 100.0
    acumulado = acumulado.sort_index()

    n_dias = len(diarios)
    linhas = {}
    for nome in diarios.columns:
        total = acumulado[nome].iloc[-1] / 100 - 1
        media_anual = diarios[nome].mean() * DIAS_UTEIS_ANO
        vol_anual = diarios[nome].std() * np.sqrt(DIAS_UTEIS_ANO)
        linhas[nome] = {
            "retorno_total": total,
            # Retorno anualizado composto (CAGR): taxa constante que, aplicada
            # todo ano, levaria ao mesmo retorno total no mesmo prazo.
            "retorno_anualizado": (1 + total) ** (DIAS_UTEIS_ANO / n_dias) - 1,
            "volatilidade": vol_anual,
            "sharpe": (media_anual - taxa_livre_risco) / vol_anual,
            "max_drawdown": maximo_drawdown(acumulado[nome]),
        }
    metricas = pd.DataFrame(linhas).T
    return acumulado, metricas
