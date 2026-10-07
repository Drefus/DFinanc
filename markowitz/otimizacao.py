"""
Módulo de OTIMIZAÇÃO DE CARTEIRAS (Teoria Moderna de Carteiras).

Ideia central de Markowitz: o investidor racional quer o MAIOR retorno
possível para um dado nível de risco (ou o MENOR risco para um dado retorno).
As carteiras que satisfazem essa condição formam a FRONTEIRA EFICIENTE.
Qualquer carteira abaixo dela é "ineficiente": existe outra com o mesmo
risco e mais retorno.

Este módulo implementa três abordagens complementares:

1. SIMULAÇÃO DE MONTE CARLO
   Sorteia milhares de carteiras com pesos aleatórios e calcula retorno,
   risco e Sharpe de cada uma. O resultado é uma "nuvem" de pontos que
   mostra visualmente o conjunto de carteiras possíveis.

2. OTIMIZAÇÃO NUMÉRICA (scipy.optimize.minimize, método SLSQP)
   Encontra matematicamente as carteiras ótimas:
   - MÍNIMA VARIÂNCIA: a carteira de menor risco possível.
   - MÁXIMO SHARPE (carteira tangente): a de melhor relação retorno/risco
     considerando a Selic como taxa livre de risco.

3. FRONTEIRA EFICIENTE
   Para vários retornos-alvo, encontra a carteira de menor risco que
   atinge aquele retorno. Ligando esses pontos obtemos a fronteira.

Restrições usadas em todas as abordagens:
   - soma dos pesos = 1 (100% do capital investido);
   - cada peso entre 0 e 1 (sem venda a descoberto, ou seja, não é permitido
     "vender ações que não se tem" para apostar na queda).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risco_retorno import metricas_carteira


# ---------------------------------------------------------------------------
# 1) Simulação de Monte Carlo
# ---------------------------------------------------------------------------

def simular_carteiras(mu: pd.Series, cov: pd.DataFrame, taxa_livre_risco: float,
                      n_carteiras: int = 10_000, semente: int = 42) -> pd.DataFrame:
    """
    Gera n_carteiras aleatórias e calcula suas métricas.

    Lógica de programação:
    - np.random.default_rng(semente) cria um gerador de números aleatórios
      com SEMENTE FIXA: toda execução sorteia exatamente as mesmas carteiras,
      o que torna os resultados reproduzíveis.
    - rng.dirichlet(1, 1, ..., 1) sorteia vetores de pesos não negativos que
      somam exatamente 1, distribuídos de forma uniforme entre todas as
      combinações possíveis. (Dividir números aleatórios pela soma também
      funcionaria, mas concentraria os sorteios perto dos pesos iguais.)
    - Os cálculos são VETORIZADOS (todas as carteiras de uma vez, com
      álgebra matricial), o que é muito mais rápido que um laço for.

    Retorna um DataFrame com colunas retorno, volatilidade, sharpe e
    uma coluna de peso para cada ativo.
    """
    rng = np.random.default_rng(semente)
    n_ativos = len(mu)

    # Matriz de pesos: n_carteiras linhas x n_ativos colunas.
    pesos = rng.dirichlet(np.ones(n_ativos), size=n_carteiras)

    # Retorno de cada carteira: w^T mu  ->  para todas de uma vez: W @ mu
    retornos = pesos @ mu.values

    # Variância de cada carteira: w^T Sigma w. A função einsum calcula essa
    # forma quadrática para cada linha da matriz W sem precisar de laço.
    variancias = np.einsum("ij,jk,ik->i", pesos, cov.values, pesos)
    volatilidades = np.sqrt(variancias)

    sharpe = (retornos - taxa_livre_risco) / volatilidades

    resultado = pd.DataFrame({
        "retorno": retornos,
        "volatilidade": volatilidades,
        "sharpe": sharpe,
    })
    for i, ticker in enumerate(mu.index):
        resultado[f"peso_{ticker}"] = pesos[:, i]
    return resultado


# ---------------------------------------------------------------------------
# 2) Otimização numérica
# ---------------------------------------------------------------------------

def _restricoes_basicas(n_ativos: int):
    """
    Define as restrições comuns a todas as otimizações.

    bounds: cada peso w_i deve ficar entre 0 e 1 (sem venda a descoberto).
    constraints: igualdade soma(w) - 1 = 0 (todo o capital investido).
    """
    limites = tuple((0.0, 1.0) for _ in range(n_ativos))
    soma_um = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    return limites, [soma_um]


def _otimizar(funcao_objetivo, n_ativos: int, restricoes_extras=None, gradiente=None) -> np.ndarray:
    """
    Executa o otimizador SLSQP (Sequential Least Squares Programming).

    SLSQP é adequado para problemas com função objetivo suave e restrições
    de igualdade e de limites, como o problema de Markowitz.
    O ponto de partida é a carteira de pesos iguais (1/N para cada ativo).
    """
    limites, restricoes = _restricoes_basicas(n_ativos)
    if restricoes_extras:
        restricoes = restricoes + restricoes_extras

    chute_inicial = np.full(n_ativos, 1.0 / n_ativos)
    resultado = minimize(
        funcao_objetivo,
        chute_inicial,
        method="SLSQP",
        jac=gradiente,
        bounds=limites,
        constraints=restricoes,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not resultado.success:
        warnings.warn(f"Otimização não convergiu: {resultado.message}")

    # Pequenos valores negativos (ex.: -1e-17) surgem por erro numérico de
    # ponto flutuante; zeramos e renormalizamos para a soma ser exatamente 1.
    pesos = np.clip(resultado.x, 0.0, None)
    return pesos / pesos.sum()


def carteira_minima_variancia(mu: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    """
    Encontra os pesos que MINIMIZAM a variância da carteira (w^T Sigma w).

    É a carteira mais à esquerda da fronteira eficiente: a de menor risco
    possível combinando os ativos escolhidos. Repare que ela NÃO usa os
    retornos esperados, apenas a matriz de covariância.
    """
    sigma = cov.values

    def variancia(w):
        return w @ sigma @ w

    def grad_variancia(w):
        # Derivada de w^T Sigma w em relação a w é 2 * Sigma * w.
        # Informar o gradiente deixa o otimizador mais rápido e preciso.
        return 2 * sigma @ w

    return _otimizar(variancia, len(mu), gradiente=grad_variancia)


def carteira_maximo_sharpe(mu: pd.Series, cov: pd.DataFrame,
                           taxa_livre_risco: float) -> np.ndarray:
    """
    Encontra os pesos que MAXIMIZAM o índice de Sharpe.

    Os otimizadores do scipy apenas MINIMIZAM funções; por isso minimizamos
    o NEGATIVO do Sharpe (maximizar f é o mesmo que minimizar -f).

    Essa carteira é chamada de CARTEIRA TANGENTE: é o ponto onde a reta que
    sai da taxa livre de risco (Linha de Mercado de Capitais) toca a
    fronteira eficiente. Combinando-a com o ativo livre de risco (Selic)
    obtém-se o melhor retorno possível para cada nível de risco.
    """
    m = mu.values
    sigma = cov.values

    def sharpe_negativo(w):
        retorno = w @ m
        volatilidade = np.sqrt(w @ sigma @ w)
        return -(retorno - taxa_livre_risco) / volatilidade

    return _otimizar(sharpe_negativo, len(mu))


# ---------------------------------------------------------------------------
# 3) Fronteira eficiente
# ---------------------------------------------------------------------------

def fronteira_eficiente(mu: pd.Series, cov: pd.DataFrame, taxa_livre_risco: float,
                        n_pontos: int = 50) -> pd.DataFrame:
    """
    Traça a fronteira eficiente por otimização.

    Para cada retorno-alvo entre o retorno da carteira de mínima variância e
    o maior retorno individual entre os ativos, resolve:

        minimizar   w^T Sigma w            (risco)
        sujeito a   w^T mu = retorno_alvo  (atingir o retorno desejado)
                    soma(w) = 1, 0 <= w_i <= 1

    Retornos-alvo abaixo da mínima variância são ignorados porque formariam
    a parte INEFICIENTE da curva (mais risco para menos retorno).
    O maior retorno possível sem venda a descoberto é o do melhor ativo
    (100% nele), por isso ele é o limite superior.
    """
    sigma = cov.values
    m = mu.values

    w_min = carteira_minima_variancia(mu, cov)
    retorno_min = float(w_min @ m)
    retorno_max = float(m.max())

    pontos = []
    for alvo in np.linspace(retorno_min, retorno_max, n_pontos):
        # Restrição extra: o retorno da carteira deve ser igual ao alvo.
        # "alvo=alvo" fixa o valor atual da variável dentro da função lambda
        # (sem isso, todas as lambdas usariam o último valor do laço).
        restricao_retorno = {"type": "eq", "fun": lambda w, alvo=alvo: w @ m - alvo}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = _otimizar(lambda w: w @ sigma @ w, len(m),
                          restricoes_extras=[restricao_retorno],
                          gradiente=lambda w: 2 * sigma @ w)
        metricas = metricas_carteira(w, m, sigma, taxa_livre_risco)
        # Só aceita o ponto se o otimizador realmente atingiu o retorno-alvo.
        if abs(metricas["retorno"] - alvo) < 1e-4:
            pontos.append(metricas)

    return pd.DataFrame(pontos)
