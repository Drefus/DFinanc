"""
TRABALHO PRÁTICO - ADMINISTRAÇÃO FINANCEIRA (Sistemas de Informação)
Tema: Risco e Retorno - Teoria Moderna de Carteiras de Markowitz

Este é o programa principal. Ele ORQUESTRA as etapas da análise, chamando
os módulos do pacote "markowitz" na ordem correta:

    1. Leitura dos parâmetros da linha de comando (argparse)
    2. Coleta de dados reais (Yahoo Finance + Banco Central), com fallback CSV
    3. Cálculo de retornos, riscos, covariância, correlação e beta
    4. Simulação de Monte Carlo e otimização das carteiras
    5. Teste do desempenho histórico contra o Ibovespa
    6. Geração dos gráficos, do relatório HTML, do CSV e do resumo no terminal

Como executar (exemplos):
    python main.py
    python main.py --tickers PETR4 VALE3 ITUB4 --anos 5
    python main.py --inicio 2021-01-01 --fim 2024-12-31 --simulacoes 20000
    python main.py --offline          (usa apenas os CSVs da pasta data/)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from markowitz import graficos, otimizacao, relatorio
from markowitz import risco_retorno as rr
from markowitz.coleta_dados import ErroColeta, coletar_dados

# Valores padrão pedidos no enunciado do trabalho.
TICKERS_PADRAO = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "ABEV3.SA", "BBAS3.SA"]
ANOS_PADRAO = 3
SIMULACOES_PADRAO = 10_000
SEMENTE_PADRAO = 42
PONTOS_FRONTEIRA_PADRAO = 50
PASTA_SAIDA = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------

def _data(texto: str) -> date:
    """Converte 'AAAA-MM-DD' em date; usado pelo argparse para validar a entrada."""
    try:
        return datetime.strptime(texto, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"data inválida '{texto}' (use AAAA-MM-DD)") from exc


def ler_argumentos(argv=None) -> argparse.Namespace:
    """
    Define e lê os argumentos de linha de comando.

    Todos têm valor padrão, então "python main.py" funciona sem nenhum argumento.
    """
    parser = argparse.ArgumentParser(
        description="Otimização de carteiras de ações da B3 pela Teoria de Markowitz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=TICKERS_PADRAO,
                        help="códigos das ações (o sufixo .SA é adicionado se faltar)")
    parser.add_argument("--anos", type=float, default=ANOS_PADRAO,
                        help="tamanho do período em anos, contado até a data final")
    parser.add_argument("--inicio", type=_data, default=None,
                        help="data inicial AAAA-MM-DD (tem prioridade sobre --anos)")
    parser.add_argument("--fim", type=_data, default=None,
                        help="data final AAAA-MM-DD (padrão: hoje)")
    parser.add_argument("--simulacoes", type=int, default=SIMULACOES_PADRAO,
                        help="número de carteiras aleatórias no Monte Carlo")
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO,
                        help="semente aleatória (garante resultados reproduzíveis)")
    parser.add_argument("--pontos-fronteira", type=int, default=PONTOS_FRONTEIRA_PADRAO,
                        help="quantidade de pontos usados para traçar a fronteira eficiente")
    parser.add_argument("--offline", action="store_true",
                        help="não acessa a internet; usa os CSVs salvos em data/")
    args = parser.parse_args(argv)

    # Padroniza os tickers: maiúsculas e sufixo .SA (ações da B3 no Yahoo).
    # Índices (começam com ^) e códigos que já têm sufixo são mantidos.
    tickers = []
    for t in args.tickers:
        t = t.strip().upper()
        if not t.startswith("^") and "." not in t:
            t += ".SA"
        if t not in tickers:  # evita duplicados
            tickers.append(t)
    args.tickers = tickers

    if len(args.tickers) < 2:
        parser.error("informe pelo menos 2 ações para formar uma carteira")
    if args.simulacoes < 100:
        parser.error("use pelo menos 100 simulações (o padrão é 10.000)")

    # Calcula o período: fim = hoje (se não informado); início = fim - N anos.
    args.fim = args.fim or date.today()
    if args.inicio is None:
        args.inicio = (pd.Timestamp(args.fim) - pd.DateOffset(days=round(args.anos * 365.25))).date()
    if args.inicio >= args.fim:
        parser.error("a data inicial deve ser anterior à data final")
    return args


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # Ajuste de codificação para os acentos aparecerem corretamente:
    # - se a saída for redirecionada (arquivo, pipe, Git Bash), usa UTF-8;
    # - em qualquer caso, caracteres não suportados viram "?" em vez de erro.
    for fluxo in (sys.stdout, sys.stderr):
        try:
            if fluxo.isatty():
                fluxo.reconfigure(errors="replace")
            else:
                fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = ler_argumentos(argv)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)  # cria output/ se não existir

    print("\n>> Teoria Moderna de Carteiras - Markowitz (B3)")
    print(f"   Ativos: {', '.join(args.tickers)}")
    print(f"   Período solicitado: {args.inicio:%d/%m/%Y} a {args.fim:%d/%m/%Y}\n")

    # ------------------------------------------------------------------
    # ETAPA 1: coleta de dados reais
    # ------------------------------------------------------------------
    print("[1/5] Coletando dados...")
    try:
        dados = coletar_dados(args.tickers, args.inicio, args.fim, offline=args.offline)
    except ErroColeta as exc:
        print(f"\n  [ERRO] Não foi possível obter os dados: {exc}")
        return 1
    print(f"  OK: {dados.precos.shape[1]} ativos, {len(dados.precos)} pregões "
          f"({dados.precos.index[0]:%d/%m/%Y} a {dados.precos.index[-1]:%d/%m/%Y}).")
    print(f"  Selic meta: {relatorio.pct(dados.selic_anual)} a.a.")

    # ------------------------------------------------------------------
    # ETAPA 2: retorno e risco de cada ativo
    # ------------------------------------------------------------------
    print("[2/5] Calculando retornos, riscos, covariância, correlação e beta...")
    retornos = rr.calcular_retornos(dados.precos)
    retornos_mercado = rr.calcular_retornos(dados.mercado)
    mu = rr.retorno_esperado_anual(retornos)              # vetor de retornos esperados
    cov = rr.matriz_covariancia_anual(retornos)           # matriz de covariância (Sigma)
    corr = rr.matriz_correlacao(retornos)
    rf = dados.selic_anual                                 # taxa livre de risco
    estatisticas = rr.estatisticas_por_ativo(retornos, retornos_mercado, dados.precos, rf)

    # ------------------------------------------------------------------
    # ETAPA 3: Monte Carlo e otimização
    # ------------------------------------------------------------------
    print(f"[3/5] Simulando {args.simulacoes} carteiras e otimizando...")
    simulacao = otimizacao.simular_carteiras(mu, cov, rf, args.simulacoes, args.semente)
    w_min = otimizacao.carteira_minima_variancia(mu, cov)
    w_sharpe = otimizacao.carteira_maximo_sharpe(mu, cov, rf)
    w_iguais = np.full(len(mu), 1 / len(mu))               # benchmark ingênuo: 1/N
    fronteira = otimizacao.fronteira_eficiente(mu, cov, rf, args.pontos_fronteira)

    # Dicionário com os pesos de cada carteira (a ordem define a ordem nas tabelas).
    carteiras = {"Mínima Variância": w_min, "Máximo Sharpe": w_sharpe, "Pesos Iguais": w_iguais}
    pesos = pd.DataFrame(carteiras, index=mu.index)
    metricas = pd.DataFrame({nome: rr.metricas_carteira(w, mu, cov, rf)
                             for nome, w in carteiras.items()}).T
    diversificacao = {nome: rr.efeito_diversificacao(w, estatisticas["volatilidade"], cov)
                      for nome, w in carteiras.items()}

    # ------------------------------------------------------------------
    # ETAPA 4: desempenho histórico x Ibovespa
    # ------------------------------------------------------------------
    print("[4/5] Comparando o desempenho histórico com o Ibovespa...")
    acumulado, desempenho = rr.desempenho_historico(retornos, carteiras, retornos_mercado, rf)

    # ------------------------------------------------------------------
    # ETAPA 5: gráficos e relatórios
    # ------------------------------------------------------------------
    print("[5/5] Gerando gráficos e relatórios...")
    caminhos_graficos = {
        "precos": graficos.grafico_precos_normalizados(dados.precos, dados.mercado, PASTA_SAIDA),
        "correlacao": graficos.grafico_correlacao(corr, PASTA_SAIDA),
        "fronteira": graficos.grafico_fronteira(
            simulacao, fronteira, estatisticas,
            {nome: metricas.loc[nome].to_dict() for nome in carteiras}, rf, PASTA_SAIDA),
        "pesos": graficos.grafico_pesos(pesos[["Mínima Variância", "Máximo Sharpe"]], PASTA_SAIDA),
        "desempenho": graficos.grafico_desempenho(acumulado, PASTA_SAIDA),
    }

    resultados = relatorio.ResultadosAnalise(
        parametros={
            "simulacoes": args.simulacoes,
            "semente": args.semente,
            "pontos_fronteira": args.pontos_fronteira,
            "inicio_efetivo": f"{dados.precos.index[0]:%d/%m/%Y}",
            "fim_efetivo": f"{dados.precos.index[-1]:%d/%m/%Y}",
            "n_pregoes": len(dados.precos),
        },
        dados=dados,
        estatisticas=estatisticas,
        correlacao=corr,
        covariancia=cov,
        pesos=pesos,
        metricas_carteiras=metricas,
        desempenho=desempenho,
        diversificacao=diversificacao,
        simulacao=simulacao,
        graficos=caminhos_graficos,
    )

    arq_html = relatorio.gerar_html(resultados, PASTA_SAIDA / "relatorio_markowitz.html")
    arq_csv = relatorio.gerar_csv_carteiras(resultados, PASTA_SAIDA / "carteiras_otimas.csv")

    # Arquivos complementares para quem quiser conferir os números no Excel.
    estatisticas.to_csv(PASTA_SAIDA / "estatisticas_ativos.csv", sep=";", decimal=",",
                        float_format="%.6f", encoding="utf-8-sig", index_label="ativo")
    corr.to_csv(PASTA_SAIDA / "matriz_correlacao.csv", sep=";", decimal=",",
                float_format="%.6f", encoding="utf-8-sig")
    cov.to_csv(PASTA_SAIDA / "matriz_covariancia_anual.csv", sep=";", decimal=",",
               float_format="%.8f", encoding="utf-8-sig")

    relatorio.imprimir_resumo(resultados, [arq_html, arq_csv, *caminhos_graficos.values()])
    print(f"Abra o relatório no navegador: {arq_html}")
    return 0


if __name__ == "__main__":
    # sys.exit devolve o código de saída (0 = sucesso, 1 = erro) ao sistema operacional.
    sys.exit(main())
