"""
Módulo de COLETA DE DADOS.

Responsável por obter os dados REAIS usados na análise:

1. Cotações históricas diárias (fechamento AJUSTADO) das ações da B3 e do
   Ibovespa, via biblioteca yfinance (dados do Yahoo Finance).
   - "Ajustado" significa que o preço já foi corrigido por proventos
     (dividendos, JCP) e eventos societários (desdobramentos, grupamentos).
     Isso é essencial: sem o ajuste, um desdobramento de 1 para 2 pareceria
     uma queda de 50% no preço e distorceria completamente o cálculo de risco.

2. Taxa Selic META atual, pela API pública SGS do Banco Central
   (série 432). Ela é usada como TAXA LIVRE DE RISCO no índice de Sharpe,
   porque representa a remuneração de um título público pós-fixado
   (ex.: Tesouro Selic), considerado o investimento de menor risco no Brasil.

3. FALLBACK (plano B): se não houver internet ou alguma API falhar, os dados
   são carregados de arquivos CSV na pasta data/. Sempre que um download
   online dá certo, esses CSVs são atualizados automaticamente, de modo que
   a próxima execução offline use os dados mais recentes possíveis.
"""

from __future__ import annotations  # permite anotações de tipo modernas no Python 3.8+

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constantes de configuração
# ---------------------------------------------------------------------------

# Path(__file__) é o caminho deste arquivo; .parent.parent sobe até a raiz do
# projeto. Assim os caminhos funcionam independentemente da pasta de onde o
# usuário executa o programa.
PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_RAIZ / "data"
ARQ_PRECOS = PASTA_DADOS / "precos.csv"   # preços ajustados (ações + Ibovespa)
ARQ_SELIC = PASTA_DADOS / "selic.csv"     # última Selic meta obtida

# Código do Ibovespa no Yahoo Finance. O Ibovespa é o principal índice da B3
# e funciona como "carteira de mercado" (benchmark) para o cálculo do beta.
TICKER_MERCADO = "^BVSP"

# Endpoint da API SGS do Banco Central para a série 432 (meta Selic definida
# pelo Copom, em % ao ano). A série é diária e o BCB já publica a meta vigente
# até a data da próxima reunião do Copom; por isso consultamos uma janela de
# datas que termina HOJE e pegamos o último valor (a meta em vigor hoje).
URL_SELIC = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"

# Valor de último recurso para a Selic, usado SOMENTE se a API falhar e também
# não existir o CSV local. O usuário é avisado se isso acontecer.
SELIC_PADRAO_PERCENTUAL = 15.0

# Número mínimo de pregões para que as estatísticas façam sentido.
# Com poucas observações, médias e covariâncias ficam muito instáveis.
MIN_OBSERVACOES = 60


class ErroColeta(Exception):
    """Erro específico de coleta de dados (facilita tratar e acionar o fallback)."""


@dataclass
class DadosMercado:
    """
    Estrutura que agrupa tudo o que foi coletado.

    Usar uma dataclass deixa o código mais legível do que devolver uma tupla
    com vários itens: cada informação tem um nome explícito.
    """

    precos: pd.DataFrame          # preços ajustados das ações (linhas=datas, colunas=tickers)
    mercado: pd.Series            # preços do Ibovespa nas mesmas datas
    selic_anual: float            # Selic em forma decimal (ex.: 0.1375 = 13,75% a.a.)
    data_selic: str               # data de referência informada pelo Banco Central
    fonte_precos: str             # descrição da origem dos preços (online ou CSV)
    fonte_selic: str              # descrição da origem da Selic
    data_coleta: str              # data/hora em que a coleta foi feita
    usou_fallback: bool = False   # True se algum dado veio dos CSVs locais
    avisos: list[str] = field(default_factory=list)  # mensagens para o usuário


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _descrever_erro(exc: Exception) -> str:
    """
    Traduz exceções de rede para uma frase curta em português.

    As mensagens originais da biblioteca requests são longas e técnicas;
    para o usuário basta saber se foi falta de conexão, demora ou outro erro.
    """
    if isinstance(exc, requests.exceptions.Timeout):
        return "tempo de resposta esgotado"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "sem conexão com a internet ou servidor indisponível"
    texto = str(exc)
    return texto if len(texto) <= 150 else texto[:150] + "..."


def _avisar(avisos: list[str], mensagem: str) -> None:
    """Imprime um aviso no terminal e o guarda para aparecer no relatório."""
    print(f"  [AVISO] {mensagem}")
    avisos.append(mensagem)


def _remover_fuso(indice: pd.Index) -> pd.DatetimeIndex:
    """
    Converte o índice para datas "sem fuso horário".

    O Yahoo Finance pode devolver datas com fuso (America/Sao_Paulo). Datas
    com e sem fuso não podem ser comparadas entre si no pandas, então
    padronizamos tudo sem fuso para evitar erros ao filtrar e juntar dados.
    """
    indice = pd.to_datetime(indice)
    if getattr(indice, "tz", None) is not None:
        indice = indice.tz_localize(None)
    return indice.normalize()  # zera hora/minuto, mantendo apenas a data


# ---------------------------------------------------------------------------
# Cotações: download online
# ---------------------------------------------------------------------------

def _baixar_precos_online(tickers: list[str], inicio: date, fim: date) -> pd.DataFrame:
    """
    Baixa os preços de fechamento ajustado das ações e do Ibovespa.

    Retorna um DataFrame com uma coluna por ticker (incluindo ^BVSP).
    Lança ErroColeta se o download falhar ou vier incompleto.
    """
    # Import feito aqui dentro (e não no topo do arquivo) de propósito:
    # se o yfinance não estiver instalado, o programa ainda consegue rodar
    # no modo offline usando os CSVs.
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ErroColeta("biblioteca yfinance não está instalada") from exc

    # O yfinance escreve muitas mensagens de erro técnicas quando não há
    # internet. Silenciamos essas mensagens e mostramos as nossas, em português.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    todos = list(tickers) + [TICKER_MERCADO]

    # auto_adjust=True faz a coluna "Close" já vir com o preço AJUSTADO.
    # O parâmetro "end" do Yahoo é exclusivo, por isso somamos 1 dia.
    try:
        bruto = yf.download(
            todos,
            start=inicio.isoformat(),
            end=(fim + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=20,
        )
    except Exception as exc:  # qualquer falha de rede/API cai aqui
        raise ErroColeta(f"falha no download do Yahoo Finance: {exc}") from exc

    if bruto is None or bruto.empty:
        raise ErroColeta("o Yahoo Finance não retornou nenhum dado (sem internet?)")

    # Com vários tickers, o yfinance devolve colunas em dois níveis
    # (tipo de preço, ticker). Pegamos apenas o nível "Close".
    if isinstance(bruto.columns, pd.MultiIndex):
        fechamento = bruto["Close"].copy()
    else:
        fechamento = bruto[["Close"]].rename(columns={"Close": todos[0]})

    fechamento.index = _remover_fuso(fechamento.index)
    # Remove colunas totalmente vazias (tickers que falharam no download).
    fechamento = fechamento.dropna(axis=1, how="all")

    if TICKER_MERCADO not in fechamento.columns:
        raise ErroColeta("não foi possível baixar o Ibovespa (^BVSP)")

    # Descarta pregões ainda NÃO CONSOLIDADOS: durante o dia (ou logo após o
    # fechamento), o Yahoo publica uma cotação provisória do Ibovespa com
    # volume zero. Usar esse preço provisório distorceria o último retorno.
    if isinstance(bruto.columns, pd.MultiIndex) and "Volume" in bruto.columns.get_level_values(0):
        volume_ibov = bruto["Volume"][TICKER_MERCADO]
        volume_ibov.index = _remover_fuso(volume_ibov.index)
        provisorios = volume_ibov.index[volume_ibov.fillna(0) <= 0]
        fechamento = fechamento.drop(index=provisorios, errors="ignore")

    return fechamento


# ---------------------------------------------------------------------------
# Cotações: arquivos CSV locais (fallback)
# ---------------------------------------------------------------------------

def _salvar_precos_csv(novos: pd.DataFrame) -> None:
    """
    Salva os preços baixados em data/precos.csv.

    Se já existir um CSV, os dados novos são COMBINADOS com os antigos
    (combine_first): valores novos têm prioridade e tickers/datas antigos
    que não estão no download atual são preservados. Assim, rodar o programa
    com outra lista de ações não apaga os dados de exemplo já salvos.
    """
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    combinado = novos
    if ARQ_PRECOS.exists():
        try:
            antigos = pd.read_csv(ARQ_PRECOS, index_col=0, parse_dates=True)
            antigos.index = _remover_fuso(antigos.index)
            combinado = novos.combine_first(antigos)
        except Exception:
            combinado = novos  # CSV antigo corrompido: substitui pelo novo
    combinado = combinado.sort_index()
    combinado.to_csv(ARQ_PRECOS, index_label="Data", float_format="%.6f")


def _carregar_precos_csv() -> pd.DataFrame:
    """Lê data/precos.csv. Lança ErroColeta se o arquivo não existir."""
    if not ARQ_PRECOS.exists():
        raise ErroColeta(
            f"arquivo {ARQ_PRECOS} não encontrado. Execute o programa uma vez "
            "com internet para gerá-lo."
        )
    dados = pd.read_csv(ARQ_PRECOS, index_col=0, parse_dates=True)
    dados.index = _remover_fuso(dados.index)
    return dados.sort_index()


# ---------------------------------------------------------------------------
# Selic (taxa livre de risco)
# ---------------------------------------------------------------------------

def _baixar_selic_online() -> tuple[float, str]:
    """
    Consulta a API do Banco Central e retorna (selic_percentual, data).

    Exemplo de resposta da API: [{"data": "05/10/2026", "valor": "13.75"}, ...]
    O valor vem em % ao ano, como texto; convertemos para float.
    """
    hoje = date.today()
    parametros = {
        "formato": "json",
        "dataInicial": (hoje - timedelta(days=30)).strftime("%d/%m/%Y"),  # formato exigido pela API
        "dataFinal": hoje.strftime("%d/%m/%Y"),
    }
    resposta = requests.get(URL_SELIC, params=parametros, timeout=15)
    resposta.raise_for_status()  # lança exceção se o status HTTP for de erro
    conteudo = resposta.json()
    ultimo = conteudo[-1]
    return float(ultimo["valor"]), str(ultimo["data"])


def _salvar_selic_csv(valor_percentual: float, data_ref: str) -> None:
    """Guarda a Selic obtida online em data/selic.csv (para o modo offline)."""
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "data_referencia": data_ref,
            "selic_meta_percentual_aa": valor_percentual,
            "data_coleta": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }]
    ).to_csv(ARQ_SELIC, index=False)


def _carregar_selic_csv() -> tuple[float, str]:
    """Lê a Selic salva em data/selic.csv."""
    if not ARQ_SELIC.exists():
        raise ErroColeta(f"arquivo {ARQ_SELIC} não encontrado")
    tabela = pd.read_csv(ARQ_SELIC)
    linha = tabela.iloc[-1]
    return float(linha["selic_meta_percentual_aa"]), str(linha["data_referencia"])


def obter_selic(offline: bool, avisos: list[str]) -> tuple[float, str, str, bool]:
    """
    Obtém a Selic meta seguindo a ordem: API do BCB -> CSV local -> valor padrão.

    Retorna (selic_decimal, data_referencia, descricao_da_fonte, usou_fallback).
    """
    if not offline:
        try:
            valor, data_ref = _baixar_selic_online()
            _salvar_selic_csv(valor, data_ref)
            return valor / 100, data_ref, "API SGS do Banco Central (série 432 - meta Selic)", False
        except Exception as exc:
            _avisar(avisos, f"Não foi possível consultar a Selic no Banco Central ({_descrever_erro(exc)}). "
                            "Usando o valor salvo em data/selic.csv.")

    try:
        valor, data_ref = _carregar_selic_csv()
        return valor / 100, data_ref, f"arquivo local {ARQ_SELIC.name} (modo offline)", True
    except Exception as exc:
        _avisar(avisos, f"Selic local indisponível ({exc}). Usando valor padrão de "
                        f"{SELIC_PADRAO_PERCENTUAL:.2f}% a.a.")
        return SELIC_PADRAO_PERCENTUAL / 100, "valor padrão", "valor padrão do programa", True


# ---------------------------------------------------------------------------
# Função principal do módulo
# ---------------------------------------------------------------------------

def coletar_dados(tickers: list[str], inicio: date, fim: date,
                  offline: bool = False) -> DadosMercado:
    """
    Coleta preços e Selic, com fallback automático para os CSVs locais.

    Parâmetros
    ----------
    tickers : lista de códigos das ações (ex.: ["PETR4.SA", "VALE3.SA"])
    inicio, fim : período da análise
    offline : se True, nem tenta acessar a internet (útil para testes)
    """
    avisos: list[str] = []
    usou_fallback = False
    tabela: pd.DataFrame | None = None

    # ---------- 1) Preços ----------
    if not offline:
        print("  Baixando cotações do Yahoo Finance...")
        try:
            tabela = _baixar_precos_online(tickers, inicio, fim)
            _salvar_precos_csv(tabela)  # atualiza o "plano B" para próximas execuções
            fonte_precos = "Yahoo Finance (online, via yfinance)"
        except Exception as exc:
            _avisar(avisos, f"Falha ao baixar cotações online ({_descrever_erro(exc)}). "
                            "Carregando dados salvos em data/precos.csv.")
            tabela = None

    if tabela is None:
        # Modo offline (forçado ou por falha): lê o CSV e filtra o período pedido.
        usou_fallback = True
        completo = _carregar_precos_csv()
        if TICKER_MERCADO not in completo.columns:
            raise ErroColeta("o CSV local não contém o Ibovespa (^BVSP)")
        tabela = completo.loc[pd.Timestamp(inicio):pd.Timestamp(fim)]
        if len(tabela.dropna(how="all")) < MIN_OBSERVACOES:
            # O período pedido não está no CSV: usa tudo o que houver, avisando.
            _avisar(avisos, "O período solicitado não está disponível no CSV local; "
                            "usando todo o histórico salvo.")
            tabela = completo
        fonte_precos = f"arquivo local {ARQ_PRECOS.name} (modo offline)"

    # ---------- 2) Seleção e limpeza dos tickers ----------
    disponiveis = [t for t in tickers if t in tabela.columns and tabela[t].notna().any()]
    faltando = [t for t in tickers if t not in disponiveis]
    if faltando:
        _avisar(avisos, "Sem dados para: " + ", ".join(faltando) +
                ". Esses ativos foram retirados da análise.")
    if len(disponiveis) < 2:
        raise ErroColeta("são necessários pelo menos 2 ativos com dados para montar carteiras")

    # Mantém só as datas em que TODOS os ativos e o Ibovespa têm preço.
    # Isso garante que retornos e covariâncias sejam calculados sobre os
    # mesmos dias (comparação justa entre os ativos).
    alinhado = tabela[disponiveis + [TICKER_MERCADO]].dropna()
    if len(alinhado) < MIN_OBSERVACOES:
        raise ErroColeta(
            f"apenas {len(alinhado)} pregões em comum entre os ativos; "
            f"o mínimo é {MIN_OBSERVACOES}"
        )

    # ---------- 3) Selic ----------
    print("  Consultando a taxa Selic (Banco Central)...")
    selic, data_selic, fonte_selic, selic_fallback = obter_selic(offline, avisos)
    usou_fallback = usou_fallback or selic_fallback

    if offline:
        _avisar(avisos, "Execução em modo offline: dados carregados dos arquivos CSV da pasta data/.")

    return DadosMercado(
        precos=alinhado[disponiveis],
        mercado=alinhado[TICKER_MERCADO],
        selic_anual=selic,
        data_selic=data_selic,
        fonte_precos=fonte_precos,
        fonte_selic=fonte_selic,
        data_coleta=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        usou_fallback=usou_fallback,
        avisos=avisos,
    )
