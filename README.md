# Risco e Retorno: Teoria Moderna de Carteiras com ações da B3

Trabalho prático da disciplina de **Administração Financeira** (curso de Sistemas de Informação).

**Alunos:** 
* João Paulo Cardoso dos Santos - 2023001778
* Henrique Castro Caetano - 2023104941

## Objetivo

Aplicar na prática a **Teoria Moderna de Carteiras de Harry Markowitz (1952)** usando dados
reais do mercado brasileiro. O programa:

1. baixa as cotações históricas de ações da B3 e do Ibovespa (Yahoo Finance);
2. consulta a taxa **Selic** atual na API do Banco Central (taxa livre de risco);
3. calcula retorno, risco, covariância, correlação e beta de cada ação;
4. simula **10.000 carteiras aleatórias** (Monte Carlo);
5. encontra por otimização a carteira de **mínima variância** e a de **máximo índice de Sharpe**;
6. traça a **fronteira eficiente**;
7. compara o desempenho histórico das carteiras com uma carteira de pesos iguais e com o Ibovespa;
8. gera um **relatório HTML**, gráficos PNG, arquivos CSV e um resumo no terminal.

## Teoria resumida

| Conceito | O que é | Como é calculado |
|---|---|---|
| **Retorno** | Variação percentual do preço | `r = P_hoje / P_ontem - 1` (preço ajustado por proventos) |
| **Retorno esperado** | Estimativa do retorno futuro | Média dos retornos diários × 252 |
| **Risco (volatilidade)** | O quanto o retorno oscila | Desvio-padrão diário × √252 |
| **Covariância** | Se dois ativos se movem juntos | `Cov(rA, rB)`; forma a matriz Σ |
| **Correlação** | Covariância padronizada (-1 a +1) | `Cov(rA, rB) / (σA × σB)` |
| **Risco da carteira** | Depende das covariâncias, não só dos riscos individuais | `σp² = wᵀ Σ w` |
| **Beta** | Sensibilidade ao mercado (risco sistemático) | `Cov(r_ação, r_Ibov) / Var(r_Ibov)` |
| **Índice de Sharpe** | Retorno extra por unidade de risco | `(Retorno − Selic) / Volatilidade` |
| **Fronteira eficiente** | Carteiras com o maior retorno para cada nível de risco | Otimização para vários retornos-alvo |

**Diversificação:** como as ações não se movem de forma perfeitamente igual (correlação menor que 1),
o risco de uma carteira é menor que a média ponderada dos riscos das ações que a compõem. A
diversificação elimina o risco *específico* de cada empresa, mas não o risco *sistemático* (de mercado),
que é medido pelo beta.

**Carteira de mínima variância:** a combinação de menor risco possível.
**Carteira de máximo Sharpe (tangente):** a de melhor relação retorno/risco; é o ponto em que a reta que
sai da Selic (Linha de Mercado de Capitais) toca a fronteira eficiente.

Restrições usadas: os pesos somam 100% e cada peso fica entre 0% e 100% (sem venda a descoberto).

## Instalação

Requisito: **Python 3.9 ou superior**.

```bash
# (opcional) criar um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# instalar as bibliotecas
pip install -r requirements.txt
```

Bibliotecas usadas: `pandas`, `numpy`, `scipy`, `matplotlib`, `yfinance` e `requests`.

## Como executar

Com os valores padrão (PETR4, VALE3, ITUB4, WEGE3, ABEV3 e BBAS3, últimos 3 anos):

```bash
python main.py
```

Outros exemplos:

```bash
# escolher ações (o sufixo .SA é adicionado automaticamente) e período de 5 anos
python main.py --tickers PETR4 VALE3 ITUB4 BBDC4 --anos 5

# período com datas exatas e mais simulações
python main.py --inicio 2021-01-01 --fim 2024-12-31 --simulacoes 20000

# rodar sem internet, usando os CSVs da pasta data/
python main.py --offline

# ver todas as opções
python main.py --help
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--tickers` | PETR4.SA VALE3.SA ITUB4.SA WEGE3.SA ABEV3.SA BBAS3.SA | Ações a analisar (mínimo 2) |
| `--anos` | 3 | Tamanho do período, contado até a data final |
| `--inicio` / `--fim` | hoje − 3 anos / hoje | Datas no formato AAAA-MM-DD (`--inicio` tem prioridade sobre `--anos`) |
| `--simulacoes` | 10000 | Número de carteiras aleatórias do Monte Carlo |
| `--semente` | 42 | Semente aleatória (mesma semente = mesmos resultados) |
| `--pontos-fronteira` | 50 | Pontos usados para traçar a fronteira eficiente |
| `--offline` | desligado | Não acessa a internet; usa somente os CSVs |

### Funcionamento sem internet (fallback)

- Sempre que o download online dá certo, os dados são salvos em `data/precos.csv` e `data/selic.csv`.
- Se a internet ou alguma API falhar, o programa **carrega automaticamente esses CSVs** e avisa o
  usuário no terminal e no relatório.
- O repositório já vem com CSVs de exemplo, então o programa funciona mesmo na primeira execução sem internet.
- Se o período pedido não existir no CSV, é usado todo o histórico salvo (com aviso).

## Estrutura do projeto

```
DFinanc/
├── main.py                 # programa principal: lê argumentos e orquestra as etapas
├── requirements.txt
├── README.md
├── markowitz/
│   ├── coleta_dados.py     # Yahoo Finance, API do Banco Central e fallback em CSV
│   ├── risco_retorno.py    # retornos, volatilidade, covariância, correlação, beta, backtest
│   ├── otimizacao.py       # Monte Carlo, mínima variância, máximo Sharpe, fronteira eficiente
│   ├── graficos.py         # gráficos PNG (matplotlib)
│   └── relatorio.py        # relatório HTML, CSV de resultados e resumo no terminal
├── data/                   # CSVs com os dados coletados (usados no modo offline)
└── output/                 # criada automaticamente com os resultados
```

## Arquivos gerados (pasta `output/`)

| Arquivo | Conteúdo |
|---|---|
| `relatorio_markowitz.html` | Relatório completo, com gráficos embutidos (abra no navegador) |
| `carteiras_otimas.csv` | Pesos e métricas das carteiras (separador `;`, decimal `,` para abrir no Excel) |
| `estatisticas_ativos.csv` | Retorno, volatilidade, Sharpe, beta e drawdown de cada ação |
| `matriz_correlacao.csv`, `matriz_covariancia_anual.csv` | Matrizes usadas nos cálculos |
| `01_precos_normalizados.png` | Evolução dos preços em base 100 |
| `02_correlacao.png` | Mapa de calor da correlação |
| `03_fronteira_eficiente.png` | Nuvem de carteiras simuladas, fronteira eficiente e carteiras ótimas |
| `04_pesos_carteiras.png` | Composição das carteiras ótimas |
| `05_desempenho_historico.png` | Evolução de R$ 100 nas carteiras e no Ibovespa |

## Como interpretar o relatório

1. **Parâmetros:** confira o período, a quantidade de pregões, a Selic usada e a fonte dos dados
   (online ou CSV). Se aparecer a caixa amarela de avisos, algum dado veio do modo offline.
2. **Estatísticas por ativo:** compare retorno e volatilidade. Sharpe negativo significa que a ação
   rendeu menos que a Selic no período. Beta acima de 1 indica uma ação que amplifica os movimentos do Ibovespa.
3. **Matriz de correlação:** vermelho forte indica ações que andam juntas (pouca diversificação);
   tons claros indicam correlação baixa, que é o que mais reduz o risco.
4. **Fronteira eficiente:** cada ponto azul é uma carteira sorteada; quanto mais escuro, maior o Sharpe.
   A linha preta é o limite do que é possível: carteiras abaixo dela são ineficientes. Os pontos cinza
   são as ações isoladas; repare que existem carteiras com risco menor que o de qualquer ação sozinha.
5. **Carteiras ótimas:** a de mínima variância costuma ser bem distribuída entre as ações; a de máximo
   Sharpe tende a se concentrar nas ações com melhor relação retorno/risco no período.
6. **Desempenho histórico:** mostra quanto R$ 100 teriam virado em cada carteira. Como as carteiras ótimas
   foram calculadas com os mesmos dados (análise dentro da amostra), esse resultado é otimista.
7. **Interpretação:** texto gerado automaticamente com base nos números, explicando a diversificação,
   as correlações, os betas e o índice de Sharpe.

## Limitações

- Médias e covariâncias passadas não garantem o comportamento futuro (principalmente a média de retornos).
- A Selic atual é usada para todo o período, como simplificação.
- Não são considerados custos de transação nem impostos.
- Trabalho acadêmico: **não é recomendação de investimento**.

## Solução de problemas

- **`ModuleNotFoundError`**: rode `pip install -r requirements.txt` no mesmo Python usado para executar o programa.
- **Sem internet**: o programa usa os CSVs automaticamente; você também pode forçar com `--offline`.
- **Ticker sem dados**: confira o código no site do Yahoo Finance (ações da B3 terminam em `.SA`).
  Ações sem dados são removidas da análise com um aviso.
