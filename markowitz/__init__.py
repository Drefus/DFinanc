"""
Pacote "markowitz": aplicação prática da Teoria Moderna de Carteiras
(Harry Markowitz, 1952) com ações reais da B3.

Organização dos módulos (cada um tem uma responsabilidade única):

    coleta_dados.py   -> baixa cotações (Yahoo Finance) e a Selic (Banco Central),
                         com fallback para arquivos CSV locais na pasta data/.
    risco_retorno.py  -> calcula retornos, volatilidades, covariância,
                         correlação, beta e o desempenho histórico das carteiras.
    otimizacao.py     -> simulação de Monte Carlo, carteira de mínima variância,
                         carteira de máximo Sharpe e fronteira eficiente.
    graficos.py       -> gera os gráficos PNG com matplotlib.
    relatorio.py      -> monta o relatório HTML, o CSV de resultados e o
                         resumo impresso no terminal.

O arquivo main.py (na raiz do projeto) orquestra todos esses módulos.
"""
