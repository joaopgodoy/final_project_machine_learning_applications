# Pipeline de Análise de Mortalidade com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Disciplina**: SCC0233 - Aplicações de Aprendizado de Máquina e Mineração de Dados
> **Instituição**: Instituto de Ciências Matemáticas e de Computação (ICMC) - USP
> **Objetivo**: Análise exploratória, clustering e predição de óbitos usando dados do tipo SIM/DATASUS

---

## Índice

1. [O que é este projeto?](#o-que-é-este-projeto)
2. [Por que este projeto é importante?](#por-que-este-projeto-é-importante)
3. [Dados utilizados](#dados-utilizados)
4. [Tecnologias e bibliotecas](#tecnologias-e-bibliotecas)
5. [Instalação e execução](#instalação-e-execução)
6. [Transformações nos Dados](#transformações-nos-dados)
7. [Clustering de Municípios](#clustering-de-municípios)
8. [Modelos de Predição](#modelos-de-predição)
9. [Modelos de Regressão e Boosting por Cluster](#modelos-de-regressão-e-boosting-por-cluster)
10. [Autores](#autores)

---

## O que é este projeto?

Este projeto implementa um **pipeline completo de análise de dados de mortalidade** utilizando técnicas de **Machine Learning**. Ele foi desenvolvido como trabalho prático para a disciplina de Aplicações de ML e Mineração de Dados do ICMC-USP.

### Em termos simples:

Imagine que você tem milhões de registros de óbitos (mortes) de diferentes municípios brasileiros ao longo de vários anos. Este projeto usa esses dados para:

1. **Entender padrões**: Quais são as principais causas de morte? Como elas variam por região, sexo e idade?
2. **Agrupar municípios similares**: Identificar grupos de municípios que têm perfis de mortalidade parecidos (por exemplo, municípios com muitas mortes por violência vs. municípios com mais mortes por doenças crônicas).
3. **Prever o futuro**: Criar modelos inteligentes que conseguem prever quantos óbitos um município terá no próximo ano, ajudando gestores públicos a se planejarem melhor.

### Técnicas de ML utilizadas:

- **Machine Learning Não Supervisionado (Clustering)**: Agrupamento automático de municípios por similaridade
- **Machine Learning Supervisionado (Regressão)**: Previsão de valores futuros baseada em dados históricos

---

## Por que este projeto é importante?

### Contexto de Saúde Pública

Os dados de mortalidade são fundamentais para:

- **Planejamento em Saúde**: Saber onde e quando recursos hospitalares serão necessários
- **Políticas Públicas**: Direcionar programas de prevenção (ex: campanhas de trânsito em regiões com muitas mortes por acidente)
- **Equidade**: Identificar desigualdades em saúde entre diferentes grupos populacionais
- **Vigilância Epidemiológica**: Detectar surtos e tendências preocupantes

### Aplicações Práticas

Com este tipo de análise, é possível:

- Identificar municípios que precisam de mais investimento em segurança pública
- Prever aumentos sazonais de mortalidade e preparar a rede de saúde
- Avaliar o impacto de políticas públicas ao longo do tempo
- Alocar recursos de forma mais eficiente e justa

---

## Dados utilizados

### Fonte

Os dados simulam a estrutura do **Sistema de Informações sobre Mortalidade (SIM)** do DATASUS (Departamento de Informática do SUS), que registra todos os óbitos no Brasil.

### Arquivo principal

- **Nome**: `dataset/dataset.csv`
- **Tamanho**: ~2.547.578 registros (linhas)
- **Período**: Dados de 2000 a 2023

### Estrutura das colunas

| Coluna      | Descrição                                      | Exemplo        |
|-------------|------------------------------------------------|----------------|
| `Municipio` | Nome do município                              | "Rio Branco"   |
| `Ano`       | Ano do óbito                                   | 2000           |
| `DTOBITO`   | Data do óbito                                  | "2000-01-19"   |
| `CAUSABAS`  | Causa básica da morte (CID-10)                | "I219"         |
| `CODMUNRES` | Código IBGE do município de residência        | "1200401"      |
| `IDADE`     | Idade do falecido                              | 76             |
| `SEXO`      | Sexo (M/F)                                     | "M"            |
| `RACACOR`   | Raça/cor (códigos: 1-Branca, 2-Preta, etc.)   | 1              |
| `ESTCIV`    | Estado civil                                   | 2              |
| `ESC`       | Escolaridade                                   | 5              |
| `OCUP`      | Ocupação                                       | "00700"        |
| `ID`        | Identificador único do registro                | 1              |

### Classificação Internacional de Doenças (CID-10)

A coluna `CAUSABAS` usa códigos CID-10, onde cada letra representa um capítulo:

- **A, B**: Doenças infecciosas e parasitárias
- **C, D**: Neoplasias (tumores/câncer)
- **E**: Doenças endócrinas e metabólicas (diabetes, etc.)
- **I**: Doenças do aparelho circulatório (infarto, AVC)
- **J**: Doenças do aparelho respiratório (pneumonia, etc.)
- **V, W, X, Y**: Causas externas (acidentes, violência)
- Outros capítulos para demais causas

---

## Tecnologias e bibliotecas

### Linguagem

- **Python 3.8+**

### Bibliotecas principais

#### Manipulação de dados
- `pandas`: Análise e manipulação de dados tabulares
- `numpy`: Operações numéricas e arrays

#### Machine Learning
- `scikit-learn`: Algoritmos de ML (K-Means, Regressão Linear, métricas, StandardScaler)
- `lightgbm`: Gradient Boosting de alto desempenho

#### Visualização
- `matplotlib`: Gráficos estáticos
- `seaborn`: Visualizações estatísticas elegantes

#### Ambiente
- `jupyter`: Notebooks interativos

---

## Instalação e execução

### Pré-requisitos

1. **Python 3.8 ou superior** instalado
2. **Jupyter Notebook** ou **JupyterLab**
3. **Git** (para clonar o repositório)

### Passo 1: Clonar o repositório

```bash
git clone https://github.com/seu-usuario/machine_learning_applications.git
cd machine_learning_applications
```

### Passo 2: Criar ambiente virtual (recomendado)

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

### Passo 3: Instalar dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm jupyter
```

### Passo 4: Garantir que o dataset está no lugar

Certifique-se de que o arquivo `dataset/dataset.csv` existe:

```bash
ls dataset/dataset.csv
```

### Passo 5: Executar o notebook

```bash
jupyter notebook analise_mortalidade.ipynb
```

### Passo 6: Executar as células

No Jupyter:
- **Opção 1**: Executar célula por célula (Shift + Enter)
- **Opção 2**: Executar tudo de uma vez (Menu: Cell → Run All)

---

## Transformações nos Dados

Nesta etapa, os dados brutos de óbitos individuais foram transformados e agregados para criar um painel analítico estruturado.

### 1. Conversão de Tipos de Dados

**Conversão de data:**
- `DTOBITO` (string) → `datetime`
- Criação de variáveis derivadas: `ano_obito` e `mes_obito`

**Conversão de códigos:**
- `CODMUNRES` mantido como string (preservando zeros à esquerda dos códigos IBGE)
- `IDADE` convertida para numérico, com tratamento de valores inválidos

### 2. Criação de Grupos de Causas

Os códigos CID-10 foram agrupados em **7 categorias principais** baseadas no primeiro caractere:

| CID-10 (1º caractere) | Grupo de Causa    | Exemplos                           |
|-----------------------|-------------------|------------------------------------|
| A, B                  | Infecciosas       | Tuberculose, HIV, COVID-19         |
| C, D                  | Neoplasias        | Câncer de pulmão, leucemia         |
| E                     | Endócrinas        | Diabetes, desnutrição              |
| I                     | Circulatórias     | Infarto, AVC, hipertensão          |
| J                     | Respiratórias     | Pneumonia, asma, DPOC              |
| V, W, X, Y            | Externas          | Acidentes, homicídios, suicídios   |
| Outros                | Outras            | Demais causas                      |

**Distribuição encontrada:**
- Circulatórias: 55.2%
- Respiratórias: 22.6%
- Endócrinas: 12.2%
- Neoplasias: 6.1%
- Outras: 3.9%

### 3. Criação de Variáveis Binárias

- `is_masculino`: 1 se sexo = "M", 0 caso contrário
- `is_preta_parda`: 1 se raça/cor = 2 (Preta) ou 4 (Parda), 0 caso contrário

### 4. Agrupamento para Construção do Painel Município-Ano

Os dados foram agregados em nível **município-ano**, transformando milhões de registros individuais em um painel estruturado onde **cada linha representa um município em um ano específico**.

**Operações de agregação:**
- **Contagens**: Total de óbitos e contagem por grupo de causa
- **Proporções**: Cálculo das proporções de cada grupo de causa em relação ao total
- **Médias**: Idade média, proporção de sexo masculino, proporção de raça/cor

### 5. Estrutura Final do Painel Município-Ano

O painel final contém **19 colunas principais**:

#### Identificadores
| Coluna          | Descrição                           |
|-----------------|-------------------------------------|
| `CODMUNRES`     | Código IBGE do município            |
| `ano_obito`     | Ano de referência                   |

#### Métricas Agregadas
| Coluna                    | Descrição                                          |
|---------------------------|----------------------------------------------------|
| `obitos_total`            | Total de óbitos no município-ano                   |
| `idade_media`             | Idade média dos óbitos                             |
| `prop_masculino`          | Proporção de óbitos masculinos (0 a 1)             |
| `prop_preta_parda`        | Proporção de óbitos de pessoas pretas/pardas (0 a 1)|

#### Contagens por Grupo de Causa
| Coluna                    | Descrição                                          |
|---------------------------|----------------------------------------------------|
| `obitos_circulatorias`    | Número de óbitos por doenças circulatórias         |
| `obitos_endocrinas`       | Número de óbitos por doenças endócrinas            |
| `obitos_neoplasias`       | Número de óbitos por neoplasias                    |
| `obitos_outras`           | Número de óbitos por outras causas                 |
| `obitos_respiratorias`    | Número de óbitos por doenças respiratórias         |

#### Proporções por Grupo de Causa
| Coluna                    | Descrição                                          |
|---------------------------|----------------------------------------------------|
| `prop_circulatorias`      | Proporção de óbitos circulatórios (0 a 1)          |
| `prop_endocrinas`         | Proporção de óbitos endócrinos (0 a 1)             |
| `prop_neoplasias`         | Proporção de óbitos por neoplasias (0 a 1)         |
| `prop_outras`             | Proporção de outras causas (0 a 1)                 |
| `prop_respiratorias`      | Proporção de óbitos respiratórios (0 a 1)          |

**Exemplo de registro do painel:**
```
CODMUNRES: 110020
ano_obito: 2023
obitos_total: 841
idade_media: 69.8 anos
prop_masculino: 0.530 (53%)
prop_circulatorias: 0.490 (49%)
prop_respiratorias: 0.251 (25.1%)
...
```

Este formato permite análises longitudinais, comparações entre municípios e serve como base para os modelos de clustering e predição.

---

## Clustering de Municípios

O objetivo desta etapa é **agrupar municípios com perfis de mortalidade similares** usando técnicas de aprendizado não supervisionado.

### 1. Preparação da Matriz de Features

**Ano selecionado:** 2023 (ano mais recente disponível)
**Total de municípios:** 27

**Features selecionadas (9 variáveis):**

| Categoria                | Features                                                                |
|--------------------------|-------------------------------------------------------------------------|
| Proporções de causas (7) | `prop_circulatorias`, `prop_endocrinas`, `prop_neoplasias`, `prop_outras`, `prop_respiratorias`, `prop_infecciosas`, `prop_externas` |
| Demográficas (2)         | `idade_media`, `prop_masculino`                                         |
| Étnico-racial (1)        | `prop_preta_parda` (opcional, conforme análise)                         |

**Dimensão da matriz:** 27 municípios × 9 features

### 2. Padronização das Features (Z-Score)

Antes de aplicar o algoritmo K-Means, todas as features foram padronizadas usando **StandardScaler**:

```
z = (x - média) / desvio_padrão
```

**Por que padronizar?**
- Garante que todas as variáveis tenham a mesma escala
- Evita que features com valores maiores dominem o cálculo de distância
- Melhora a performance e convergência do K-Means

### 3. Teste de Diferentes Valores de K

O algoritmo K-Means requer que o número de clusters (K) seja definido previamente. Testamos **K de 2 a 8** usando duas métricas:

#### Métricas de Avaliação

| Métrica              | Descrição                                                      | Objetivo        |
|----------------------|----------------------------------------------------------------|-----------------|
| **Inércia**          | Soma das distâncias ao quadrado de cada ponto ao centro do cluster | Minimizar      |
| **Silhouette Score** | Mede quão bem separados estão os clusters (-1 a 1)            | Maximizar       |

**Resultados dos testes:**

| K | Inércia  | Silhouette |
|---|----------|------------|
| 2 | 167.27   | 0.230      |
| 3 | 125.73   | **0.243**  |
| 4 | 102.61   | 0.211      |
| 5 | 91.41    | 0.208      |
| 6 | 79.78    | 0.200      |
| 7 | 72.29    | 0.185      |
| 8 | 60.49    | 0.199      |

**Método de seleção:**
- Usamos o **maior Silhouette Score** como critério principal
- **K = 3** foi selecionado (Silhouette = 0.243)

### 4. Treinamento com K Ótimo

**Algoritmo:** K-Means
**Número de clusters:** 3
**Parâmetros:**
- `n_clusters=3`
- `random_state=42` (reprodutibilidade)
- `n_init=20` (múltiplas inicializações para melhor convergência)

**Distribuição final dos municípios:**

| Cluster | Número de Municípios | Proporção |
|---------|----------------------|-----------|
| 0       | 7                    | 25.9%     |
| 1       | 5                    | 18.5%     |
| 2       | 15                   | 55.6%     |

### 5. Perfil dos Clusters

Cada cluster apresenta características distintas de mortalidade:

#### Cluster 0 (7 municípios - 26%)
**Características:**
- Maior proporção masculina (55.2%)
- Alta mortalidade respiratória (25.4%)
- Idade média mais baixa (70.0 anos)
- Alta proporção preta/parda (68.1%)

**Interpretação:** Municípios com população mais jovem e maior mortalidade respiratória e por causas externas.

#### Cluster 1 (5 municípios - 19%)
**Características:**
- Menor proporção masculina (46.7%)
- Alta mortalidade por neoplasias (8.8%)
- Maior idade média (76.0 anos)
- Menor proporção preta/parda (27.1%)

**Interpretação:** Municípios com população mais envelhecida, mortalidade dominada por doenças crônicas.

#### Cluster 2 (15 municípios - 56%)
**Características:**
- Perfil equilibrado de proporções
- Alta mortalidade circulatória (52.5%)
- Idade média moderada (73.6 anos)
- Proporção preta/parda intermediária (57.5%)

**Interpretação:** Municípios com perfil de mortalidade mais típico, dominado por doenças circulatórias.

---

## Modelos de Predição

Nesta etapa, desenvolvemos modelos de **Machine Learning supervisionado** para prever o número de óbitos que um município terá no próximo ano.

### 1. Criação do Dataset com Lag Temporal

Para criar um modelo preditivo, precisamos de:
- **Input (X)**: Características do município no ano **t**
- **Output (y)**: Número de óbitos no ano **t+1**

**Transformações aplicadas:**

| Variável              | Descrição                                                    |
|-----------------------|--------------------------------------------------------------|
| `obitos_t1`           | Target: óbitos no próximo ano (shift -1)                     |
| `obitos_t_1`          | Lag: óbitos no ano anterior (shift +1)                       |
| `crescimento_obitos`  | Taxa de crescimento: (óbitos_t - óbitos_t-1) / óbitos_t-1    |

**Features utilizadas (12 variáveis):**

| Categoria           | Features                                                              |
|---------------------|-----------------------------------------------------------------------|
| Óbitos              | `obitos_total`, `crescimento_obitos`                                  |
| Demográficas        | `idade_media`, `prop_masculino`, `prop_preta_parda`                   |
| Cluster             | `cluster` (resultado do clustering)                                   |
| Temporal            | `ano_obito` (tendência temporal)                                      |
| Proporções (5)      | `prop_circulatorias`, `prop_endocrinas`, `prop_neoplasias`, `prop_outras`, `prop_respiratorias` |

**Dataset final:**
- **Total de registros:** 459 município-ano
- **Período:** 2006 a 2022 (com target para 2007 a 2023)

### 2. Split Treino-Teste Temporal

Para evitar **vazamento de informação**, usamos split temporal (não aleatório):

```
Treino: Anos 2006 a 2021 (432 registros)
Teste:  Ano 2022 (27 registros)
```

Isso simula uma situação real: **usar o passado para prever o futuro**.

### 3. Resultados da Regressão Linear

**Modelo:** `LinearRegression` (scikit-learn)

**Métricas no conjunto de teste:**

| Métrica | Valor      | Interpretação                                    |
|---------|------------|--------------------------------------------------|
| **MAE** | 178.22     | Erro médio absoluto de ~178 óbitos               |
| **RMSE**| 388.86     | Raiz do erro quadrático médio                    |
| **R²**  | **0.9966** | Explica 99.66% da variação nos óbitos            |

**Top 5 Features mais importantes (por coeficiente absoluto):**

| Feature                 | Coeficiente | Impacto                              |
|-------------------------|-------------|--------------------------------------|
| `prop_neoplasias`       | +1096.59    | Maior proporção → mais óbitos        |
| `prop_respiratorias`    | -1051.22    | Maior proporção → menos óbitos total |
| `prop_outras`           | +1041.95    | Maior proporção → mais óbitos        |
| `crescimento_obitos`    | -897.67     | Crescimento recente → ajuste negativo|
| `prop_circulatorias`    | -579.70     | Maior proporção → menos óbitos total |

**Interpretação:**
- A Regressão Linear teve **performance excepcional** (R² > 0.99)
- O número de óbitos é altamente previsível baseado em padrões históricos
- O modelo é **simples, rápido e interpretável**

### 4. Resultados do LightGBM

**Modelo:** LightGBM Regressor (Gradient Boosting)

**Parâmetros:**
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 200 (com early stopping)
}
```

**Métricas no conjunto de teste:**

| Métrica | Valor      | Interpretação                                    |
|---------|------------|--------------------------------------------------|
| **MAE** | 500.32     | Erro médio absoluto de ~500 óbitos               |
| **RMSE**| 876.39     | Raiz do erro quadrático médio                    |
| **R²**  | **0.9828** | Explica 98.28% da variação nos óbitos            |

**Top 5 Features mais importantes (por gain):**

| Feature                 | Importância Relativa |
|-------------------------|----------------------|
| `obitos_total`          | 89.5%                |
| `prop_preta_parda`      | 4.4%                 |
| `prop_respiratorias`    | 3.0%                 |
| `cluster`               | 2.3%                 |
| `idade_media`           | 1.7%                 |

**Interpretação:**
- O LightGBM teve **boa performance** mas inferior à Regressão Linear
- O modelo captura relações não-lineares
- `obitos_total` (ano corrente) é de longe a feature mais importante

### 5. Comparação Final dos Modelos

| Modelo              | MAE     | RMSE    | R²      | Melhor em      |
|---------------------|---------|---------|---------|----------------|
| Regressão Linear    | 178.22  | 388.86  | **0.9966** | Todos os critérios |
| LightGBM            | 500.32  | 876.39  | 0.9828  | Flexibilidade  |

**Modelo vencedor:** **Regressão Linear**

**Por que a Regressão Linear venceu?**
- O problema apresenta **forte linearidade** entre features e target
- Número de óbitos é estável ano a ano (alta autocorrelação temporal)
- Dataset relativamente pequeno favorece modelos mais simples
- Evita overfitting que pode ocorrer com modelos mais complexos

### 6. Aplicação Prática

O modelo final foi usado para:
- **Prever óbitos em 2024** para os 27 municípios
- Identificar municípios com maior aumento previsto (ex: Boa Vista +25.7%, Palmas +20.2%)
- Identificar municípios com redução prevista (ex: Macapá -5.6%, Rio Branco -4.5%)
- Fornecer **intervalos de confiança** (IC 95% ≈ ± 762 óbitos)

**Utilidade para gestão pública:**
- Planejamento orçamentário e alocação de recursos
- Dimensionamento de equipes de saúde
- Alertas precoces para municípios com tendência de alta

---

## Modelos de Regressão e Boosting por Cluster

Nesta seção, treinamos modelos específicos para cada cluster identificado, permitindo capturar dinâmicas locais de mortalidade.

### Resultados por Cluster

| Cluster | LR R² | LGB R² | Melhor Modelo |
|---------|-------|--------|---------------|
| 0       | 0.990 | 0.737  | Regressão Linear |
| 1       | 0.992 | 0.900  | Regressão Linear |
| 2       | 0.997 | 0.987  | Regressão Linear |

**Análise:**
- **Cluster 0**: A Regressão Linear teve desempenho muito superior (R² 0.99 vs 0.74).
- **Cluster 1**: Ambos foram bem, mas LR venceu novamente.
- **Cluster 2**: Resultados excelentes em ambos, com leve vantagem para LR.

**Conclusão:**
Assim como no modelo geral, a **Regressão Linear** se mostrou mais robusta e precisa para todos os clusters individualmente.

---

## Autores

- **João** (e equipe)