# 📊 Plano de Apresentação - Pipeline de Análise de Mortalidade com ML (REVISADO)

**Disciplina**: SCC0233 - Aplicações de Aprendizado de Máquina e Mineração de Dados
**Duração**: 10-15 minutos
**Número de slides**: 11
**Foco**: Metodologia técnica e resultados

---

## 🎯 Estrutura Geral da Apresentação

1. **Slides Introdutórios** (2.5 min - 3 slides)
   - Capa
   - O Problema
   - Conjunto de Dados

2. **Limpeza e Transformação de Dados** (2.5 min - 2 slides)
   - Processos de Tratamento Inicial
   - Construção do Painel Município-Ano

3. **Clusterização** (4.5 min - 3 slides)
   - Preparação da Matriz de Features
   - Escolha do Número de Clusters
   - Resultados dos Clusters

4. **Modelos Preditivos** (4.5 min - 3 slides)
   - Construção da Base de Regressão
   - Seleção de Features
   - Modelos Treinados e Resultados

**Tempo Total**: ~14 minutos

---

## 📑 SLIDES DETALHADOS

---

### **SLIDE 1: Capa**
**Tempo**: 15 segundos

#### Conteúdo:
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Pipeline de Análise de Mortalidade com ML        │
│   Clustering e Predição para Políticas Públicas    │
│                                                     │
│   SCC0233 - ICMC/USP                                │
│   [Seu nome/grupo]                                  │
│   Dezembro 2025                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### Elementos visuais:
- Logo do ICMC/USP (canto superior)
- Ícone de dados governamentais ou saúde pública
- Fundo sóbrio e profissional (azul escuro ou gradiente)

#### O que dizer:
> "Bom dia/boa tarde. Hoje vou apresentar nosso projeto final: um pipeline de Machine Learning para análise de dados de mortalidade, com foco em clustering e predição."

---

### **SLIDE 2: O Problema**
**Tempo**: 1 minuto

#### Título:
**Desafio: Diversidade de Perfis de Mortalidade no Brasil**

#### Conteúdo:
```
🏥 CONTEXTO
• Municípios brasileiros têm realidades de saúde muito distintas
• Causas de morte variam drasticamente entre regiões
• Necessidade de ferramentas para segmentação e previsão

❓ PERGUNTAS-CHAVE
1. Como agrupar municípios com perfis de mortalidade similares?
2. É possível prever óbitos futuros para planejamento?
3. Quais padrões podem ser identificados nos dados?
```

#### Elementos visuais:
- Mapa do Brasil com municípios em cores diferentes
- Ícones: hospital, gráfico de tendência, dados

#### O que dizer:
> "O Brasil tem mais de 5.500 municípios com realidades muito diferentes. Nosso desafio é usar dados de mortalidade para identificar perfis similares e prever tendências futuras usando técnicas de Machine Learning."

---

### **SLIDE 3: Conjunto de Dados**
**Tempo**: 1 minuto

#### Título:
**Base de Dados: SIM/DATASUS**

#### Conteúdo:
```
📊 DATASET
• Fonte: Sistema de Informações sobre Mortalidade (DATASUS)
• Volume: ~2,5 milhões de registros de óbitos
• Período: 2000-2023
• Cobertura: Municípios brasileiros

📋 PRINCIPAIS VARIÁVEIS
• Data e local do óbito (município, código IBGE)
• Causa básica (CID-10)
• Dados demográficos: idade, sexo, raça/cor
• Estado civil, escolaridade, ocupação
• Total: 12 variáveis

🔓 DADOS ABERTOS
Disponíveis publicamente em datasus.saude.gov.br
```

#### Elementos visuais:
- Logo do DATASUS
- Ícone de "dados abertos" (cadeado aberto)
- Miniatura de uma tabela mostrando as colunas principais

#### O que dizer:
> "Utilizamos dados do SIM, sistema oficial de registro de óbitos do Ministério da Saúde. São 2,5 milhões de registros públicos, cobrindo mais de 20 anos de história. Isso nos dá uma base sólida para análise."

---

## 📝 SESSÃO 1: LIMPEZA E TRANSFORMAÇÃO DE DADOS

---

### **SLIDE 4: Processos de Tratamento Inicial**
**Tempo**: 1.25 minutos

#### Título:
**Transformações nos Dados: Do Bruto ao Estruturado**

#### Conteúdo:
```
🔧 CONVERSÕES DE TIPOS
• DTOBITO: string → datetime
  → Criação de ano_obito e mes_obito
• CODMUNRES: mantido como string (preservar zeros)
• IDADE: conversão para numérico com validação

📊 CRIAÇÃO DE GRUPOS DE CAUSAS
Agrupamento de códigos CID-10 em 7 categorias:
• Circulatórias (I) → Infarto, AVC, hipertensão
• Respiratórias (J) → Pneumonia, asma, DPOC
• Endócrinas (E) → Diabetes, desnutrição
• Neoplasias (C, D) → Câncer
• Infecciosas (A, B) → Tuberculose, HIV
• Externas (V, W, X, Y) → Acidentes, violência
• Outras → Demais causas

🔢 VARIÁVEIS BINÁRIAS
• is_masculino: 1 se sexo = "M"
• is_preta_parda: 1 se raça/cor = 2 ou 4
```

#### Elementos visuais:
- Diagrama de fluxo: Dado Bruto → Transformações → Dado Limpo
- Tabela mostrando exemplo de agrupamento CID-10
- Ícones para cada tipo de transformação

#### O que dizer:
> "Primeiro passo: transformar dados brutos em formato analisável. Convertemos datas, agrupamos os códigos CID-10 em 7 grandes grupos de causas, e criamos variáveis binárias para facilitar análises demográficas. Essas transformações são essenciais para as etapas seguintes."

---

### **SLIDE 5: Construção do Painel Município-Ano**
**Tempo**: 1.25 minutos

#### Título:
**Painel Município-Ano: Base para Clustering**

#### Conteúdo:
```
🎯 OBJETIVO DA AGREGAÇÃO
Transformar 2.5M de registros individuais em painel estruturado
→ Cada linha = 1 município em 1 ano específico

📊 ESTRUTURA FINAL (19 colunas)

Identificadores (2):
• CODMUNRES, ano_obito

Métricas Agregadas (4):
• obitos_total
• idade_media
• prop_masculino, prop_preta_parda

Contagens por Causa (5):
• obitos_circulatorias, obitos_endocrinas
• obitos_neoplasias, obitos_outras
• obitos_respiratorias

Proporções por Causa (5):
• prop_circulatorias, prop_endocrinas
• prop_neoplasias, prop_outras
• prop_respiratorias

✅ RESULTADO
Painel pronto para clustering e modelagem preditiva
```

#### Elementos visuais:
- Diagrama: Dados Individuais → Agregação → Painel
- Tabela exemplo mostrando algumas linhas do painel
- Destaque visual para as 19 colunas

#### O que dizer:
> "Agregamos os dados em nível município-ano. Cada linha do painel representa um município em um ano, com 19 colunas incluindo contagens e proporções de causas. Este formato é ideal para clustering, pois cada município-ano vira um ponto no espaço de features."

---

## 🔵 SESSÃO 2: CLUSTERIZAÇÃO

---

### **SLIDE 6: Preparação da Matriz de Features**
**Tempo**: 1.5 minutos

#### Título:
**Clustering: Preparação da Matriz e Padronização**

#### Conteúdo:
```
🎯 OBJETIVO
Agrupar municípios com perfis de mortalidade similares

📊 SELEÇÃO DE DADOS
• Ano utilizado: 2023 (mais recente)
• Total de municípios: 27
• Algoritmo: K-Means

🔧 FEATURES SELECIONADAS (9 variáveis)

Proporções de Causas (7):
• prop_circulatorias
• prop_endocrinas
• prop_neoplasias
• prop_outras
• prop_respiratorias
• prop_infecciosas (implícita)
• prop_externas (implícita)

Demográficas (2):
• idade_media
• prop_masculino

⚖️ PADRONIZAÇÃO Z-SCORE
z = (x - média) / desvio_padrão

POR QUE?
✓ Equaliza escalas diferentes
✓ Evita dominância de features com valores maiores
✓ Melhora convergência do K-Means
```

#### Elementos visuais:
- Fórmula do z-score destacada
- Diagrama mostrando antes/depois da padronização
- Matriz 27×9 representada visualmente

#### O que dizer:
> "Para o clustering, selecionamos dados de 2023 com 27 municípios. Usamos 9 features: proporções das causas de morte e características demográficas. CRUCIAL: padronizamos tudo com z-score para que features em escalas diferentes não distorçam o agrupamento."

---

### **SLIDE 7: Escolha do Número de Clusters**
**Tempo**: 1.5 minutos

#### Título:
**Determinação do K Ótimo**

#### Conteúdo:
```
🔍 METODOLOGIA
Testamos K de 2 a 8 clusters

📊 MÉTRICAS DE AVALIAÇÃO

1. Inércia (WCSS)
   • Soma das distâncias ao quadrado
   • Objetivo: minimizar
   • Busca por "cotovelo" no gráfico

2. Silhouette Score
   • Mede qualidade da separação
   • Intervalo: -1 a 1
   • Objetivo: maximizar

📈 RESULTADOS DOS TESTES

┌───┬──────────┬────────────┐
│ K │ Inércia  │ Silhouette │
├───┼──────────┼────────────┤
│ 2 │  167.27  │   0.230    │
│ 3 │  125.73  │   0.243 ⭐ │
│ 4 │  102.61  │   0.211    │
│ 5 │   91.41  │   0.208    │
│ 6 │   79.78  │   0.200    │
│ 7 │   72.29  │   0.185    │
│ 8 │   60.49  │   0.199    │
└───┴──────────┴────────────┘

✅ K ÓTIMO = 3
Critério: Maior Silhouette Score (0.243)
```

#### Elementos visuais:
- **IMAGEM PRINCIPAL**: Gráfico de cotovelo (se disponível)
- **IMAGEM SECUNDÁRIA**: Gráfico de silhouette por K
- Tabela formatada com destaque para K=3

#### O que dizer:
> "Testamos sistematicamente K de 2 a 8. [Apontar para tabela] Usamos inércia e silhouette como métricas. O K=3 apresentou o melhor silhouette score (0.243), indicando boa separação entre clusters. Este será nosso modelo final."

---

### **SLIDE 8: Resultados dos Clusters**
**Tempo**: 1.5 minutos

#### Título:
**3 Perfis Distintos de Mortalidade**

#### Conteúdo:
```
📊 DISTRIBUIÇÃO DOS MUNICÍPIOS
• Cluster 0: 7 municípios (25.9%)
• Cluster 1: 5 municípios (18.5%)
• Cluster 2: 15 municípios (55.6%)

🔵 CLUSTER 0 - "Jovens e Respiratórias"
Características:
• Maior proporção masculina (55.2%)
• Alta mortalidade respiratória (25.4%)
• Idade média mais baixa (70.0 anos)
• Alta proporção preta/parda (68.1%)

Interpretação:
→ Municípios com população mais jovem
→ Destaque para doenças respiratórias

🔴 CLUSTER 1 - "Envelhecidos e Neoplasias"
Características:
• Menor proporção masculina (46.7%)
• Alta mortalidade por neoplasias (8.8%)
• Maior idade média (76.0 anos)
• Menor proporção preta/parda (27.1%)

Interpretação:
→ População mais envelhecida
→ Perfil de doenças crônicas

🟢 CLUSTER 2 - "Perfil Circulatório"
Características:
• Perfil equilibrado
• Dominância circulatória (52.5%)
• Idade média moderada (73.6 anos)
• Proporção preta/parda intermediária (57.5%)

Interpretação:
→ Perfil de mortalidade mais típico
→ Doenças circulatórias como principal causa
```

#### Elementos visuais:
- **IMAGEM PRINCIPAL**: Gráfico de barras comparando proporções por cluster
- **IMAGEM SECUNDÁRIA**: Heatmap do perfil dos clusters
- Usar cores para cada cluster (azul, vermelho, verde)

#### O que dizer:
> "Identificamos 3 perfis bem distintos. [Apontar para gráfico] Cluster 0 tem população mais jovem com alta mortalidade respiratória. Cluster 1 é mais envelhecido com destaque para neoplasias. Cluster 2 é o mais comum, dominado por doenças circulatórias. Cada perfil sugere necessidades de saúde pública diferentes."

---

## 📈 SESSÃO 3: MODELOS PREDITIVOS

---

### **SLIDE 9: Construção da Base de Regressão**
**Tempo**: 1.5 minutos

#### Título:
**Modelo Preditivo: Estruturação com Lag Temporal**

#### Conteúdo:
```
🎯 OBJETIVO
Prever número de óbitos em t+1 baseado em dados de t

🔧 CRIAÇÃO DO LAG TEMPORAL

Variáveis Criadas:
• obitos_t1 (target)
  → Óbitos no próximo ano (shift -1)

• obitos_t_1 (lag)
  → Óbitos no ano anterior (shift +1)

• crescimento_obitos
  → (obitos_t - obitos_t-1) / obitos_t-1
  → Taxa de crescimento

📊 ESTRUTURA DO PROBLEMA

Input (X):  Características do município no ano t
Output (y): Número de óbitos no ano t+1

⏰ SPLIT TEMPORAL (evita vazamento de informação)

┌─────────────────────────────────────────┐
│  TREINO: 2006 a 2021 → 432 registros   │
│  TESTE:  2022        →  27 registros   │
└─────────────────────────────────────────┘

✅ DATASET FINAL
• Total: 459 registros município-ano
• Período: 2006-2022 (com target para 2007-2023)
```

#### Elementos visuais:
- Diagrama temporal mostrando t-1, t, t+1
- Linha do tempo mostrando split treino/teste
- Equação: f(município_t) → óbitos_{t+1}

#### O que dizer:
> "Para predição, criamos features com lag temporal. O target é óbitos no próximo ano, e incluímos crescimento recente como feature. IMPORTANTE: fizemos split temporal, não aleatório. Treinamos com 2006-2021 e testamos em 2022, simulando uso real."

---

### **SLIDE 10: Seleção de Features**
**Tempo**: 1.5 minutos

#### Título:
**Features Preditivas: 12 Variáveis Selecionadas**

#### Conteúdo:
```
📊 MATRIZ DE FEATURES (12 variáveis)

🔢 Óbitos (2):
• obitos_total → Total de óbitos no ano t
• crescimento_obitos → Taxa de crescimento recente

👥 Demográficas (3):
• idade_media → Idade média dos óbitos
• prop_masculino → Proporção de óbitos masculinos
• prop_preta_parda → Proporção preta/parda

🏥 Cluster (1):
• cluster → Grupo do município (resultado do clustering)

📅 Temporal (1):
• ano_obito → Tendência temporal

⚕️ Proporções de Causas (5):
• prop_circulatorias
• prop_endocrinas
• prop_neoplasias
• prop_outras
• prop_respiratorias

💡 RACIONAL DA SELEÇÃO
✓ Óbitos atuais: forte preditor do futuro (autocorrelação)
✓ Crescimento: captura tendências recentes
✓ Cluster: incorpora perfil de mortalidade
✓ Proporções: padrões de causas influenciam total
✓ Temporal: captura tendências de longo prazo
```

#### Elementos visuais:
- Diagrama visual organizando as 12 features por categoria
- Ícones para cada categoria (números, pessoas, relógio, hospital)
- Destaque para "obitos_total" como feature principal

#### O que dizer:
> "Selecionamos 12 features divididas em 6 categorias. O número de óbitos atual é o preditor mais forte (autocorrelação temporal). Adicionamos crescimento recente, cluster do município, e proporções de causas para capturar nuances. O ano também entra para capturar tendências de longo prazo."

---

### **SLIDE 11: Modelos Treinados e Resultados**
**Tempo**: 1.5 minutos

#### Título:
**Comparação de Modelos: Regressão Linear Vence**

#### Conteúdo:
```
🤖 MODELOS TESTADOS

1. Regressão Linear (scikit-learn)
2. LightGBM (Gradient Boosting)

📊 RESULTADOS NO CONJUNTO DE TESTE

┌────────────────────┬──────────┬──────────┬─────────┐
│ Modelo             │   MAE    │   RMSE   │   R²    │
├────────────────────┼──────────┼──────────┼─────────┤
│ Regressão Linear ⭐│  178.22  │  388.86  │  0.9966 │
│ LightGBM           │  500.32  │  876.39  │  0.9828 │
└────────────────────┴──────────┴──────────┴─────────┘

🏆 MODELO VENCEDOR: REGRESSÃO LINEAR

✅ PERFORMANCE EXCEPCIONAL
• MAE = 178 óbitos (erro médio absoluto)
• R² = 0.9966 → Explica 99.66% da variação!
• RMSE = 388.86

🔍 POR QUE REGRESSÃO LINEAR VENCEU?
✓ Problema apresenta forte linearidade
✓ Autocorrelação temporal é muito alta
✓ Dataset pequeno favorece modelos simples
✓ Evita overfitting de modelos complexos

📈 TOP 5 FEATURES MAIS IMPORTANTES
(por coeficiente absoluto)

1. prop_neoplasias     (+1096.59)
2. prop_respiratorias  (-1051.22)
3. prop_outras         (+1041.95)
4. crescimento_obitos  (-897.67)
5. prop_circulatorias  (-579.70)
```

#### Elementos visuais:
- Tabela formatada comparando os 2 modelos
- **IMAGEM**: Scatter plot previsto vs real (se disponível)
- Destaque visual para R² = 0.9966
- Gráfico de barras com feature importance

#### O que dizer:
> "Testamos Regressão Linear e LightGBM. Surpreendentemente, a Regressão Linear venceu com R² de 0.9966! [Apontar para tabela] Erro médio de apenas 178 óbitos. Por que venceu? O problema é altamente linear - óbitos são estáveis ano a ano. Dataset pequeno também favorece modelos simples. As proporções de causas foram as features mais importantes."

---

## 📊 RESUMO: TIMING E ESTRUTURA

| # | Slide | Tempo | Sessão |
|---|-------|-------|--------|
| 1 | Capa | 15s | Introdução |
| 2 | O Problema | 1min | Introdução |
| 3 | Conjunto de Dados | 1min | Introdução |
| 4 | Processos de Tratamento | 1.25min | Transformação |
| 5 | Painel Município-Ano | 1.25min | Transformação |
| 6 | Preparação da Matriz | 1.5min | Clustering |
| 7 | Escolha do K | 1.5min | Clustering |
| 8 | Resultados Clusters | 1.5min | Clustering |
| 9 | Base de Regressão | 1.5min | Modelos |
| 10 | Seleção de Features | 1.5min | Modelos |
| 11 | Modelos e Resultados | 1.5min | Modelos |
| **TOTAL** | **11 slides** | **~13-14 min** | |

---

## 🎨 DICAS DE DESIGN

### Paleta de Cores Sugerida:
- **Azul escuro** (#1E3A8A): Títulos principais
- **Azul claro** (#60A5FA): Destaques e gráficos
- **Verde** (#10B981): Resultados positivos
- **Vermelho** (#EF4444): Alertas (Cluster 1)
- **Amarelo** (#F59E0B): Destaques numéricos
- **Cinza** (#6B7280): Texto secundário

### Fontes:
- **Títulos**: Montserrat Bold ou Arial Black
- **Corpo**: Open Sans ou Calibri
- **Dados**: Consolas ou Courier New

### Estilo:
- Minimalista e técnico
- Usar bastante espaço em branco
- Máximo 3-4 bullets principais por slide
- Tabelas e gráficos grandes e legíveis
- Evitar parágrafos longos

---

## 📝 ROTEIRO DE FALA CONDENSADO

### Introdução (Slides 1-3) - 2.5 min
*Contextualize o problema e apresente os dados*

### Transformação (Slides 4-5) - 2.5 min
*Explique as transformações e construção do painel*

### Clustering (Slides 6-8) - 4.5 min
*Detalhe a metodologia, escolha de K e resultados*

### Modelos (Slides 9-11) - 4.5 min
*Apresente a estruturação, features e comparação de modelos*

---

## ✅ CHECKLIST PRÉ-APRESENTAÇÃO

- [ ] Números reais do notebook inseridos (substituir [X])
- [ ] Gráficos gerados estão salvos e prontos
- [ ] Timing ensaiado (não ultrapassar 15 min)
- [ ] Texto é legível de longe (fonte >= 18pt)
- [ ] Transições entre slides estão suaves
- [ ] Backup dos arquivos

---

## 🎤 POSSÍVEIS PERGUNTAS E RESPOSTAS

### P1: "Por que K=3 e não K=4 ou mais?"
**R**: "Usamos o silhouette score como critério principal de qualidade. K=3 apresentou o melhor score (0.243), indicando melhor separação entre clusters. K maiores fragmentavam demais os grupos sem ganho de qualidade."

### P2: "Por que a Regressão Linear venceu o LightGBM?"
**R**: "O problema apresenta forte linearidade - o número de óbitos é muito estável ano a ano (alta autocorrelação temporal). Com dataset relativamente pequeno (459 registros), modelos simples evitam overfitting e performam melhor."

### P3: "Como garantem que não há vazamento de informação?"
**R**: "Fizemos split temporal rigoroso: treinamos com anos até 2021 e testamos apenas em 2022. Isso simula o uso real, onde usamos o passado para prever o futuro."

### P4: "Qual a principal limitação do projeto?"
**R**: "Não incorporamos variáveis socioeconômicas (PIB, IDH) que influenciam mortalidade. Também trabalhamos com agregação anual, perdendo sazonalidade mensal. São oportunidades para trabalhos futuros."

---

## 🎯 OBJETIVO FINAL DA APRESENTAÇÃO

Ao final, a banca deve entender:
1. ✅ Processo completo de transformação dos dados brutos
2. ✅ Metodologia rigorosa de clustering (K-Means com K=3)
3. ✅ Construção de modelo preditivo com lag temporal
4. ✅ Comparação objetiva entre modelos
5. ✅ Interpretação clara dos resultados

---

**BOA APRESENTAÇÃO! 🚀**
