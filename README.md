# GraphSAGE Music Recommendation for Cold-Start

> Trabalho de Conclusão de Curso (TCC) — Implementação de um sistema de recomendação musical para o problema de **Cold-Start** usando **Graph Neural Networks (GNNs)** e **Graph Coarsening**.

## Visão Geral

O objetivo é recomendar faixas musicais novas (lançadas após uma data de corte) que não possuem histórico de interações. A solução combina **estrutura topológica** (via GraphSAGE) e **features de áudio** em uma abordagem híbrida que une filtragem colaborativa e métodos baseados em conteúdo.

### Contribuições Principais

| Componente | Descrição |
|---|---|
| **Graph Coarsening** | Pipeline híbrido que comprime ~300k → ~20k nós preservando comunidades locais |
| **GraphSAGE Indutivo** | Gera embeddings para nós não vistos (super-nós) via agregação de vizinhança |
| **Inferência Cold-Start** | Mapeia novas faixas para super-nós via KNN em features brutas de áudio |

## Estrutura do Projeto

```
tcc-music-recsys-gnn/
├── codes/                   # Notebooks do pipeline (executar em ordem)
│   ├── S0_init_project.ipynb
│   ├── S1_ingest_prep.ipynb
│   ├── S2_transform_load.ipynb
│   ├── S3_load.ipynb
│   ├── S4_coarsening.ipynb
│   ├── S5_projection.ipynb
│   ├── S6_GraphSAGE.ipynb
│   ├── S7_embeddings_new_tracks.ipynb
│   ├── S8_recommender.ipynb
│   ├── S9_baselines.ipynb
│   └── S10_visualization.ipynb
├── conf/
│   └── config.yaml          # Configuração central (paths, hiperparâmetros)
├── reports/
│   └── figures/             # Figuras e gráficos gerados
├── .env.example             # Template de variáveis de ambiente
└── requirements.txt         # Dependências do projeto
```

> **Nota:** As pastas `data/` e `graphs/` são ignoradas pelo git por conterem arquivos pesados (parquet, npz). Veja a seção [Dados](#dados) abaixo.

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/lucasborgs/tcc-music-recsys-gnn.git
cd tcc-music-recsys-gnn
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> **PyTorch Geometric** requer instalação adicional. Consulte a [documentação oficial](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) e escolha a versão compatível com sua versão de CUDA/CPU.

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite `.env` com suas credenciais da API do Spotify (necessário apenas para os notebooks S1 e S2):

```
SPOTIFY_CLIENT_ID=seu_client_id
SPOTIFY_CLIENT_SECRET=seu_client_secret
```

Crie suas credenciais em [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard).

## Dados

Os dados brutos e processados **não estão no repositório** por serem grandes demais. O notebook `S0_init_project.ipynb` cria a estrutura de pastas necessária, e `S1_ingest_prep.ipynb` faz a ingestão dos dados via Kaggle e Spotify API.

Estrutura esperada após execução:

```
data/
├── raw/
├── interim/
└── processed/

graphs/
├── bipartite/
│   └── coarsened/
├── item_item/
└── super_item_item/
```

## Como Executar

Execute os notebooks em ordem sequencial:

| Step | Notebook | Descrição |
|------|----------|-----------|
| S0 | `S0_init_project.ipynb` | Cria estrutura de pastas |
| S1 | `S1_ingest_prep.ipynb` | Ingestão de dados (Kaggle + Spotify API) |
| S2 | `S2_transform_load.ipynb` | Transformação e enriquecimento |
| S3 | `S3_load.ipynb` | Carregamento e validação |
| S4 | `S4_coarsening.ipynb` | Graph Coarsening bipartido |
| S5 | `S5_projection.ipynb` | Projeção item-item |
| S6 | `S6_GraphSAGE.ipynb` | Treinamento do GraphSAGE |
| S7 | `S7_embeddings_new_tracks.ipynb` | Embeddings para faixas novas |
| S8 | `S8_recommender.ipynb` | Sistema de recomendação |
| S9 | `S9_baselines.ipynb` | Avaliação de baselines |
| S10 | `S10_visualization.ipynb` | Visualizações finais |

Todos os paths são gerenciados via `conf/config.yaml` — não é necessário alterar caminhos manualmente.

## Configuração (`conf/config.yaml`)

```yaml
params:
  coarsening_target_nodes: 20000   # Nós alvo após coarsening
  gnn_hidden_channels: 64          # Dimensão dos embeddings
  gnn_lr: 0.001                    # Learning rate
  gnn_epochs: 200
  knn_neighbors: 50                # Vizinhos para cold-start
  seed: 42
```

## Resultados

Os resultados e métricas de avaliação ficam em `reports/`:

- `reports/figures/` — Gráficos de curvas de aprendizado, comparação de métricas, t-SNE
- `reports/metrics/` — Arquivos parquet com métricas detalhadas por playlist

## Tecnologias

- **Python 3.10+**
- **PyTorch + PyTorch Geometric** — GNN (GraphSAGE)
- **NetworkX** — Manipulação de grafos
- **scikit-learn** — KNN, normalização, métricas
- **pandas / pyarrow** — Pipeline de dados
- **Spotipy** — API do Spotify
