# 🛠️ Scripts de Utilitários

Scripts automatizados para validação e manutenção do sistema.

---

## 📜 Scripts Disponíveis

### [`validate_model.py`](validate_model.py) ⭐

**Script de Validação Automatizada** - Verifica integridade do pipeline GraphSAGE.

#### Validações Implementadas

1. **Vazamento de Dados** 🔴 CRÍTICO
   - Verifica que treino e teste são disjuntos
   - Resultado esperado: 0 músicas em overlap

2. **Normalização L2** 🟡 ALTA
   - Valida que embeddings têm norma = 1.0
   - Essencial para cosine similarity

3. **Qualidade do Coarsening** 🟢 MÉDIA
   - Verifica balanceamento de super-nós
   - Detecta over/under-coarsening

4. **Convergência do Modelo** 🔵 BAIXA
   - Valida que loss é decrescente
   - (Requer salvar histórico de treinamento)

5. **Estrutura de Arquivos** 🟢 MÉDIA
   - Verifica que arquivos esperados existem

#### Uso

**Executar todas as validações**:
```bash
python scripts/validate_model.py
```

**Modo verbose** (com detalhes):
```bash
python scripts/validate_model.py --verbose
```

**Validação específica**:
```bash
# Apenas vazamento de dados
python scripts/validate_model.py --check leak

# Apenas normalização L2
python scripts/validate_model.py --check l2

# Apenas qualidade do coarsening
python scripts/validate_model.py --check coarsening

# Apenas estrutura de arquivos
python scripts/validate_model.py --check files
```

#### Output Esperado

```
==============================================================================
                      VALIDAÇÃO: VAZAMENTO DE DADOS
==============================================================================

Músicas no treino: 321,160
Músicas no teste: 60,417
Overlap: 0
✅ Nenhum vazamento detectado (split disjunto)

==============================================================================
                      VALIDAÇÃO: NORMALIZAÇÃO L2
==============================================================================

Treino (GraphSAGE):
  Norma média: 1.000000 (±0.000000)
  Range: [1.000000, 1.000000]
✅ Treino (GraphSAGE): Norma unitária OK

Teste (Cold-Start):
  Norma média: 1.000000 (±0.000000)
  Range: [1.000000, 1.000000]
✅ Teste (Cold-Start): Norma unitária OK

==============================================================================
                             RESUMO DAS VALIDAÇÕES
==============================================================================

✅ Estrutura de Arquivos
✅ Vazamento de Dados
✅ Normalização L2
✅ Qualidade do Coarsening
⚠️  Convergência do Modelo

Total: 4/5 validações passaram
```

#### Exit Codes

- `0`: Todas as validações passaram
- `1`: Pelo menos uma validação falhou

**Uso em CI/CD**:
```bash
# No GitHub Actions, GitLab CI, etc.
python scripts/validate_model.py || exit 1
```

---

## 🔧 Outros Scripts (Futuros)

### `generate_report.py` (PLANEJADO)

Gera relatório PDF com resultados do modelo.

```bash
python scripts/generate_report.py --output reports/model_report.pdf
```

### `benchmark_performance.py` (PLANEJADO)

Mede tempo de execução de cada componente.

```bash
python scripts/benchmark_performance.py
```

---

## 🎯 Como Adicionar Novas Validações

1. Abra [`validate_model.py`](validate_model.py)
2. Adicione nova função `validate_XXX(verbose=False)`:
```python
def validate_XXX(verbose=False):
    print_header("VALIDAÇÃO: XXX")
    try:
        # Sua lógica aqui
        if check_passed:
            print_success("XXX está OK")
            return True
        else:
            print_error("XXX falhou")
            return False
    except Exception as e:
        print_error(f"Erro: {e}")
        return False
```

3. Adicione ao dicionário `validators` em `main()`:
```python
validators = {
    'leak': validate_data_leak,
    'l2': validate_l2_normalization,
    'xxx': validate_XXX,  # ← Nova validação
}
```

4. Adicione à função `validate_all()`:
```python
results = {
    "Vazamento de Dados": validate_data_leak(verbose),
    "Normalização L2": validate_l2_normalization(verbose),
    "XXX": validate_XXX(verbose),  # ← Nova validação
}
```

---

## 📊 Integração com CI/CD

### GitHub Actions

```yaml
# .github/workflows/validate.yml
name: Validação do Modelo

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python scripts/validate_model.py
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
python scripts/validate_model.py --check leak --check l2
if [ $? -ne 0 ]; then
    echo "❌ Validações falharam. Commit abortado."
    exit 1
fi
```

---

## 🐛 Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'pandas'`

**Solução**: Instale dependências
```bash
pip install pandas numpy
```

### Erro: `FileNotFoundError: .../track_embeddings_mean.parquet`

**Solução**: Execute os notebooks na ordem correta
```bash
# 1. Coarsening
jupyter nbconvert --execute codes/S4_coarsening.ipynb

# 2. GraphSAGE
jupyter nbconvert --execute codes/S6_GraphSAGE.ipynb

# 3. Cold-Start
jupyter nbconvert --execute codes/S7_embeddings_new_tracks.ipynb

# 4. Avaliação
jupyter nbconvert --execute codes/S8_recommender.ipynb
```

### Warning: `Histórico de treinamento não encontrado`

**Solução**: Salve métricas de treinamento no S6

Adicione ao final do loop de treinamento:
```python
# Salvar histórico
df_history = pd.DataFrame(history)
df_history.to_parquet(P['metrics'] / 'training_history.parquet')
```

---

## 📝 Checklist de Validação Pré-Publicação

Rode antes de submeter paper:

```bash
# 1. Validar integridade
python scripts/validate_model.py

# 2. Se tudo passar, gerar relatório
python scripts/generate_report.py  # (quando implementado)

# 3. Commit e push
git add .
git commit -m "Validações passaram - pronto para publicação"
git push
```

---

## 🎓 Referências

- **Validação de Split**: [Kohavi, 1995] - "A study of cross-validation and bootstrap"
- **Normalização de Embeddings**: [Mikolov et al. 2013] - Word2Vec
- **Métricas de Coarsening**: [Valejo et al. 2020] - MLPb paper

---

*Scripts criados em 24/01/2026*
*Parte da documentação de revisão técnica*
