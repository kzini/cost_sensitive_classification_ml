# Estudo comparativo de estratégias de seleção de features e modelos em um problema de classificação sensível a falso negativo
## Aplicação em diagnóstico de câncer de mama (Breast Cancer Wisconsin Dataset)

## Objetivo do projeto

Investigar o impacto de diferentes estratégias de seleção de features, redução de dimensionalidade e modelagem supervisionada em um problema de classificação binária sensível a falso negativo, utilizando o diagnóstico de câncer de mama como caso de estudo.
Tendo como foco não apenas desempenho absoluto, mas a análise da estrutura do erro (FN vs FP), dos trade-offs entre métricas, da estabilidade em validação cruzada e da coerência entre decisões metodológicas e comportamento do modelo.

## Técnicas utilizadas
### Foram avaliadas múltiplas estratégias de seleção de features e redução de dimensionalidade, incluindo:

- Análise por correlação com inspeção visual (violino + swarmplot);
- SelectKBest;
- RFE;
- Redução de dimensionalidade com PCA.

## Modelos avaliados

- Regressão logística;
- Random forest;
- XGBoost (com e sem otimização de hiperparâmetros);
- SVC;
- KNN.

## Validação

- Validação cruzada estratificada (StratifiedKFold)

## Critério de decisão e resultados

O modelo final selecionado foi o XGBoost com hiperparâmetros otimizados, por apresentar o melhor equilíbrio entre recall, estabilidade em validação cruzada e controle de trade-off com precisão.

| Métrica   | Valor   |
|-----------|---------|
| Recall    | 0.9767  |
| Precisão  | 1.0000  |
| F1-Score  | 0.9882  |
| AUC       | 0.9964  |

Resultados interpretados como:
- Detecção eficaz de casos malignos;  
- Nenhum falso positivo;
- Estabilidade entre treino e validação cruzada.

## Decisão sobre balanceamento de classes

O dataset apresenta distribuição moderadamente assimétrica (62,74% vs 37,26%), Por esse motivo, optei por não aplicar técnicas agressivas de reamostragem. 

Para mitigar possível viés da classe majoritária, utilizei class_weight='balanced' em modelos sensíveis, ajustando a penalização dos erros sem alterar a estrutura original dos dados.

Experimentos com undersampling e oversampling indicaram aumento de variância e indícios de overfitting. O uso de SMOTE não apresentou ganho consistente de recall ou melhoria estrutural nas métricas.

## Outras estratégias avaliadas para aumento de Recall

- Threshold Moving – reduções no limiar aumentaram marginalmente o recall, porém com queda substancial das demais métricas.

## Preparação para deploy

O modelo final foi encapsulado em uma Pipeline reproduzível, treinada novamente no conjunto de treino e serializado em:  
models/xgboost_breast_cancer_fs_optimized.pkl   
Esse modelo é carregado pelo aplicativo interativo (app.py), permitindo simulação de predições e análise operacional do sistema.

## Lições Aprendidas

- A importância de priorizar a métrica certa conforme o contexto (neste caso, Recall);
- Como a seleção de features pode impactar no desempenho dos modelos;
- O papel dos hiperparâmetros no refinamento do modelo;
- A importância da validação cruzada para evitar overfitting;
- Estratégias para lidar com balanceamento de classes e entender quando são realmente necessárias;
- Deploy.

## Estrutura do projeto

```
breast-cancer-classification/
├── data/
│ └── breast cancer kaggle.csv
├── notebooks/
│ └── breast_cancer.ipynb
├── src/
│ ├── feature_selection.py
│ ├── model_evaluation.py
│ ├── models.py
│ └── utils.py
├── app.py
├── Dockerfile
├── README.md
└── requirements.txt
```

## Dataset

- **Fonte**: [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Descrição**: Dados clínicos de exames de mama com rótulo binário (maligno ou benigno)

## Como Reproduzir

### 1. Clonar o repositório
```bash
git clone https://github.com/kzini/cost_sensitive_classification_ml.git
cd cost_sensitive_classification_ml
```

### 2. Rodar o aplicativo interativo com Docker
```bash 
docker build -t cancer-app .
docker run -p 8501:8501 -v "$(pwd)/data:/src/data" cancer-app
```

Abra no navegador: http://localhost:8501

### 3. Reproduzir experimentos e análises
```bash
pip install -r requirements.txt
jupyter notebook notebook/
```

> Desenvolvido por Bruno Casini  
> LinkedIn: www.linkedin.com/in/kzini
