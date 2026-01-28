# 🤟 Tradutor de Libras - Sistema de Reconhecimento em Tempo Real

Sistema profissional de reconhecimento de alfabeto em Libras (Língua Brasileira de Sinais) utilizando visão computacional e aprendizado de máquina, desenvolvido com arquitetura limpa e padrões de design modernos.

## 🎯 Características

- **Reconhecimento em Tempo Real**: Traduz sinais de Libras instantaneamente via webcam
- **Interface Profissional**: Painéis informativos com indicadores de confiança e progresso
- **Arquitetura Limpa**: Separação clara entre domínio, casos de uso e adaptadores
- **Alta Acurácia**: Modelo KNN otimizado com normalização de landmarks
- **Sistema de Logging**: Rastreamento completo de operações e erros
- **Facilmente Extensível**: Adicione novos classificadores sem alterar o core

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Webcam funcional
- Windows/Linux/macOS

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/tradutor-de-libras.git
cd tradutor-de-libras
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

## 📖 Como Usar

### Opção 1: Interface Gráfica (Recomendado)

Execute o launcher que centraliza todas as funcionalidades:

```bash
python launcher.py
```

A interface oferece três módulos principais:
- **📷 Capturar Novos Sinais**: Coleta dados para treinamento
- **🧠 Treinar IA**: Treina o modelo com os dados coletados
- **🚀 Iniciar Tradutor**: Executa o sistema de reconhecimento

### Opção 2: Linha de Comando

#### 1. Coletar Dados
```bash
python coletor_dados.py
```
**Comandos:**
- Pressione `A-Y` para gravar sinais da letra correspondente
- `ESPAÇO` para pausar
- `0` para sair

**Recomendação:** Colete pelo menos 100 amostras de cada letra em diferentes posições e iluminações.

#### 2. Treinar Modelo
```bash
python treinar_modelo.py
```
O script irá:
- Carregar todos os dados coletados
- Treinar modelo KNN
- Gerar matriz de confusão (`matriz_confusao.png`)
- Salvar modelo treinado (`hand_model.joblib`)

#### 3. Executar Tradutor
```bash
python main.py
```
**Comandos durante execução:**
- `Q` - Sair
- `ESPAÇO` - Pausar/Continuar reconhecimento
- `C` - Limpar texto
- `B` - Adicionar espaço

## 🏗️ Arquitetura do Projeto

```
tradutor-de-libras/
│
├── config.py                    # Configurações centralizadas
├── launcher.py                  # Interface gráfica principal
├── main.py                      # Aplicação de reconhecimento
├── coletor_dados.py            # Coleta de amostras
├── treinar_modelo.py           # Treinamento do modelo
│
├── core/                        # Camada de Domínio (regras de negócio)
│   ├── domain.py               # Entidades (Gesto, MaoDetectada)
│   └── use_cases.py            # Casos de uso (ReconhecerSinalUseCase)
│
├── adapters/                    # Camada de Adaptação
│   ├── mediapipe_adapter.py    # Adaptador para MediaPipe
│   └── sklearn_adapter.py      # Adaptador para scikit-learn
│
├── dados_letras/               # Dataset de treinamento (CSVs)
│   ├── A.csv
│   ├── B.csv
│   └── ...
│
├── hand_model.joblib           # Modelo treinado
├── matriz_confusao.png         # Gráfico de avaliação
└── tradutor_libras.log         # Logs do sistema
```

### Princípios Arquiteturais

- **Clean Architecture**: Separação em camadas (Domain, Use Cases, Adapters)
- **Dependency Inversion**: Core não depende de frameworks externos
- **Interface Segregation**: Contratos claros (IClassificador)
- **Single Responsibility**: Cada classe com responsabilidade única

## ⚙️ Configuração

Todas as configurações estão centralizadas em `config.py`:

```python
# Reconhecimento
LIMITE_BUFFER = 30          # Frames para confirmar letra (~1 seg)
CONFIANCA_MINIMA = 0.6      # Confiança mínima (0.0 a 1.0)
FRAME_SKIP = 2              # Processar 1 a cada N frames

# Câmera
CAMERA_INDEX = 0            # Índice da webcam

# Modelo
KNN_NEIGHBORS = 5           # Vizinhos do KNN
```

## 🔧 Tecnologias Utilizadas

| Tecnologia | Uso |
|-----------|-----|
| **OpenCV** | Captura de vídeo e interface |
| **MediaPipe** | Detecção de landmarks da mão |
| **scikit-learn** | Classificação (KNN) |
| **NumPy/Pandas** | Manipulação de dados |
| **Matplotlib/Seaborn** | Visualização de métricas |
| **Tkinter** | Interface gráfica do launcher |

## 📊 Performance

- **Acurácia**: ~95% (varia conforme qualidade dos dados)
- **FPS**: 15-30 (dependendo do hardware)
- **Latência**: <100ms por predição

## 🧪 Testes e Validação

O sistema inclui:
- ✅ Validação cruzada (80% treino / 20% teste)
- ✅ Matriz de confusão detalhada
- ✅ Relatório de classificação por letra
- ✅ Logging de operações críticas

## 🐛 Troubleshooting

### Erro: "Modelo não encontrado"
**Solução:** Execute `python treinar_modelo.py` antes de usar o tradutor.

### Erro: "Câmera não encontrada"
**Solução:** 
1. Verifique se a webcam está conectada
2. Altere `CAMERA_INDEX` em `config.py` (tente 0, 1 ou 2)
3. Feche outros programas usando a câmera

### Baixa acurácia
**Solução:**
1. Colete mais amostras (mínimo 100 por letra)
2. Varie posições e iluminação durante coleta
3. Mantenha mão totalmente visível
4. Ajuste `MIN_DETECTION_CONFIDENCE` em `config.py`

### Sistema lento
**Solução:**
1. Aumente `FRAME_SKIP` em `config.py` (ex: 3 ou 4)
2. Reduza resolução da webcam
3. Feche aplicações em segundo plano

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

**Thiago**
- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- Email: seu-email@example.com

## 🙏 Agradecimentos

- **MediaPipe** pela excelente biblioteca de detecção de mãos
- **scikit-learn** pelos algoritmos de ML robustos
- Comunidade surda brasileira pela inspiração

## 📚 Referências

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [Clean Architecture - Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Libras - Língua Brasileira de Sinais](http://www.libras.org.br/)

---

<div align="center">
  
**⭐ Se este projeto foi útil, deixe uma estrela! ⭐**

</div>
