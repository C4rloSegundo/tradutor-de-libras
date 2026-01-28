import pandas as pd
import glob
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def carregar_dados():
    """Carrega todos os arquivos CSV da pasta de dados.
    
    Returns:
        DataFrame do pandas com todos os dados concatenados.
        
    Raises:
        SystemExit: Se nenhum arquivo CSV for encontrado.
    """
    logger.info("Carregando dados CSV...")
    csv_files = glob.glob(os.path.join(config.DATA_DIR, '*.csv'))
    
    if not csv_files:
        logger.error(f"Nenhum arquivo CSV encontrado em '{config.DATA_DIR}'")
        print(f"ERRO: Nenhum arquivo CSV encontrado em '{config.DATA_DIR}'.")
        print("Execute o coletor_dados.py primeiro para capturar sinais!")
        exit(1)

    dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dataframes.append(df)
            logger.info(f"Carregado: {os.path.basename(f)} ({len(df)} amostras)")
        except Exception as e:
            logger.error(f"Erro ao carregar {f}: {e}")
    
    full_data = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total de amostras: {len(full_data)}")
    logger.info(f"Classes encontradas: {sorted(full_data['label'].unique())}")
    return full_data

def treinar():
    data = carregar_dados()

    # 2. Preparação
    X = data.drop('label', axis=1) # As coordenadas (Features)
    y = data['label']              # A letra (Target)

    # Separa 20% para teste (validação)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Treinamento (KNN)
    print("\nTreinando modelo KNN...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # 4. Avaliação
    print("Avaliando modelo...")
    y_pred = model.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"   Acurácia Final: {acuracia * 100:.2f}%")

    # --- GERAÇÃO DE GRÁFICOS PARA O TCC ---
    print("\nGerando Matriz de Confusão para o TCC...")
    
    # Cria a matriz
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(y.unique()) # Garante ordem alfabética

    # Desenha o gráfico
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Matriz de Confusão (Acurácia: {acuracia*100:.1f}%)')
    plt.ylabel('Letra Real')
    plt.xlabel('Previsão do Modelo')
    
    # Salva a imagem
    plt.savefig(config.CONFUSION_MATRIX_FILE)
    print(f"   Gráfico salvo como '{config.CONFUSION_MATRIX_FILE}'. Coloque isso no seu PDF!")
    
    # (Opcional) Mostra na tela
    # plt.show() 

    # 5. Salvar Modelo
    joblib.dump(model, config.MODEL_FILE)
    print(f"\nSUCESSO: Modelo salvo em '{config.MODEL_FILE}'")

if __name__ == "__main__":
    treinar()