import pandas as pd
import glob
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configurações
DATA_DIR = 'dados_letras'
MODEL_FILE = 'hand_model.joblib'
CONFUSION_MATRIX_FILE = 'matriz_confusao.png'

def carregar_dados():
    print("Carregando dados CSV...")
    # Pega todos os arquivos .csv na pasta dados_letras
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    if not csv_files:
        print(f"ERRO: Nenhum arquivo CSV encontrado em '{DATA_DIR}'. Rode o coletor_dados.py primeiro!")
        exit()

    dataframes = []
    for f in csv_files:
        df = pd.read_csv(f)
        dataframes.append(df)
    
    # Junta tudo num tabelão só
    full_data = pd.concat(dataframes, ignore_index=True)
    print(f"   Total de amostras: {len(full_data)}")
    print(f"   Classes encontradas: {full_data['label'].unique()}")
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
    plt.savefig(CONFUSION_MATRIX_FILE)
    print(f"   Gráfico salvo como '{CONFUSION_MATRIX_FILE}'. Coloque isso no seu PDF!")
    
    # (Opcional) Mostra na tela
    # plt.show() 

    # 5. Salvar Modelo
    joblib.dump(model, MODEL_FILE)
    print(f"\nSUCESSO: Modelo salvo em '{MODEL_FILE}'")

if __name__ == "__main__":
    treinar()