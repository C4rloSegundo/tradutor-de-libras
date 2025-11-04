import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib # Para salvar o modelo

# 1. Carregar os dados
DATA_FILE = 'hand_data.csv'
data = pd.read_csv(DATA_FILE)

print(f"Dados carregados. Total de {len(data)} amostras.")
print("Contagem de amostras por classe:")
print(data['label'].value_counts())

# 2. Separar features (X) e rótulos (y)
# X são as 42 colunas de coordenadas
X = data.drop('label', axis=1) 
# y é a coluna 'label'
y = data['label']

# 3. Dividir os dados em conjuntos de Treino e Teste
# 80% para treino, 20% para teste
# random_state=42 garante que a divisão seja sempre a mesma (reprodutibilidade)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDividindo dados: {len(X_train)} para treino, {len(X_test)} para teste.")

# 4. Treinar o modelo (k-Nearest Neighbors)
# n_neighbors=5 significa que ele vai olhar para os 5 "vizinhos" mais próximos
# para decidir qual é o sinal. Você pode ajustar esse número.
model = KNeighborsClassifier(n_neighbors=5)

print("Iniciando treinamento do modelo k-NN...")
model.fit(X_train, y_train)
print("Treinamento concluído.")

# 5. Testar o modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n--- Resultados ---")
print(f"Acurácia do modelo no conjunto de teste: {acc * 100:.2f}%")

# 6. Salvar o modelo treinado
MODEL_FILE = 'hand_model.joblib'
joblib.dump(model, MODEL_FILE)
print(f"\nModelo salvo com sucesso como '{MODEL_FILE}'")