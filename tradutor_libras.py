import cv2
import mediapipe as mp
import joblib # Para carregar o modelo treinado
import numpy as np

# 1. Carregar o modelo treinado
MODEL_FILE = 'hand_model.joblib'
try:
    model = joblib.load(MODEL_FILE)
    print(f"Modelo '{MODEL_FILE}' carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo do modelo '{MODEL_FILE}' não encontrado.")
    print("Por favor, execute o script 'treinar_modelo.py' primeiro.")
    exit()

# 2. Inicializar o MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 3. Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

print("\nIniciando tradutor em tempo real...")
print("Pressione 'q' para sair.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image_output = image.copy()

    prediction = "" # Texto da previsão

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Desenha a mão
        mp_drawing.draw_landmarks(
            image_output, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS
        )

        try:
            # --- Preparação dos Dados (EXATAMENTE IGUAL AO COLETOR) ---
            
            # 1. Pega todos os pontos
            points = hand_landmarks.landmark
            
            # 2. Encontra o ponto (x,y) mínimo
            x_coords = [point.x for point in points]
            y_coords = [point.y for point in points]
            min_x = min(x_coords)
            min_y = min(y_coords)

            # 3. Calcula a posição relativa (normalização)
            data_row = []
            for point in points:
                data_row.append(point.x - min_x)
                data_row.append(point.y - min_y)
            
            # 4. Transformar para o formato que o modelo espera (Numpy array 2D)
            # Precisamos usar [data_row] para simular "uma amostra"
            data_to_predict = np.array([data_row])

            # --- Fazer a Previsão ---
            prediction_array = model.predict(data_to_predict)
            prediction = prediction_array[0] # Pega o primeiro (e único) resultado
        
        except Exception as e:
            print(f"Erro durante a previsão: {e}")
            prediction = "Erro"

    # --- Mostrar a Previsão na Tela ---
    # Configurações do texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    text_color = (0, 255, 0) # Verde
    
    # Desenha o texto da previsão no canto superior esquerdo
    cv2.putText(
        image_output, 
        str(prediction), 
        (50, 100), # Posição (x, y)
        font, 
        font_scale, 
        text_color, 
        font_thickness
    )

    # Mostra a imagem final
    cv2.imshow('Tradutor de Libras em Tempo Real', image_output)

    # Pressione 'q' para sair
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libera os recursos
hands.close()
cap.release()
cv2.destroyAllWindows()