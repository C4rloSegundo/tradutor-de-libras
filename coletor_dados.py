import cv2
import mediapipe as mp
import numpy as np
import csv
import os # Usaremos para verificar se o arquivo já existe

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,              # Focar em apenas uma mão
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Configurações do arquivo de dados
DATA_FILE = 'hand_data.csv'
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '/'] # <-- ADICIONE MAIS LETRAS/SINAIS AQUI
print(f"Pressione as teclas: {', '.join(LABELS)} para salvar os dados.")
print("Pressione '0' para sair.")

# Cria o arquivo CSV e o cabeçalho se ele não existir
file_exists = os.path.isfile(DATA_FILE)
with open(DATA_FILE, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        # Cria o cabeçalho (label + 42 features: x0, y0, x1, y1, ...)
        header = ['label']
        for i in range(21):
            header += [f'x{i}', f'y{i}']
        writer.writerow(header)

# Inicia a captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image_output = image.copy()

    hand_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            image_output, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS
        )

    # Mostra a imagem
    cv2.imshow('Coletor de Dados - MediaPipe', image_output)

    # --- Lógica de Coleta ---
    key = cv2.waitKey(5) & 0xFF
    
    # Sair com 'q'
    if key == ord('0'):
        break
        
    # Converte a tecla pressionada para maiúscula
    key_char = chr(key).upper()

    # Se a tecla pressionada for uma das nossas LABELS e uma mão for detectada
    if key_char in LABELS and hand_landmarks:
        try:
            # --- Normalização dos Dados ---
            # O MediaPipe já fornece coordenadas (x,y) normalizadas [0.0, 1.0]
            # Vamos torná-las relativas à própria mão (ao invés da tela)
            
            # 1. Pega todos os pontos
            points = hand_landmarks.landmark
            
            # 2. Encontra o ponto (x,y) mínimo (o canto "superior esquerdo" da mão)
            x_coords = [point.x for point in points]
            y_coords = [point.y for point in points]
            min_x = min(x_coords)
            min_y = min(y_coords)

            # 3. Calcula a posição de todos os pontos RELATIVO a esse ponto mínimo
            # Isso torna o sinal independente da *posição* na tela
            data_row = []
            for point in points:
                data_row.append(point.x - min_x)
                data_row.append(point.y - min_y)
            
            # 4. Adiciona o rótulo (label) no início
            data_row.insert(0, key_char)

            # 5. Salva a linha no arquivo CSV
            with open(DATA_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
            
            print(f"Salvo! Dados para a letra: {key_char}")

        except Exception as e:
            print(f"Erro ao processar landmarks: {e}")

# Libera os recursos
hands.close()
cap.release()
cv2.destroyAllWindows()