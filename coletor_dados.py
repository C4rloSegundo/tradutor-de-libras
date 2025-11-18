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


# Configurações do diretório e labels
DATA_DIR = 'dados_letras'
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', '/'] # <-- ADICIONE MAIS LETRAS/SINAIS AQUI

print(f"Pressione as teclas: {', '.join(LABELS)} para selecionar a letra.")
print("Pressione '0' para sair.")
print("Os dados serão coletados continuamente para a letra selecionada.")

# Função para garantir que o arquivo da letra existe e tem cabeçalho
def garantir_arquivo_letra(letra):
    file_path = os.path.join(DATA_DIR, f'{letra}.csv')
    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label']
            for i in range(21):
                header += [f'x{i}', f'y{i}', f'z{i}']
            writer.writerow(header)
    return file_path

# Inicia a captura de vídeo
cap = cv2.VideoCapture(0)

selected_label = None

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

    key = cv2.waitKey(5) & 0xFF

    # Sair com '0'
    if key == ord('0'):
        break

    # Seleciona a letra se pressionada
    if key != 255:  # 255 = nenhuma tecla pressionada
        key_char = chr(key).upper()
        if key_char in LABELS:
            selected_label = key_char
            print(f'Letra selecionada: {selected_label}')

    # Coleta dados continuamente para a letra selecionada
    if selected_label and hand_landmarks:
        try:
            # 1. Pega todos os pontos
            points = hand_landmarks.landmark
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
                data_row.append(point.z)  # Adiciona Z sem normalizar (MediaPipe já normaliza)

            # 4. Adiciona o rótulo (label) no início
            data_row.insert(0, selected_label)

            # 5. Salva a linha no arquivo CSV da letra
            file_path = garantir_arquivo_letra(selected_label)
            with open(file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)

            print(f"Salvo! Dados para a letra: {selected_label} em {file_path}")

        except Exception as e:
            print(f"Erro ao processar landmarks: {e}")

# Libera os recursos
hands.close()
cap.release()
cv2.destroyAllWindows()