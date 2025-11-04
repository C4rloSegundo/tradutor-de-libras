import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configura o detector de mãos
# max_num_hands=1 para focar em apenas uma mão (mais simples para começar)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inicia a captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    # Inverte a imagem horizontalmente (efeito espelho)
    image = cv2.flip(image, 1)
    
    # Converte a imagem de BGR (OpenCV) para RGB (MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Processa a imagem e detecta as mãos
    results = hands.process(image_rgb)

    # Limpa a imagem (cria uma cópia limpa)
    image_output = image.copy()

    # Desenha os marcos da mão se uma mão for detectada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_output, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS) # Desenha as conexões

    # Mostra a imagem resultante
    cv2.imshow('Tradutor de Libras - MediaPipe', image_output)

    # Pressione 'q' para sair
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libera os recursos
hands.close()
cap.release()
cv2.destroyAllWindows()