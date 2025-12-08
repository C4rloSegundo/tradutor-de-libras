import mediapipe as mp
import cv2
from core.domain import MaoDetectada

class MediaPipeAdapter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

    def processar(self, imagem) -> MaoDetectada:
        # Converte imagem
        img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        pontos_flat = []
        if results.multi_hand_landmarks:
            # Pega a primeira mão
            landmarks = results.multi_hand_landmarks[0]
            # Extrai apenas os números (x, y, z)
            for point in landmarks.landmark:
                pontos_flat.extend([point.x, point.y, point.z])
                
        return MaoDetectada(pontos=pontos_flat), results

    def desenhar(self, imagem, results):
        if results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                imagem, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS
            )