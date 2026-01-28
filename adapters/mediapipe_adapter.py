import cv2
import numpy as np
import logging
from typing import Tuple, Any
from core.domain import MaoDetectada
import config

# Tentar importar MediaPipe com API nova ou antiga
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_NEW_API = True
except ImportError:
    import mediapipe as mp
    MEDIAPIPE_NEW_API = False

logger = logging.getLogger(__name__)

class MediaPipeAdapter:
    """Adaptador para detecção de mãos usando MediaPipe."""
    
    def __init__(self):
        """Inicializa o detector MediaPipe com configurações do config.py."""
        try:
            if MEDIAPIPE_NEW_API:
                self._init_new_api()
            else:
                self._init_old_api()
            logger.info("MediaPipe inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar MediaPipe: {e}")
            raise
    
    def _init_new_api(self):
        """Inicializa usando API nova do MediaPipe (0.10.30+)"""
        # Criar modelo básico sem arquivo (usa modelo interno)
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=self._get_default_model()),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.detector = HandLandmarker.create_from_options(options)
        self.new_api = True
        
    def _init_old_api(self):
        """Inicializa usando API antiga do MediaPipe (<0.10.30)"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.new_api = False
    
    def _get_default_model(self):
        """Baixa e retorna o modelo padrão do MediaPipe"""
        import urllib.request
        import os
        
        model_path = os.path.join(os.getcwd(), 'hand_landmarker.task')
        
        if not os.path.exists(model_path):
            logger.info("Baixando modelo do MediaPipe...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            logger.info("Modelo baixado com sucesso")
        
        with open(model_path, 'rb') as f:
            return f.read()

    def processar(self, imagem) -> Tuple[MaoDetectada, Any]:
        """Processa uma imagem e detecta landmarks da mão.
        
        Args:
            imagem: Frame da câmera em formato BGR.
            
        Returns:
            Tupla contendo (MaoDetectada, resultados brutos do MediaPipe).
        """
        try:
            if self.new_api:
                return self._processar_new_api(imagem)
            else:
                return self._processar_old_api(imagem)
        except Exception as e:
            logger.error(f"Erro ao processar imagem: {e}")
            return MaoDetectada(pontos=[]), None
    
    def _processar_new_api(self, imagem):
        """Processa com API nova"""
        rgb_frame = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)
        
        pontos_flat = []
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            for landmark in landmarks:
                pontos_flat.extend([landmark.x, landmark.y, landmark.z])
                
        return MaoDetectada(pontos=pontos_flat), results
    
    def _processar_old_api(self, imagem):
        """Processa com API antiga"""
        img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        pontos_flat = []
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            for point in landmarks.landmark:
                pontos_flat.extend([point.x, point.y, point.z])
                
        return MaoDetectada(pontos=pontos_flat), results

    def desenhar(self, imagem, results):
        """Desenha os landmarks da mão na imagem.
        
        Args:
            imagem: Frame onde desenhar.
            results: Resultados do MediaPipe.
        """
        try:
            if self.new_api:
                self._desenhar_new_api(imagem, results)
            else:
                self._desenhar_old_api(imagem, results)
        except Exception as e:
            logger.error(f"Erro ao desenhar landmarks: {e}")
    
    def _desenhar_new_api(self, imagem, results):
        """Desenha com API nova"""
        if not results or not results.hand_landmarks:
            return
            
        altura, largura, _ = imagem.shape
        
        # Conexões da mão
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
            (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
            (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
            (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
            (0, 17), (17, 18), (18, 19), (19, 20),  # Mínimo
            (5, 9), (9, 13), (13, 17)  # Palma
        ]
        
        landmarks = results.hand_landmarks[0]
        
        # Desenhar conexões
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_point = (int(start.x * largura), int(start.y * altura))
            end_point = (int(end.x * largura), int(end.y * altura))
            
            cv2.line(imagem, start_point, end_point, (0, 255, 0), 2)
        
        # Desenhar landmarks
        for landmark in landmarks:
            x = int(landmark.x * largura)
            y = int(landmark.y * altura)
            cv2.circle(imagem, (x, y), 5, (0, 0, 255), -1)
    
    def _desenhar_old_api(self, imagem, results):
        """Desenha com API antiga"""
        if results and results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                imagem, 
                results.multi_hand_landmarks[0], 
                self.mp_hands.HAND_CONNECTIONS
            )
    
    def __del__(self):
        """Libera recursos do MediaPipe."""
        try:
            if hasattr(self, 'detector'):
                self.detector.close()
            elif hasattr(self, 'hands'):
                self.hands.close()
        except:
            pass