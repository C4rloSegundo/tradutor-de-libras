"""
Configurações centralizadas do Sistema de Reconhecimento de Libras
"""
import os

# ==================== CAMINHOS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dados_letras')
MODEL_FILE = os.path.join(BASE_DIR, 'hand_model.joblib')
CONFUSION_MATRIX_FILE = os.path.join(BASE_DIR, 'matriz_confusao.png')
LOG_FILE = os.path.join(BASE_DIR, 'tradutor_libras.log')

# ==================== MODELO ====================
# Alfabeto de Libras suportado
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Configurações do classificador KNN
KNN_NEIGHBORS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==================== MEDIAPIPE ====================
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ==================== RECONHECIMENTO ====================
# Quantos frames consecutivos da mesma letra antes de adicionar ao texto
LIMITE_BUFFER = 30  # ~1 segundo a 30fps

# Confiança mínima para considerar predição válida (0.0 a 1.0)
CONFIANCA_MINIMA = 0.6

# Otimização: processar 1 a cada N frames
FRAME_SKIP = 2  # Processar 1 a cada 2 frames

# ==================== INTERFACE ====================
# Configurações da janela
WINDOW_TITLE = "Tradutor de Libras"
CAMERA_INDEX = 0

# Cores (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GRAY = (128, 128, 128)

# Interface do Launcher
LAUNCHER_TITLE = "Tradutor de Libras - Central de Controle"
LAUNCHER_WIDTH = 550
LAUNCHER_HEIGHT = 500
