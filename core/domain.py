from dataclasses import dataclass
from typing import List

@dataclass
class Gesto:
    """Representa um gesto classificado.
    
    Attributes:
        nome: Letra ou símbolo reconhecido (A-Z).
        confianca: Nível de confiança da predição (0.0 a 1.0).
    """
    nome: str
    confianca: float = 0.0

@dataclass
class MaoDetectada:
    """Representa uma mão detectada com seus landmarks.
    
    Attributes:
        pontos: Lista de coordenadas (x, y, z) dos 21 pontos da mão.
                Total de 63 valores (21 pontos × 3 coordenadas).
    """
    pontos: List[float] 
    
    def normalizar(self) -> List[float]:
        """Normaliza os pontos da mão para invariância de posição.
        
        A normalização subtrai as coordenadas mínimas (x, y) para que
        o gesto seja reconhecido independentemente da posição na tela.
        
        Returns:
            Lista normalizada de coordenadas [x0, y0, z0, x1, y1, z1, ...].
            Lista vazia se não houver pontos.
        """
        if not self.pontos:
            return []
        
        # Reconstrói estrutura (x, y, z) para calcular min
        pontos_xyz = [self.pontos[i:i+3] for i in range(0, len(self.pontos), 3)]
        x_coords = [p[0] for p in pontos_xyz]
        y_coords = [p[1] for p in pontos_xyz]
        
        min_x, min_y = min(x_coords), min(y_coords)
        
        dados_normalizados = []
        for p in pontos_xyz:
            dados_normalizados.extend([p[0] - min_x, p[1] - min_y, p[2]])
            
        return dados_normalizados