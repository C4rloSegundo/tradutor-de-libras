from dataclasses import dataclass
from typing import List

@dataclass
class Gesto:
    nome: str
    confianca: float = 0.0

@dataclass
class MaoDetectada:
    # Representa os dados puros da mão (21 pontos x, y, z)
    pontos: List[float] 
    
    def normalizar(self) -> List[float]:
        """Regra de negócio: Normalização é intrínseca ao domínio para garantir invariância."""
        # Lógica de subtrair min_x e min_y vem pra cá
        if not self.pontos: return []
        
        # Reconstrói estrutura (x, y, z) para calcular min
        pontos_xyz = [self.pontos[i:i+3] for i in range(0, len(self.pontos), 3)]
        x_coords = [p[0] for p in pontos_xyz]
        y_coords = [p[1] for p in pontos_xyz]
        
        min_x, min_y = min(x_coords), min(y_coords)
        
        dados_normalizados = []
        for p in pontos_xyz:
            dados_normalizados.extend([p[0] - min_x, p[1] - min_y, p[2]])
            
        return dados_normalizados