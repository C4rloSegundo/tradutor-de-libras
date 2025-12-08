import joblib
import numpy as np
from core.use_cases import IClassificador
from core.domain import MaoDetectada, Gesto

class SKLearnAdapter(IClassificador):
    def __init__(self, caminho_modelo):
        self.model = joblib.load(caminho_modelo)

    def classificar(self, mao: MaoDetectada) -> Gesto:
        # Pega os dados normalizados da entidade
        dados = mao.normalizar()
        # O modelo espera array 2D
        dados_np = np.array([dados])
        # Predição
        resultado = self.model.predict(dados_np)[0]
        return Gesto(nome=str(resultado))