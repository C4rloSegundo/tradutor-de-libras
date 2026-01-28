import joblib
import numpy as np
import pandas as pd
import os
import logging
from core.use_cases import IClassificador
from core.domain import MaoDetectada, Gesto

logger = logging.getLogger(__name__)

class SKLearnAdapter(IClassificador):
    """Adaptador para classificação de gestos usando scikit-learn."""
    
    def __init__(self, caminho_modelo):
        """Inicializa o adaptador carregando o modelo treinado.
        
        Args:
            caminho_modelo: Caminho para o arquivo .joblib do modelo.
            
        Raises:
            FileNotFoundError: Se o modelo não for encontrado.
            Exception: Se houver erro ao carregar o modelo.
        """
        if not os.path.exists(caminho_modelo):
            logger.error(f"Modelo não encontrado: {caminho_modelo}")
            raise FileNotFoundError(
                f"Modelo não encontrado em '{caminho_modelo}'. "
                "Execute 'treinar_modelo.py' primeiro!"
            )
        
        try:
            self.model = joblib.load(caminho_modelo)
            
            # Criar nomes de features compatíveis com o treinamento
            self.feature_names = []
            for i in range(21):  # 21 landmarks da mão
                self.feature_names.extend([f'x{i}', f'y{i}', f'z{i}'])
            
            logger.info(f"Modelo carregado com sucesso: {caminho_modelo}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise Exception(f"Erro ao carregar modelo: {e}")

    def classificar(self, mao: MaoDetectada) -> Gesto:
        """Classifica um gesto de mão detectado.
        
        Args:
            mao: Objeto MaoDetectada contendo os pontos da mão.
            
        Returns:
            Gesto com a letra prevista e confiança.
        """
        try:
            # Pega os dados normalizados da entidade
            dados = mao.normalizar()
            
            if not dados:
                return Gesto(nome="?", confianca=0.0)
            
            # Converte para DataFrame com os mesmos nomes de features do treinamento
            dados_df = pd.DataFrame([dados], columns=self.feature_names)
            
            # Predição com probabilidade
            resultado = self.model.predict(dados_df)[0]
            
            # Obtém probabilidades se o modelo suportar
            confianca = 0.0
            if hasattr(self.model, 'predict_proba'):
                probabilidades = self.model.predict_proba(dados_df)[0]
                confianca = float(np.max(probabilidades))
            
            return Gesto(nome=str(resultado), confianca=confianca)
            
        except Exception as e:
            logger.error(f"Erro na classificação: {e}")
            return Gesto(nome="?", confianca=0.0)