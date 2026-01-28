from .domain import Gesto, MaoDetectada
from typing import Dict
import logging
import config

logger = logging.getLogger(__name__)

class IClassificador:
    """Interface para classificadores de gestos.
    
    Qualquer implementação de classificador (KNN, SVM, Rede Neural, etc.)
    deve implementar este contrato.
    """
    
    def classificar(self, mao: MaoDetectada) -> Gesto:
        """Classifica um gesto de mão.
        
        Args:
            mao: Objeto MaoDetectada contendo os landmarks.
            
        Returns:
            Gesto classificado com nome e confiança.
        """
        pass

class ReconhecerSinalUseCase:
    """Caso de uso principal para reconhecimento de sinais de Libras.
    
    Implementa a lógica de negócio para:
    - Estabilização de predições (buffer)
    - Filtragem por confiança
    - Construção de frases
    """
    
    def __init__(self, classificador: IClassificador):
        """Inicializa o caso de uso.
        
        Args:
            classificador: Implementação de IClassificador a ser usada.
        """
        self.classificador = classificador
        self.buffer_letras = []
        self.LIMITE_BUFFER = config.LIMITE_BUFFER
        self.CONFIANCA_MINIMA = config.CONFIANCA_MINIMA
        self.letra_atual = ""
        self.contagem = 0
        logger.info(f"ReconhecerSinalUseCase inicializado (buffer={self.LIMITE_BUFFER})")

    def executar(self, mao: MaoDetectada) -> Dict[str, any]:
        """Executa o reconhecimento de sinal.
        
        Args:
            mao: Objeto MaoDetectada com os landmarks.
            
        Returns:
            Dicionário com letra_detectada, frase, progresso e confianca.
        """
        # Normaliza
        if not mao.pontos:
            return {
                "letra_detectada": "?",
                "frase": "".join(self.buffer_letras),
                "progresso": 0,
                "confianca": 0.0
            }

        # Classifica
        gesto = self.classificador.classificar(mao)
        
        # Filtra por confiança mínima
        letra_valida = gesto.nome if gesto.confianca >= self.CONFIANCA_MINIMA else "?"
        
        # Lógica de Estabilização (Buffer)
        frase_atualizada = self._processar_buffer(letra_valida)
        
        return {
            "letra_detectada": letra_valida,
            "frase": frase_atualizada,
            "progresso": self.contagem,
            "confianca": gesto.confianca
        }

    def _processar_buffer(self, letra_nova: str) -> str:
        """Processa o buffer de estabilização de letras.
        
        Args:
            letra_nova: Letra detectada no frame atual.
            
        Returns:
            Frase completa construída até o momento.
        """
        if letra_nova == self.letra_atual:
            self.contagem += 1
        else:
            self.letra_atual = letra_nova
            self.contagem = 0
            
        if self.contagem >= self.LIMITE_BUFFER:
            if self.letra_atual != "?":  # Não adiciona interrogações
                self.buffer_letras.append(self.letra_atual)
                logger.info(f"Letra adicionada: {self.letra_atual}")
            self.contagem = 0
            
        return "".join(self.buffer_letras)
    
    def limpar_frase(self):
        """Limpa a frase atual e reseta o buffer."""
        self.buffer_letras = []
        self.contagem = 0
        self.letra_atual = ""
        logger.info("Frase limpa")
    
    def adicionar_espaco(self):
        """Adiciona um espaço à frase."""
        if self.buffer_letras and self.buffer_letras[-1] != " ":
            self.buffer_letras.append(" ")
            logger.info("Espaço adicionado")