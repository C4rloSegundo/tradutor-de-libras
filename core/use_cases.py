from .domain import Gesto, MaoDetectada

# Interface (Contrato) para o classificador.
# O Caso de Uso não quer saber se é KNN ou Rede Neural, só quer o método `classificar`.
class IClassificador:
    def classificar(self, mao: MaoDetectada) -> Gesto:
        pass

class ReconhecerSinalUseCase:
    def __init__(self, classificador: IClassificador):
        self.classificador = classificador
        self.buffer_letras = []
        self.LIMITE_BUFFER = 30
        self.letra_atual = ""
        self.contagem = 0

    def executar(self, mao: MaoDetectada) -> dict:
        # 1. Normaliza
        if not mao.pontos:
            return {"letra_detectada": "?", "frase": "", "progresso": 0}

        # 2. Classifica (usando a interface, sem saber que é sklearn)
        gesto = self.classificador.classificar(mao)
        
        # 3. Lógica de Estabilização (Buffer)
        frase_atualizada = self._processar_buffer(gesto.nome)
        
        return {
            "letra_detectada": gesto.nome,
            "frase": frase_atualizada,
            "progresso": self.contagem
        }

    def _processar_buffer(self, letra_nova):
        # (Sua lógica de contagem de frames vem pra cá)
        if letra_nova == self.letra_atual:
            self.contagem += 1
        else:
            self.letra_atual = letra_nova
            self.contagem = 0
            
        if self.contagem >= self.LIMITE_BUFFER:
            self.buffer_letras.append(self.letra_atual)
            self.contagem = 0
            
        return "".join(self.buffer_letras)
    
    def adicionar_espaco(self):
        """Adiciona um espaço à frase."""
        if self.buffer_letras and self.buffer_letras[-1] != " ":
            self.buffer_letras.append(" ")
    
    def limpar_frase(self):
        """Limpa toda a frase."""
        self.buffer_letras = []
        self.contagem = 0
        self.letra_atual = ""