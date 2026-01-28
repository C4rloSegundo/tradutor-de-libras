import cv2
import logging
import config
from adapters.mediapipe_adapter import MediaPipeAdapter
from adapters.sklearn_adapter import SKLearnAdapter
from core.use_cases import ReconhecerSinalUseCase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def desenhar_interface_profissional(frame, resultado, pausado=False):
    """Desenha interface profissional com todos os indicadores."""
    altura, largura = frame.shape[:2]
    
    # Painel superior - Letra detectada
    cv2.rectangle(frame, (0, 0), (largura, 100), (50, 50, 50), -1)
    
    # Status (pausado ou ativo)
    if pausado:
        cv2.putText(frame, "PAUSADO", (largura - 150, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
    else:
        cv2.putText(frame, "ATIVO", (largura - 120, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_GREEN, 2)
    
    # Letra atual (grande)
    letra = resultado['letra_detectada']
    cor_letra = config.COLOR_GREEN if letra != "?" else config.COLOR_RED
    cv2.putText(frame, f"Letra: {letra}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, cor_letra, 3)
    
    # Confiança
    confianca = resultado.get('confianca', 0.0)
    cor_confianca = config.COLOR_GREEN if confianca >= config.CONFIANCA_MINIMA else config.COLOR_RED
    cv2.putText(frame, f"Confianca: {confianca*100:.0f}%", (250, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_confianca, 2)
    
    # Barra de progresso
    progresso = resultado['progresso']
    largura_barra = int((progresso / config.LIMITE_BUFFER) * 300)
    cv2.rectangle(frame, (250, 55), (550, 75), config.COLOR_GRAY, -1)
    cv2.rectangle(frame, (250, 55), (250 + largura_barra, 75), config.COLOR_GREEN, -1)
    cv2.putText(frame, f"{progresso}/{config.LIMITE_BUFFER}", (255, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1)
    
    # Painel inferior - Frase
    cv2.rectangle(frame, (0, altura - 80), (largura, altura), (50, 50, 50), -1)
    frase = resultado['frase'] if resultado['frase'] else "[Nenhum texto ainda]"
    cv2.putText(frame, f"Texto: {frase}", (10, altura - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_WHITE, 2)
    
    # Instruções
    cv2.putText(frame, "Comandos: [ESPAÇO] Pausar | [C] Limpar | [B] Espaco | [Q] Sair", 
                (10, altura - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_YELLOW, 1)

def main():
    """Função principal do tradutor de Libras."""
    try:
        logger.info("Iniciando Tradutor de Libras...")
        
        # Inicializar Infraestrutura e Adaptadores
        video = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not video.isOpened():
            logger.error("Não foi possível abrir a câmera")
            print("ERRO: Câmera não encontrada. Verifique se está conectada.")
            return
        
        logger.info("Câmera inicializada com sucesso")
        
        detector = MediaPipeAdapter()
        classificador = SKLearnAdapter(config.MODEL_FILE)
        use_case = ReconhecerSinalUseCase(classificador)
        
        pausado = False
        frame_count = 0
        
        logger.info("Sistema pronto. Pressione 'Q' para sair.")
        
        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                logger.warning("Falha ao ler frame da câmera")
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Otimização: processar apenas alguns frames
            if not pausado and frame_count % config.FRAME_SKIP == 0:
                # Captura e Adaptação (Infra -> Domain)
                mao_detectada, raw_results = detector.processar(frame)
                
                # Execução da Regra de Negócio (Core)
                resultado = use_case.executar(mao_detectada)
                
                # Apresentação (Output)
                detector.desenhar(frame, raw_results)
            else:
                # Quando pausado, ainda mostra a última detecção
                resultado = {
                    'letra_detectada': '?',
                    'frase': ''.join(use_case.buffer_letras),
                    'progresso': 0,
                    'confianca': 0.0
                }
            
            # Desenha interface profissional
            desenhar_interface_profissional(frame, resultado, pausado)
            
            cv2.imshow(config.WINDOW_TITLE, frame)
            
            # Comandos do teclado
            key = cv2.waitKey(5) & 0xFF
            
            if key == ord('q'):
                logger.info("Encerrando por comando do usuário")
                break
            elif key == ord(' '):  # ESPAÇO - Pausar/Continuar
                pausado = not pausado
                status = "pausado" if pausado else "retomado"
                logger.info(f"Sistema {status}")
            elif key == ord('c'):  # C - Limpar texto
                use_case.limpar_frase()
            elif key == ord('b'):  # B - Adicionar espaço (Backspace seria melhor mas pode conflitar)
                use_case.adicionar_espaco()
        
        video.release()
        cv2.destroyAllWindows()
        logger.info("Tradutor encerrado com sucesso")
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        print(f"\nERRO: {e}")
        print("Solução: Execute 'python treinar_modelo.py' primeiro!")
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        print(f"\nERRO FATAL: {e}")

if __name__ == "__main__":
    main()