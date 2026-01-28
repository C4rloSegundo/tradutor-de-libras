import cv2
import csv
import os
import logging
from adapters.mediapipe_adapter import MediaPipeAdapter
import config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Criar diretório de dados se não existir
if not os.path.exists(config.DATA_DIR):
    os.makedirs(config.DATA_DIR)
    logger.info(f"Diretório criado: {config.DATA_DIR}")

def garantir_arquivo_letra(letra):
    """Garante que o arquivo CSV da letra existe com header correto.
    
    Args:
        letra: Letra do alfabeto (A-Z).
        
    Returns:
        Caminho completo do arquivo CSV.
    """
    file_path = os.path.join(config.DATA_DIR, f'{letra}.csv')
    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label']
            for i in range(21):
                header += [f'x{i}', f'y{i}', f'z{i}']
            writer.writerow(header)
        logger.info(f"Arquivo criado: {file_path}")
    return file_path

def main():
    """Função principal do coletor de dados."""
    logger.info("=" * 50)
    logger.info("INICIANDO COLETOR DE DADOS DE LIBRAS")
    logger.info("=" * 50)
    
    try:
        logger.info("Carregando MediaPipe Adapter...")
        detector = MediaPipeAdapter()
        
        logger.info(f"Tentando acessar câmera (index {config.CAMERA_INDEX})...")
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not cap.isOpened():
            logger.error("Não foi possível abrir a câmera")
            print("\nERRO CRÍTICO: Não foi possível abrir a câmera!")
            print("Soluções:")
            print("  1. Verifique se outra aplicação está usando a webcam")
            print("  2. Tente mudar CAMERA_INDEX no config.py")
            return

        selected_label = None
        amostras_gravadas = {letra: 0 for letra in config.LABELS}
        
        logger.info("Câmera aberta com sucesso!")
        print("\n" + "=" * 50)
        print("COLETOR DE DADOS - Alfabeto de Libras")
        print("=" * 50)
        print(f"Letras disponíveis: {', '.join(config.LABELS)}")
        print("\nComandos:")
        print("  [A-Y]    - Iniciar gravação da letra")
        print("  [ESPAÇO] - Pausar gravação")
        print("  [0]      - Sair e salvar")
        print("=" * 50 + "\n")

        while cap.isOpened():
            success, image = cap.read()
            
            if not success:
                logger.warning("Falha ao ler frame da câmera")
                continue

            image = cv2.flip(image, 1)

            try:
                mao_detectada, raw_results = detector.processar(image)
                detector.desenhar(image, raw_results)
            except Exception as e:
                logger.error(f"Erro no processamento: {e}")

            # Interface
            altura, largura = image.shape[:2]
            cv2.rectangle(image, (0, 0), (largura, 80), (50, 50, 50), -1)
            
            if selected_label:
                cv2.putText(image, f"GRAVANDO: {selected_label}", (10, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.COLOR_RED, 3)
                cv2.putText(image, f"Amostras: {amostras_gravadas[selected_label]}", (10, 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
            else:
                cv2.putText(image, "PAUSADO - Pressione uma letra", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_BLUE, 2)
            
            cv2.imshow('Coletor de Dados - Libras', image)

            # Captura de Teclado
            key = cv2.waitKey(5) & 0xFF
            if key == ord('0'): 
                logger.info("Encerrando por comando do usuário")
                break
            elif key != 255:
                char = chr(key).upper()
                if char in config.LABELS:
                    selected_label = char
                    logger.info(f"Gravando letra: {selected_label}")
                    print(f"\n> GRAVANDO: {selected_label}")
                elif key == ord(' '): 
                    selected_label = None
                    logger.info("Gravação pausada")
                    print("\n> PAUSADO")

            # Salvar dados
            if selected_label and mao_detectada.pontos:
                try:
                    dados_normalizados = mao_detectada.normalizar()
                    linha_csv = [selected_label] + dados_normalizados
                    
                    file_path = garantir_arquivo_letra(selected_label)
                    with open(file_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(linha_csv)
                    
                    amostras_gravadas[selected_label] += 1
                    
                    # Feedback visual
                    cv2.circle(image, (largura - 30, 40), 15, config.COLOR_GREEN, -1)
                    
                except Exception as e:
                    logger.error(f"Erro ao salvar amostra: {e}")

        cap.release()
        cv2.destroyAllWindows()
        
        # Relatório final
        print("\n" + "=" * 50)
        print("RELATÓRIO DE COLETA")
        print("=" * 50)
        total = 0
        for letra, qtd in sorted(amostras_gravadas.items()):
            if qtd > 0:
                print(f"  {letra}: {qtd} amostras")
                total += qtd
        print(f"\nTotal: {total} amostras coletadas")
        print("=" * 50)
        logger.info(f"Coleta finalizada. Total: {total} amostras")
        
    except Exception as e:
        logger.error(f"Erro fatal no coletor: {e}", exc_info=True)
        print(f"\nERRO FATAL: {e}")

if __name__ == "__main__":
    main()