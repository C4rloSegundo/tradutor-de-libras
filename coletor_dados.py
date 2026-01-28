import cv2
import csv
import os
import time
from adapters.mediapipe_adapter import MediaPipeAdapter

# Configurações
DATA_DIR = 'dados_letras'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def garantir_arquivo_letra(letra):
    file_path = os.path.join(DATA_DIR, f'{letra}.csv')
    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label']
            for i in range(21):
                header += [f'x{i}', f'y{i}', f'z{i}']
            writer.writerow(header)
    return file_path

def main():
    print("--- INICIANDO COLETOR DE DADOS ---")
    print("1. Carregando MediaPipe Adapter...")
    detector = MediaPipeAdapter()
    
    print("2. Tentando acessar a Webcam (Indice 0)...")
    cap = cv2.VideoCapture(0)

    # Verifica se a câmera abriu mesmo
    if not cap.isOpened():
        print("ERRO CRÍTICO: Não foi possível abrir a câmera!")
        print("Tentativa: Verifique se outra aplicação está usando a webcam.")
        print("Tentativa: Tente mudar cv2.VideoCapture(0) para cv2.VideoCapture(1) no código.")
        return

    selected_label = None
    print(f"SUCESSO: Câmera aberta!")
    print(f"Comandos: Teclas A-Y para gravar, ESPAÇO para pausar, 0 para sair.")

    while cap.isOpened():
        success, image = cap.read()
        
        # Correção do Bug do Loop Infinito:
        if not success:
            print("AVISO: Falha ao ler quadro da câmera (frame vazio).")
            continue

        image = cv2.flip(image, 1)

        try:
            mao_detectada, raw_results = detector.processar(image)
            detector.desenhar(image, raw_results)
        except Exception as e:
            print(f"Erro no processamento: {e}")

        # Interface
        status_text = f"Gravando: {selected_label}" if selected_label else "PAUSADO (Selecione uma letra)"
        color = (0, 0, 255) if selected_label else (255, 0, 0)
        
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('Coletor Clean Arch', image)

        # Captura de Teclado
        key = cv2.waitKey(5) & 0xFF
        if key == ord('0'): 
            break
        elif key != 255:
            char = chr(key).upper()
            if char in LABELS:
                selected_label = char
                print(f"--> GRAVANDO: Letra {selected_label}")
            elif key == ord(' '): 
                selected_label = None
                print("--> PAUSADO")

        if selected_label and mao_detectada.pontos:
            try:
                dados_normalizados = mao_detectada.normalizar()
                linha_csv = [selected_label] + dados_normalizados
                
                file_path = garantir_arquivo_letra(selected_label)
                with open(file_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(linha_csv)
                
                # Feedback visual (bolinha verde piscando)
                cv2.circle(image, (30, 60), 10, (0, 255, 0), -1)
                
            except Exception as e:
                print(f"Erro ao salvar: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("Coletor finalizado.")

if __name__ == "__main__":
    main()