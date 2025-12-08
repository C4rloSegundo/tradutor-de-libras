import cv2
from adapters.mediapipe_adapter import MediaPipeAdapter
from adapters.sklearn_adapter import SKLearnAdapter
from core.use_cases import ReconhecerSinalUseCase

def main():
    # Inicializar Infraestrutura e Adaptadores
    video = cv2.VideoCapture(0)
    detector = MediaPipeAdapter()
    
    # Injetamos o adaptador concreto (SKLearn) onde o caso de uso espera a interface
    classificador = SKLearnAdapter('hand_model.joblib') 
    
    # Inicializar Caso de Uso
    use_case = ReconhecerSinalUseCase(classificador)

    while video.isOpened():
        ok, frame = video.read()
        if not ok: break
        
        frame = cv2.flip(frame, 1)

        # Captura e Adaptação (Infra -> Domain)
        mao_detectada, raw_results = detector.processar(frame)

        # Execução da Regra de Negócio (Core)
        resultado = use_case.executar(mao_detectada)

        # Apresentação (Output)
        detector.desenhar(frame, raw_results)
        
        # Desenha interface simples
        cv2.putText(frame, f"Letra: {resultado['letra_detectada']}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Texto: {resultado['frase']}", (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Barra de progresso visual
        cv2.rectangle(frame, (10, 60), (10 + (resultado['progresso'] * 10), 70), (0, 255, 0), -1)

        cv2.imshow("Clean Architecture TCC", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()