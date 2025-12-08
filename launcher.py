import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import threading

class LauncherTCC:
    def __init__(self, root):
        self.root = root
        self.root.title("Painel de Controle - TCC Libras")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        
        # Estilização
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12), padding=10)
        style.configure('TLabel', font=('Helvetica', 10))

        # Título
        lbl_titulo = tk.Label(root, text="Sistema de Reconhecimento\nde Alfabeto Manual (Libras)", 
                              font=("Helvetica", 16, "bold"), pady=20)
        lbl_titulo.pack()

        # Frame dos Botões
        frame_botoes = ttk.Frame(root, padding=20)
        frame_botoes.pack(fill='both', expand=True)

        # Botão 1: Coleta
        self.btn_coleta = ttk.Button(frame_botoes, text="1. Coletar Novos Dados", 
                                     command=lambda: self.rodar_script("coletor_dados.py", esperar=False))
        self.btn_coleta.pack(fill='x', pady=5)
        
        lbl_desc1 = ttk.Label(frame_botoes, text="Capture imagens da webcam para criar o Dataset.", foreground="gray")
        lbl_desc1.pack(pady=(0, 15))

        # Botão 2: Treinamento
        self.btn_treino = ttk.Button(frame_botoes, text="2. Treinar Modelo IA", 
                                     command=self.iniciar_treinamento)
        self.btn_treino.pack(fill='x', pady=5)
        
        lbl_desc2 = ttk.Label(frame_botoes, text="Gera o arquivo .joblib e os gráficos de Acurácia.", foreground="gray")
        lbl_desc2.pack(pady=(0, 15))

        # Botão 3: Principal
        self.btn_main = ttk.Button(frame_botoes, text="3. INICIAR SISTEMA (Main)", 
                                   command=lambda: self.rodar_script("main.py", esperar=False))
        self.btn_main.pack(fill='x', pady=5)
        
        lbl_desc3 = ttk.Label(frame_botoes, text="Executa a aplicação final com a arquitetura Clean Arch.", foreground="gray")
        lbl_desc3.pack(pady=(0, 15))

        # Área de Status/Log
        self.lbl_status = tk.Label(root, text="Status: Aguardando comando...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def rodar_script(self, script_name, esperar=False):
        """Roda scripts externos usando o mesmo interpretador Python atual"""
        self.lbl_status.config(text=f"Executando: {script_name}...", fg="blue")
        
        try:
            # sys.executable garante que usamos o python do venv
            if esperar:
                # Se for esperar (treinamento), roda e captura o output
                processo = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
                return processo
            else:
                # Se não for esperar (coletor/main), abre em paralelo
                subprocess.Popen([sys.executable, script_name])
                self.lbl_status.config(text=f"Rodando {script_name} em janela separada.", fg="green")
        
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível iniciar {script_name}.\nErro: {e}")
            self.lbl_status.config(text="Erro na execução.", fg="red")

    def iniciar_treinamento(self):
        """Roda o treinamento em uma thread separada para não travar a janela"""
        self.btn_treino.config(state="disabled")
        self.lbl_status.config(text="Treinando modelo... Isso pode levar alguns segundos.", fg="orange")
        
        def _thread_target():
            resultado = self.rodar_script("treinar_modelo.py", esperar=True)
            
            # Atualiza GUI na thread principal
            self.root.after(0, lambda: self._pos_treinamento(resultado))

        threading.Thread(target=_thread_target, daemon=True).start()

    def _pos_treinamento(self, resultado):
        self.btn_treino.config(state="normal")
        if resultado.returncode == 0:
            messagebox.showinfo("Sucesso", "Treinamento Concluído!\n\nVeja o arquivo 'matriz_confusao.png'.")
            self.lbl_status.config(text="Treinamento finalizado com sucesso.", fg="green")
            print(resultado.stdout) # Imprime no terminal se quiser ver detalhes
        else:
            messagebox.showerror("Erro no Treinamento", f"Ocorreu um erro:\n{resultado.stderr}")
            self.lbl_status.config(text="Falha no treinamento.", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherTCC(root)
    root.mainloop()