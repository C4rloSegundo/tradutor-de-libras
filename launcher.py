import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import threading

class LauncherTCC:
    def __init__(self, root):
        self.root = root
        self.root.title("Tradutor de Libras - Central de Controle")
        self.root.geometry("550x500")
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')
        
        # Estilização
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=12)
        style.configure('TLabel', font=('Helvetica', 10), background='#f0f0f0')
        style.map('TButton', background=[('active', '#e0e0e0')])

        # Título
        frame_header = tk.Frame(root, bg='#2c3e50', height=80)
        frame_header.pack(fill='x')
        frame_header.pack_propagate(False)
        
        lbl_titulo = tk.Label(frame_header, text="Sistema de Reconhecimento\nde Alfabeto em Libras", 
                              font=("Helvetica", 18, "bold"), fg='white', bg='#2c3e50', pady=15)
        lbl_titulo.pack()

        # Frame dos Botões
        frame_botoes = tk.Frame(root, bg='#f0f0f0', padx=30, pady=30)
        frame_botoes.pack(fill='both', expand=True)

        # Botão 1: Coleta
        self.btn_coleta = ttk.Button(frame_botoes, text="📷  Capturar Novos Sinais", 
                                     command=lambda: self.rodar_script("coletor_dados.py", esperar=False))
        self.btn_coleta.pack(fill='x', pady=8)
        
        lbl_desc1 = tk.Label(frame_botoes, text="Use sua webcam para gravar novos sinais de letras", 
                            foreground="#555", bg='#f0f0f0', font=('Helvetica', 9))
        lbl_desc1.pack(pady=(0, 20))

        # Botão 2: Treinamento
        self.btn_treino = ttk.Button(frame_botoes, text="🧠  Treinar Inteligência Artificial", 
                                     command=self.iniciar_treinamento)
        self.btn_treino.pack(fill='x', pady=8)
        
        lbl_desc2 = tk.Label(frame_botoes, text="Ensine o sistema a reconhecer os sinais capturados", 
                            foreground="#555", bg='#f0f0f0', font=('Helvetica', 9))
        lbl_desc2.pack(pady=(0, 20))

        # Botão 3: Principal
        self.btn_main = ttk.Button(frame_botoes, text="🚀  INICIAR TRADUTOR", 
                                   command=lambda: self.rodar_script("main.py", esperar=False))
        self.btn_main.pack(fill='x', pady=8)
        
        lbl_desc3 = tk.Label(frame_botoes, text="Comece a traduzir sinais de Libras em tempo real", 
                            foreground="#555", bg='#f0f0f0', font=('Helvetica', 9))
        lbl_desc3.pack(pady=(0, 20))

        # Área de Status/Log
        self.lbl_status = tk.Label(root, text="✓ Pronto para uso", bd=1, relief=tk.SUNKEN, 
                                  anchor=tk.W, bg='white', fg='#27ae60', font=('Helvetica', 9))
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X, ipady=5)

    def rodar_script(self, script_name, esperar=False):
        """Roda scripts externos usando o mesmo interpretador Python atual"""
        nome_amigavel = {
            "coletor_dados.py": "Captura de Sinais",
            "treinar_modelo.py": "Treinamento",
            "main.py": "Tradutor"
        }
        self.lbl_status.config(text=f"⏳ Iniciando {nome_amigavel.get(script_name, script_name)}...", fg="#3498db")
        
        try:
            # sys.executable garante que usamos o python do venv
            if esperar:
                # Se for esperar (treinamento), roda e captura o output
                processo = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
                return processo
            else:
                # Se não for esperar (coletor/main), abre em paralelo
                subprocess.Popen([sys.executable, script_name])
                self.lbl_status.config(text=f"✓ {nome_amigavel.get(script_name, script_name)} em execução", fg="#27ae60")
        
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível iniciar a aplicação.\nErro: {e}")
            self.lbl_status.config(text="✗ Erro ao executar", fg="#e74c3c")

    def iniciar_treinamento(self):
        """Roda o treinamento em uma thread separada para não travar a janela"""
        self.btn_treino.config(state="disabled")
        self.lbl_status.config(text="🧠 Treinando... Aguarde alguns instantes", fg="#f39c12")
        
        def _thread_target():
            resultado = self.rodar_script("treinar_modelo.py", esperar=True)
            
            # Atualiza GUI na thread principal
            self.root.after(0, lambda: self._pos_treinamento(resultado))

        threading.Thread(target=_thread_target, daemon=True).start()

    def _pos_treinamento(self, resultado):
        self.btn_treino.config(state="normal")
        if resultado.returncode == 0:
            messagebox.showinfo("✓ Sucesso", "Treinamento concluído com sucesso!\n\nO sistema está pronto para uso.")
            self.lbl_status.config(text="✓ Treinamento concluído", fg="#27ae60")
            print(resultado.stdout) # Imprime no terminal se quiser ver detalhes
        else:
            messagebox.showerror("✗ Erro", f"Ocorreu um erro durante o treinamento:\n{resultado.stderr}")
            self.lbl_status.config(text="✗ Falha no treinamento", fg="#e74c3c")

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherTCC(root)
    root.mainloop()