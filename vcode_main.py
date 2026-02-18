import os
import sys
import threading
import traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- BIBLIOTECAS ---
import cv2
# -------------------

# --- MÓDULOS LOCAIS ---
try:
    from extratores import Extratores
    from arff import Arff
except ImportError as e:
    print("ERRO: Módulos locais não encontrados.")
    sys.exit(1)

# --- WEKA WRAPPER ---
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.filters import Filter
from weka.core.converters import Loader
from weka.core.classes import Random

class VCodeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("IA VCode Wood - SVM Fast Fix")
        self.geometry("1000x800")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.image_path = ""                  
        self.train_arff = "arquivoarff.arff"  
        self.jvm_started = False
        
        self.setup_ui()
        
        if not os.path.exists(self.train_arff):
            self.log(f"[ALERTA] Arquivo de treino '{self.train_arff}' não encontrado.")
        
        print(">>> Iniciando JVM...")
        threading.Thread(target=self._boot_jvm, daemon=True).start()

    def _boot_jvm(self):
        try:
            if not jvm.started:
                jvm.start(max_heap_size="2048m") 
                self.jvm_started = True
                self.log(">>> [SISTEMA] Motor Online.")
        except Exception as e: 
            self.log(f">>> [ERRO JVM] {e}")

    def setup_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar, text="IA VCode Wood", font=("Arial", 22, "bold")).pack(pady=30)
        ctk.CTkLabel(self.sidebar, text="SVM Edition", font=("Arial", 14), text_color="gray").pack(pady=(0, 30))

        self.btn_open = ctk.CTkButton(self.sidebar, text="1. Selecionar Imagem", command=self.select_image)
        self.btn_open.pack(pady=10, padx=20)
        
        self.btn_run = ctk.CTkButton(self.sidebar, text="CLASSIFICAR (SVM)", 
                                     fg_color="#27ae60", hover_color="#1e8449", 
                                     height=50, font=("Arial", 14, "bold"),
                                     command=self.start_processing)
        self.btn_run.pack(pady=40, padx=20)

        self.console = ctk.CTkTextbox(self, font=("Consolas", 12), border_width=2)
        self.console.pack(side="right", fill="both", expand=True, padx=15, pady=15)

    def log(self, msg):
        print(f"[LOG] {msg}") 
        self.console.insert("end", f"{msg}\n")
        self.console.see("end")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path = path
            self.log(f"Imagem: {os.path.basename(path)}")

    def start_processing(self):
        if not self.image_path:
            messagebox.showwarning("Aviso", "Selecione uma imagem.")
            return
        if not self.jvm_started:
            messagebox.showwarning("Aviso", "Aguarde o motor Java iniciar.")
            return
            
        self.btn_run.configure(state="disabled")
        self.console.delete("1.0", "end")
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def fix_arff_header(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = []
            changed = False
            for line in lines:
                if line.lower().startswith("@attribute") and "{" in line:
                    parts = line.split("{")
                    content = parts[1].split("}")[0]
                    labels = [l.strip() for l in content.split(",")]
                    unique = sorted(list(set(labels)))
                    if len(labels) != len(unique):
                        new_lines.append(f"{parts[0]}{{{', '.join(unique)}}}\n")
                        changed = True
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            if changed:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                self.log(">>> Cabeçalho ARFF reparado.")
        except:
            pass

    def run_pipeline(self):
        temp_arff = "temp_svm_run.arff"
        try:
            self.log("="*50)
            self.log("INICIANDO SVM (FAST MODE)")
            self.log("="*50)
            
            # 1. Carregar Treino
            self.fix_arff_header(self.train_arff)
            loader = Loader(classname="weka.core.converters.ArffLoader")
            train_data = loader.load_file(self.train_arff)
            train_data.class_is_last()
            
            # 2. Imagem
            img = cv2.imread(self.image_path)
            if img is None: raise Exception("Erro ao ler imagem.")

            ext = Extratores()
            res = ext.extrai_todos(img)
            
            if not res or res[0] is None: raise Exception("Falha na extração.")
            names, types, values = res

            # Validação
            if len(values) != (train_data.num_attributes - 1):
                self.log(f"[ERRO] Atributos inválidos! Treino: {train_data.num_attributes-1}, Imagem: {len(values)}")
                raise Exception("Inconsistência no ARFF.")

            # 3. Criar Teste
            class_attr = train_data.class_attribute
            class_vals = [class_attr.value(i) for i in range(class_attr.num_values)]
            values.append(class_vals[0]) 
            
            Arff().cria(temp_arff, [values], "Test", names, types, class_vals)
            test_data = loader.load_file(temp_arff)
            test_data.class_is_last()

            # --- LIMPEZA DE DADOS (CRUCIAL PARA NÃO TRAVAR) ---
            self.log("Aplicando limpeza de dados (RemoveUseless)...")
            
            # Criamos o filtro explicitamente
            remove = Filter(classname="weka.filters.unsupervised.attribute.RemoveUseless")
            remove.options = ["-M", "99.0"] # Tolerância máxima para remover lixo
            
            # Aplicamos o filtro no TREINO para criar um NOVO dataset limpo
            remove.inputformat(train_data)
            clean_train = remove.filter(train_data)
            
            # Aplicamos o MESMO filtro no TESTE (para garantir compatibilidade)
            clean_test = remove.filter(test_data)
            
            self.log(f"Atributos originais: {train_data.num_attributes}")
            self.log(f"Atributos limpos: {clean_train.num_attributes}")

            # --- CONFIGURAÇÃO SVM RÁPIDA ---
            self.log("Treinando SVM...")
            
            svm = Classifier(classname="weka.classifiers.functions.SMO")
            # -M: Probabilidade
            # -N 0: Sem normalização interna (já limpamos os dados)
            # -C 1.0: Complexidade padrão
            # -K ...PolyKernel -E 1.0: Kernel Linear (MUITO MAIS RÁPIDO)
            svm.options = ["-M", "-N", "0", "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 1.0"]
            
            # Treina com os dados LIMPOS
            svm.build_classifier(clean_train)
            
            # 5. Classificação
            self.log("Classificando...")
            inst = clean_test.get_instance(0) # Pega instância limpa
            
            pred = svm.classify_instance(inst)
            pred_class = clean_train.class_attribute.value(int(pred)) # Usa header limpo
            
            dist = svm.distribution_for_instance(inst)
            confidence = dist[int(pred)] * 100

            self.log("\n" + "="*30)
            self.log(f"RESULTADO:")
            self.log(f"Classe: {pred_class.upper()}")
            self.log(f"Confiança: {confidence:.2f}%")
            self.log("="*30 + "\n")
            
            msg = f"Resultado:\n{pred_class.upper()}\nConfiança: {confidence:.2f}%"
            messagebox.showinfo("Classificação SVM", msg)

        except Exception as e:
            self.log(f"\n[ERRO] {e}")
            print(traceback.format_exc())
        finally:
            self.btn_run.configure(state="normal")
            if os.path.exists(temp_arff):
                try: os.remove(temp_arff)
                except: pass

if __name__ == "__main__":
    app = VCodeApp()
    app.protocol("WM_DELETE_WINDOW", lambda: (jvm.stop() if jvm.started else None, app.destroy()))
    app.mainloop()