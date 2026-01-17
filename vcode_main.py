import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tempfile
import traceback

# Import custom masked scripts
try:
    from bancoImagens import BancoImagens  
    from extratores import Extratores
    from arff import Arff
except ImportError as e:
    print(f"Error importing local modules: {e}")

# Weka Wrapper
import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
from weka.core.classes import Random

class VCodeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(" IA VCode Wood - Computer Vision Analyzer")
        self.geometry("1000x800")
        
        self.input_dir = ""
        self.jvm_started = False
        
        # Available Algorithms
        self.algorithms = {
            "Random Forest": "weka.classifiers.trees.RandomForest",
            "SVM (SMO)": "weka.classifiers.functions.SMO",
            "KNN (IBk)": "weka.classifiers.lazy.IBk",
            "J48 (Decision Tree)": "weka.classifiers.trees.J48"
        }
        
        self.setup_ui()
        # Start JVM in background
        threading.Thread(target=self._boot_jvm, daemon=True).start()

    def _boot_jvm(self):
        try:
            if not jvm.started:
                jvm.start(max_heap_size="2048m")
                self.jvm_started = True
                self.log(">>> [SYSTEM] Weka/Java Engine initialized successfully.")
        except Exception as e: 
            self.log(f">>> [JVM ERROR] Failed to load Java: {e}")

    def setup_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        self.lbl_logo = ctk.CTkLabel(self.sidebar, text="IA VCode Wood", font=("Arial", 22, "bold"))
        self.lbl_logo.pack(pady=30)

        self.btn_open = ctk.CTkButton(self.sidebar, text="1. Select Dataset", command=self.select_dir)
        self.btn_open.pack(pady=10, padx=20)

        ctk.CTkLabel(self.sidebar, text="2. Choose Algorithm:").pack(pady=(20, 0))
        self.algo_menu = ctk.CTkOptionMenu(self.sidebar, values=list(self.algorithms.keys()))
        self.algo_menu.pack(pady=10, padx=20)
        
        self.btn_run = ctk.CTkButton(self.sidebar, text="RUN ANALYSIS", 
                                     fg_color="#27ae60", hover_color="#1e8449", 
                                     command=self.start_processing)
        self.btn_run.pack(pady=40, padx=20)

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Status: Ready", font=("Arial", 11))
        self.lbl_status.pack(side="bottom", pady=20)

        self.console = ctk.CTkTextbox(self, font=("Consolas", 12), border_width=2)
        self.console.pack(side="right", fill="both", expand=True, padx=15, pady=15)

    def log(self, msg):
        self.console.insert("end", f"{msg}\n")
        self.console.see("end")

    def select_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.input_dir = path
            self.log(f"Selected folder: {path}")

    def start_processing(self):
        if not self.input_dir:
            messagebox.showwarning("Notice", "Please select the dataset folder first.")
            return
        if not self.jvm_started:
            messagebox.showwarning("Wait", "Initializing Java engine...")
            return
            
        self.btn_run.configure(state="disabled")
        self.console.delete("1.0", "end")
        threading.Thread(target=self.process_core, daemon=True).start()

    def process_core(self):
        try:
            self.log("="*50)
            self.log("STARTING FEATURE EXTRACTION")
            self.log("="*50)
            
            base_name = os.path.basename(self.input_dir)
            parent_dir = os.path.dirname(self.input_dir)
            # Save to Documents to avoid permission issues
            doc_path = os.path.join(os.path.expanduser("~"), "Documents", "VCode_Results")
            
            if not os.path.exists(doc_path):
                os.makedirs(doc_path)
            
            arff_filename = os.path.join(doc_path, f"{base_name}_analysis.arff")

            banco = BancoImagens(base_name, parent_dir)
            ext = Extratores()
            
            dados, nomes_at, tipos_at = [], [], []
            classes = banco.classes
            
            for classe in classes:
                imgs = banco.imagens_da_classe(classe)
                self.log(f"-> Class [{classe}]: Extracting {len(imgs)} images...")
                for img in imgs:
                    res = ext.extrai_todos(img)
                    if res and res[0]:
                        n, t, v = res
                        if not nomes_at: nomes_at, tipos_at = n, t
                        dados.append(v + [classe])
            
            if len(dados) == 0:
                raise Exception("No valid images found to extract data.")

            Arff().cria(arff_filename, dados, base_name, nomes_at, tipos_at, classes)
            self.log(f"\n[OK] ARFF generated: {arff_filename}")
            
            self.run_machine_learning(arff_filename)

        except Exception as e:
            self.log(f"\n[ERROR] {str(e)}")
        finally:
            self.btn_run.configure(state="normal")

    def run_machine_learning(self, arff_path):
        try:
            self.log("\n" + "="*50)
            self.log("STARTING VALIDATION")
            self.log("="*50)

            loader = Loader(classname="weka.core.converters.ArffLoader")
            data = loader.load_file(arff_path)
            data.class_is_last()
            
            num_instances = data.num_instances
            self.log(f"Dataset Size: {num_instances} instances.")

            # FIX: Adjust folds if dataset is too small
            folds = 10
            if num_instances < 10:
                folds = num_instances
                self.log(f"Warning: Low instance count. Adjusting to {folds}-Fold Validation.")
            
            if folds < 2:
                raise Exception("Not enough data. You need at least 2 images to validate.")

            algo_nome = self.algo_menu.get()
            self.log(f"Algorithm: {algo_nome}")
            
            cls = Classifier(classname=self.algorithms[algo_nome])
            evl = Evaluation(data)
            
            # Cross-Validation with adjusted folds
            evl.crossvalidate_model(cls, data, folds, Random(1))
            
            self.log("\n>>> PERFORMANCE SUMMARY:")
            self.log(evl.summary())
            
            self.log("\n>>> CLASS DETAILS:")
            self.log(evl.class_details())
            
            self.log("\n>>> CONFUSION MATRIX:")
            self.log(evl.matrix())
            
            self.log("="*50)
            self.log(f"FINAL ACCURACY: {evl.percent_correct:.2f}%")
            self.log("="*50)
            
            messagebox.showinfo("Success", f"Analysis Finished!\nAccuracy: {evl.percent_correct:.2f}%")

        except Exception as e:
            self.log(f"\n[WEKA ERROR] {str(e)}")
            self.log(traceback.format_exc())

if __name__ == "__main__":
    app = VCodeApp()
    app.protocol("WM_DELETE_WINDOW", lambda: (jvm.stop() if jvm.started else None, app.destroy()))
    app.mainloop()