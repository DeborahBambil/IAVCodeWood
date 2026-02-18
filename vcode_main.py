import os
import sys
import threading
import traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- LIBRARIES ---
import cv2
import numpy as np
# -----------------

# --- LOCAL MODULES ---
try:
    from extratores import Extratores
    from arff import Arff
except ImportError as e:
    print("ERROR: Local modules 'extratores.py' or 'arff.py' not found.")
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
        
        # Window Configuration
        self.title("IA VCode Wood - SVM Pro")
        self.geometry("1100x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        # State Variables
        self.image_path = ""                  
        self.train_arff = "" # Now selected by user
        self.jvm_started = False
        
        self.setup_ui()
        
        print(">>> Starting JVM...")
        threading.Thread(target=self._boot_jvm, daemon=True).start()

    def _boot_jvm(self):
        try:
            if not jvm.started:
                # Allocating 4GB RAM to prevent freezing during training
                jvm.start(max_heap_size="4096m") 
                self.jvm_started = True
                self.log(">>> [SYSTEM] Engine Online.")
        except Exception as e: 
            self.log(f">>> [JVM ERROR] {e}")

    def setup_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar, text="IA VCode Wood", font=("Arial", 22, "bold")).pack(pady=30)
        ctk.CTkLabel(self.sidebar, text="SVM Classifier", font=("Arial", 14), text_color="gray").pack(pady=(0, 30))

        # --- STEP 1: ARFF SELECTION ---
        ctk.CTkLabel(self.sidebar, text="Step 1: Database", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.btn_arff = ctk.CTkButton(self.sidebar, text="Select .ARFF File", 
                                      fg_color="#2980b9", hover_color="#2471a3",
                                      command=self.select_arff)
        self.btn_arff.pack(pady=5, padx=20)
        self.lbl_arff = ctk.CTkLabel(self.sidebar, text="No file selected", font=("Arial", 10), text_color="gray")
        self.lbl_arff.pack(pady=(0, 15))

        # --- STEP 2: IMAGE SELECTION ---
        ctk.CTkLabel(self.sidebar, text="Step 2: Input Image", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.btn_img = ctk.CTkButton(self.sidebar, text="Select Image", 
                                     fg_color="#d35400", hover_color="#a04000",
                                     command=self.select_image)
        self.btn_img.pack(pady=5, padx=20)
        self.lbl_img = ctk.CTkLabel(self.sidebar, text="No image selected", font=("Arial", 10), text_color="gray")
        self.lbl_img.pack(pady=(0, 25))
        
        # --- STEP 3: RUN ---
        self.btn_run = ctk.CTkButton(self.sidebar, text="3. RUN SVM", 
                                     fg_color="#27ae60", hover_color="#1e8449", 
                                     height=50, font=("Arial", 14, "bold"),
                                     state="disabled",
                                     command=self.start_processing)
        self.btn_run.pack(pady=20, padx=20)

        # Status & Console
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Status: Waiting...", font=("Arial", 11))
        self.lbl_status.pack(side="bottom", pady=20)

        self.console = ctk.CTkTextbox(self, font=("Consolas", 12), border_width=2)
        self.console.pack(side="right", fill="both", expand=True, padx=15, pady=15)

    def log(self, msg):
        print(f"[LOG] {msg}") 
        self.console.insert("end", f"{msg}\n")
        self.console.see("end")

    def select_arff(self):
        path = filedialog.askopenfilename(filetypes=[("Weka Files", "*.arff")])
        if path:
            self.train_arff = path
            self.lbl_arff.configure(text=os.path.basename(path)[:25] + "...")
            self.log(f"Database selected: {os.path.basename(path)}")
            self.check_ready()

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path = path
            self.lbl_img.configure(text=os.path.basename(path)[:25] + "...")
            self.log(f"Image selected: {os.path.basename(path)}")
            self.check_ready()

    def check_ready(self):
        if self.train_arff and self.image_path:
            self.btn_run.configure(state="normal")
            self.lbl_status.configure(text="Status: Ready to Run")

    def start_processing(self):
        if not self.jvm_started:
            messagebox.showwarning("Wait", "Java engine is still starting...")
            return
            
        self.btn_run.configure(state="disabled")
        self.console.delete("1.0", "end")
        
        # Runs inside a thread so the window doesn't freeze
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def fix_arff_header(self, filepath):
        """ Scans ARFF for duplicate attribute labels and fixes them. """
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
                self.log(">>> ARFF header repaired (duplicates removed).")
        except:
            pass

    def run_pipeline(self):
        temp_arff = "temp_svm_debug.arff"
        try:
            self.log("="*50)
            self.log("STARTING DIAGNOSIS")
            self.log("="*50)
            
            # 1. Load Training Data
            self.log(f"Loading: {os.path.basename(self.train_arff)}...")
            self.fix_arff_header(self.train_arff)
            
            loader = Loader(classname="weka.core.converters.ArffLoader")
            train_data = loader.load_file(self.train_arff)
            train_data.class_is_last()
            
            self.log(f"Training Data: {train_data.num_instances} samples.")

            # 2. Process Image
            img_bgr = cv2.imread(self.image_path)
            if img_bgr is None: raise Exception("Error reading image file.")

            # --- COLOR CONVERSION ---
            # OpenCV reads as BGR. We convert to RGB to match standard training data.
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ext = Extratores()
            res = ext.extrai_todos(img_rgb)
            
            if not res or res[0] is None: raise Exception("Feature extraction failed.")
            names, types, values = res

            # --- VISUAL DEBUG ---
            self.log("\n" + "#"*40)
            self.log("EXTRACTED DATA (First 5 values):")
            formatted_values = [f"{v:.4f}" if isinstance(v, (float, int)) else str(v) for v in values[:5]]
            self.log(str(formatted_values))
            self.log("Compare these with your .ARFF file!")
            self.log("#"*40 + "\n")
            # --------------------

            # Validation
            if len(values) != (train_data.num_attributes - 1):
                self.log(f"[ERROR] Attribute Mismatch!")
                self.log(f"Train expects: {train_data.num_attributes-1}")
                self.log(f"Image has: {len(values)}")
                raise Exception("ARFF file is incompatible with current extractors.")

            # 3. Create Temp Test File
            class_attr = train_data.class_attribute
            class_vals = [class_attr.value(i) for i in range(class_attr.num_values)]
            values.append(class_vals[0]) 
            
            Arff().cria(temp_arff, [values], "Test", names, types, class_vals)
            test_data = loader.load_file(temp_arff)
            test_data.class_is_last()

            # --- DATA CLEANING ---
            self.log("Cleaning data (removing constant attributes)...")
            remove = Filter(classname="weka.filters.unsupervised.attribute.RemoveUseless")
            remove.options = ["-M", "99.0"]
            
            # Batch filtering to ensure synchronization
            remove.inputformat(train_data)
            clean_train = remove.filter(train_data)
            clean_test = remove.filter(test_data)
            
            if clean_train.num_attributes != clean_test.num_attributes:
                self.log("Re-synchronizing filters...")
                remove.inputformat(train_data)
                clean_test = remove.filter(test_data)

            # --- SVM CONFIGURATION ---
            self.log("Training SVM (Linear Kernel - Fast Mode)...")
            
            svm = Classifier(classname="weka.classifiers.functions.SMO")
            # -M: Output probabilities
            # -N 0: No internal normalization (we already cleaned data)
            # -K PolyKernel -E 1.0: Linear Kernel (Prevents freezing on large datasets)
            svm.options = ["-M", "-N", "0", "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 1.0"]
            
            svm.build_classifier(clean_train)
            
            # 5. Classification
            self.log("Classifying...")
            inst = clean_test.get_instance(0)
            
            pred = svm.classify_instance(inst)
            pred_class = clean_train.class_attribute.value(int(pred))
            
            dist = svm.distribution_for_instance(inst)
            confidence = dist[int(pred)] * 100

            self.log("\n" + "="*30)
            self.log(f"RESULT:")
            self.log(f"Class: {pred_class.upper()}")
            self.log(f"Confidence: {confidence:.2f}%")
            self.log("="*30 + "\n")
            
            # Final Message
            msg = f"Identified Wood:\n{pred_class.upper()}\n\nConfidence: {confidence:.2f}%"
            if confidence < 99.0:
                 msg += "\n\n[WARNING] Confidence is below 100%.\nData mismatch detected."
                 messagebox.showwarning("Result (Low Confidence)", msg)
            else:
                 messagebox.showinfo("Success", msg)

        except Exception as e:
            self.log(f"\n[ERROR] {e}")
            print(traceback.format_exc())
        finally:
            self.btn_run.configure(state="normal")
            self.lbl_status.configure(text="Status: Finished")
            if os.path.exists(temp_arff):
                try: os.remove(temp_arff)
                except: pass

if __name__ == "__main__":
    app = VCodeApp()
    app.protocol("WM_DELETE_WINDOW", lambda: (jvm.stop() if jvm.started else None, app.destroy()))
    app.mainloop()