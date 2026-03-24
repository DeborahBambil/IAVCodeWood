import os
import sys
import threading
import traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- LIBRARIES ---
import cv2
import numpy as np

# --- LOCAL MODULES ---
try:
    from extratores import Extratores
    from arff import Arff
except ImportError:
    print("CRITICAL ERROR: 'extratores.py' or 'arff.py' not found in the script directory.")
    sys.exit(1)

# --- WEKA WRAPPER ---
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.core.dataset import Instance, Instances

class VCodeApp(ctk.CTk):
    """
    IA VCode Wood - SVM Classifier
    Author: Deborah Bambil
    """
    def __init__(self):
        super().__init__()
        
        # Window Configuration
        self.title("IA VCode Wood - SVM Pro (Author: Deborah Bambil)")
        self.geometry("1100x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        # State Variables
        self.image_path = ""                  
        self.train_arff = "" 
        self.jvm_started = False
        
        self.setup_ui()
        
        # Start JVM in Background
        self.log(">>> [SYSTEM] Initializing Java Engine...")
        threading.Thread(target=self._boot_jvm, daemon=True).start()

    def _boot_jvm(self):
        try:
            if not jvm.started:
                # Allocating 4GB RAM to prevent freezing during large dataset training
                jvm.start(max_heap_size="4096m") 
                self.jvm_started = True
                self.log(">>> [SYSTEM] Engine Online.")
        except Exception as e: 
            self.log(f">>> [JVM ERROR] {e}")

    def setup_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar, text="VCode Wood IA", font=("Arial", 24, "bold")).pack(pady=30)
        ctk.CTkLabel(self.sidebar, text="by Deborah Bambil", font=("Arial", 12), text_color="gray").pack(pady=(0, 20))
        
        # Step 1: Database selection
        ctk.CTkLabel(self.sidebar, text="1. Training Database (.arff)", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.btn_arff = ctk.CTkButton(self.sidebar, text="Select ARFF File", fg_color="#2980b9", hover_color="#3498db", command=self.select_arff)
        self.btn_arff.pack(pady=10, padx=20)
        self.lbl_arff = ctk.CTkLabel(self.sidebar, text="No file selected", text_color="gray", font=("Arial", 10))
        self.lbl_arff.pack()

        # Step 2: Image selection
        ctk.CTkLabel(self.sidebar, text="2. Wood Sample Image", font=("Arial", 12, "bold")).pack(pady=(20, 0))
        self.btn_img = ctk.CTkButton(self.sidebar, text="Select Image", fg_color="#d35400", hover_color="#e67e22", command=self.select_image)
        self.btn_img.pack(pady=10, padx=20)
        self.lbl_img = ctk.CTkLabel(self.sidebar, text="No image selected", text_color="gray", font=("Arial", 10))
        self.lbl_img.pack()
        
        # Step 3: Run button
        self.btn_run = ctk.CTkButton(self.sidebar, text="RUN IDENTIFICATION", fg_color="#27ae60", hover_color="#2ecc71", 
                                     height=50, font=("Arial", 14, "bold"), state="disabled", command=self.start_processing)
        self.btn_run.pack(pady=40, padx=20)

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Status: Waiting for JVM...", font=("Arial", 11))
        self.lbl_status.pack(side="bottom", pady=20)

        # Console for logging
        self.console = ctk.CTkTextbox(self, font=("Consolas", 12), border_width=2)
        self.console.pack(side="right", fill="both", expand=True, padx=15, pady=15)

    def log(self, msg):
        self.console.insert("end", f"{msg}\n")
        self.console.see("end")

    def select_arff(self):
        path = filedialog.askopenfilename(filetypes=[("Weka Files", "*.arff")])
        if path:
            self.train_arff = path
            self.lbl_arff.configure(text=os.path.basename(path))
            self.check_ready()

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp *.tif *.jpeg")])
        if path:
            self.image_path = path
            self.lbl_img.configure(text=os.path.basename(path))
            self.check_ready()

    def check_ready(self):
        if self.train_arff and self.image_path:
            self.btn_run.configure(state="normal")
            self.lbl_status.configure(text="Status: Ready to Run")

    def start_processing(self):
        if not self.jvm_started: return
        self.btn_run.configure(state="disabled")
        self.console.delete("1.0", "end")
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        try:
            self.log("="*40)
            self.log("STARTING IDENTIFICATION PIPELINE")
            self.log("="*40)
            
            # 1. Load Training Data
            self.log(">>> Loading training database...")
            loader = Loader(classname="weka.core.converters.ArffLoader")
            train_data = loader.load_file(self.train_arff)
            train_data.class_is_last()
            
            # Extract class labels
            class_attr = train_data.class_attribute
            class_labels = [class_attr.value(i) for i in range(class_attr.num_values)]
            
            # 2. Extract Features from selected image
            self.log(">>> Processing image (Feature Extraction)...")
            img_bgr = cv2.imread(self.image_path)
            if img_bgr is None: raise Exception("Unable to open image file.")

            # Note: We keep BGR because extratores.py expects it for its internal conversions
            ext = Extratores()
            # Returns: names, types, values
            res = ext.extrai_todos(img_bgr)
            values = res[2]

            # Validation: Attribute count must match the .arff file
            expected_count = train_data.num_attributes - 1
            if len(values) != expected_count:
                self.log(f"[ERROR] Attribute mismatch!")
                self.log(f"ARFF expects: {expected_count}")
                self.log(f"Extractor generated: {len(values)}")
                raise Exception("The number of extracted features does not match the training file structure.")

            # 3. Align Instance with Training Structure
            # Copy instances structure from train_data to ensure column order is perfect
            test_dataset = Instances.copy_instances(train_data, 0)
            inst = Instance.create_instance(values + [0.0]) # values + dummy class index
            inst.dataset = test_dataset
            
            # 4. Configure and Train SVM (SMO)
            # -N 1: Normalization (Crucial to prevent the major class bias you experienced)
            # -M  : Outputs probability distribution (needed for confidence %)
            self.log(">>> Building SMO (SVM) Classifier...")
            svm = Classifier(classname="weka.classifiers.functions.SMO")
            svm.options = ["-M", "-N", "1", "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 1.0"]
            svm.build_classifier(train_data)
            
            # 5. Prediction
            self.log(">>> Classifying...")
            pred_idx = svm.classify_instance(inst)
            label = class_labels[int(pred_idx)]
            
            # Calculate Confidence
            probs = svm.distribution_for_instance(inst)
            confidence = probs[int(pred_idx)] * 100

            self.log(f"\n[SUCCESS] IDENTIFIED: {label.upper()}")
            self.log(f"[SUCCESS] CONFIDENCE: {confidence:.2f}%")
            
            # Final Pop-up
            messagebox.showinfo("Result", f"Identified Wood: {label.upper()}\nConfidence: {confidence:.2f}%")

        except Exception as e:
            self.log(f"\n[PIPELINE ERROR] {e}")
            print(traceback.format_exc())
        finally:
            self.btn_run.configure(state="normal")
            self.lbl_status.configure(text="Status: Finished")

if __name__ == "__main__":
    app = VCodeApp()
    # Ensure JVM stops cleanly when closing the UI
    app.protocol("WM_DELETE_WINDOW", lambda: (jvm.stop() if jvm.started else None, app.destroy()))
    app.mainloop()