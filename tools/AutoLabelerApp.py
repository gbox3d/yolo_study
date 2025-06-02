import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

from LabelingTabUI import LabelingTab # Assuming labeling_tab_ui_py_v3 is used
from ManagementTabUI import ManagementTab # Assuming management_tab_ui_py_v2 is used

from DataManager import DataManager   

class AutoLabelerApp:
    
    def __init__(self, master):
        self.master = master
        master.title(f"자동 라벨링 앱 v1.0 (탭 로직 분리)") 
        master.geometry("1250x750") 
        
        self.data_manager = DataManager()
        
        self.class_names = self.data_manager.get_class_names_dict()
        self.class_ids = self.data_manager.get_class_ids_dict()
        
        # Frame-specific label managers (key: frame_number, value: LabelManager instance)
        self.frame_label_managers = {} 

        if not self.data_manager.is_config_loaded():
            messagebox.showwarning("설정 로드 실패", f"{DataManager.CONFIG_FILE} 로드에 문제가 있습니다. 기본 설정으로 실행됩니다.")
        
        # --- UI Setup: Notebook for Tabs ---
        self.notebook = ttk.Notebook(master)
        
        labeling_tab_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(labeling_tab_frame, text='라벨링 작업')
        self.labeling_tab_ui = LabelingTab(labeling_tab_frame, self) # Pass self as controller

        management_tab_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(management_tab_frame, text='저장된 데이터 관리')
        self.management_tab_ui = ManagementTab(management_tab_frame, self) # Pass self as controller
        
        self.notebook.pack(expand=True, fill='both')
        
        self.update_ui_state_all_tabs() # Initial UI state

    def update_ui_state_all_tabs(self):
        if hasattr(self, 'labeling_tab_ui'): self.labeling_tab_ui.update_ui_state()
        if hasattr(self, 'management_tab_ui'): self.management_tab_ui.update_ui_state()

    

    def on_closing(self):
        # if self.cap: self.cap.release()
        self.master.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = AutoLabelerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
