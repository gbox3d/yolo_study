"""
author: gbox3d
date : 2025-6-2

이 주석은 수정하지마시오.

"""
import tkinter as tk
from tkinter import ttk, messagebox
from LabelingTabUI import LabelingTab # Assuming labeling_tab_ui_py_v3 is used
from ManagementTabUI import ManagementTab # Assuming management_tab_ui_py_v2 is used

from DataManager import DataManager   

class AutoLabelerApp:
    
    def __init__(self, master):
        self.master = master
        master.title(f"자동 라벨링 앱 v1.0") 
        master.geometry("1200x700")  # 640 + 500 + 여백 고려 
        
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
        
        # 탭 변경 이벤트 바인딩
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        self.update_ui_state_all_tabs() # Initial UI state

    def on_tab_changed(self, event):
        """탭이 변경될 때 호출되는 메서드"""
        try:
            selected_tab = event.widget.tab('current')['text']
            if selected_tab == '저장된 데이터 관리':
                # 관리 탭으로 전환될 때 약간의 지연 후 목록 새로고침
                self.master.after(100, self._refresh_management_tab)
        except Exception as e:
            print(f"Tab change error: {e}")

    def _refresh_management_tab(self):
        """관리 탭 새로고침 (지연 실행)"""
        try:
            if hasattr(self, 'management_tab_ui'):
                self.management_tab_ui.refresh_list_action()
                self.management_tab_ui.update_ui_state()
        except Exception as e:
            print(f"Management tab refresh error: {e}")

    def update_ui_state_all_tabs(self):
        """모든 탭의 UI 상태를 업데이트합니다."""
        if hasattr(self, 'labeling_tab_ui'): 
            self.labeling_tab_ui.update_ui_state()
        if hasattr(self, 'management_tab_ui'): 
            self.management_tab_ui.update_ui_state()

    def on_closing(self):
        """애플리케이션 종료 시 정리 작업"""
        try:
            # 비디오 캡처 해제
            if hasattr(self.labeling_tab_ui, 'cap') and self.labeling_tab_ui.cap:
                self.labeling_tab_ui.cap.release()
                print("Video capture released.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.master.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = AutoLabelerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()