import tkinter as tk
from tkinter import ttk, messagebox
import cv2 
from PIL import Image, ImageTk 
import os 

class ManagementTab:
    def __init__(self, parent_tab_frame, app_controller):
        self.parent_tab = parent_tab_frame
        self.app = app_controller 
        self.photo_image = None # For this tab's preview

        # --- UI Elements for Management Tab ---
        left_panel = ttk.Frame(self.parent_tab, padding="5")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        
        right_panel = ttk.Frame(self.parent_tab, width=400, padding="5") 
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        list_frame = ttk.LabelFrame(left_panel, text="저장된 이미지 목록", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.btn_refresh_list = ttk.Button(list_frame, text="목록 새로고침", command=self.refresh_list_action)
        self.btn_refresh_list.pack(fill=tk.X, pady=5)

        self.labeled_images_listbox = tk.Listbox(list_frame)
        self.labeled_images_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.labeled_images_listbox.bind("<<ListboxSelect>>", self.on_image_select_action)
        
        preview_frame = ttk.LabelFrame(right_panel, text="미리보기 및 라벨 정보", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(preview_frame, background="lightgrey", anchor=tk.CENTER, text="이미지 선택 시 미리보기")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(0,5)) 
        
        self.label_info_text = tk.Text(preview_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.label_info_text.pack(fill=tk.X, pady=(5,0))
        
        self.refresh_list_action() # Initial list load

    def refresh_list_action(self):
        """UI action to refresh the list of labeled images."""
        # self.labeled_image_files = self.data_manager.list_labeled_images()
        # self.update_listbox_display(self.app.labeled_image_files)
        
        pass

    def on_image_select_action(self, event):
        """UI action when an image is selected from the list."""
        if not self.labeled_images_listbox.curselection():
            self.app.selected_labeled_image_file = None
            self.display_preview_and_labels(None, [])
            return

        selected_index = self.labeled_images_listbox.curselection()[0]
        if 0 <= selected_index < len(self.app.labeled_image_files):
            self.app.selected_labeled_image_file = self.app.labeled_image_files[selected_index]
            img_path = self.app.data_manager.get_image_path(self.app.selected_labeled_image_file)
            yolo_labels = self.app.data_manager.load_yolo_labels_from_file(self.app.selected_labeled_image_file)
            self.display_preview_and_labels(img_path, yolo_labels)

    # --- UI Update Methods (called by app_controller or self) ---
    def update_ui_state(self):
        pass # No dynamic state changes for buttons in this tab currently

    def update_listbox_display(self, image_files):
        self.labeled_images_listbox.delete(0, tk.END)
        for filename in image_files:
            self.labeled_images_listbox.insert(tk.END, filename)
        if not image_files:
            self.labeled_images_listbox.insert(tk.END, "저장된 이미지가 없습니다.")
        
        if not self.labeled_images_listbox.curselection(): # If nothing is selected (or list is empty)
            self.display_preview_and_labels(None, [])


    def display_preview_and_labels(self, image_path, yolo_labels):
        if image_path and os.path.exists(image_path):
            try:
                img_bgr = cv2.imread(image_path)
                if img_bgr is None: raise ValueError("이미지를 읽을 수 없습니다.")
                
                display_img_for_preview = img_bgr.copy()
                label_info_str_parts = [f"이미지: {os.path.basename(image_path)}\n--- 라벨 ---"]

                if yolo_labels:
                    img_h_orig, img_w_orig = display_img_for_preview.shape[:2]
                    for lbl in yolo_labels:
                        class_name = lbl['class_name']
                        cx_rel, cy_rel, w_rel, h_rel = lbl['bbox_yolo']
                        abs_w=w_rel*img_w_orig; abs_h=h_rel*img_h_orig; abs_cx=cx_rel*img_w_orig; abs_cy=cy_rel*img_h_orig
                        x1=int(abs_cx-abs_w/2); y1=int(abs_cy-abs_h/2); x2=int(abs_cx+abs_w/2); y2=int(abs_cy+abs_h/2)
                        cv2.rectangle(display_img_for_preview,(x1,y1),(x2,y2), self.app.SAVED_BBOX_COLOR,2) 
                        cv2.putText(display_img_for_preview,class_name,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.app.SAVED_BBOX_COLOR,1)
                        label_info_str_parts.append(f"{class_name}: [{cx_rel:.3f},{cy_rel:.3f},{w_rel:.3f},{h_rel:.3f}]")
                else:
                    label_info_str_parts.append("라벨 없음")

                preview_label_w = self.preview_label.winfo_width()
                preview_label_h = self.preview_label.winfo_height()
                if preview_label_w <=1 or preview_label_h <=1: preview_label_w, preview_label_h = 400,300

                img_h, img_w = display_img_for_preview.shape[:2]
                scale = min(preview_label_w / img_w, preview_label_h / img_h) if img_w > 0 and img_h > 0 else 1.0
                
                disp_w = int(img_w * scale) if scale > 0 else img_w
                disp_h = int(img_h * scale) if scale > 0 else img_h
                
                if disp_w > 0 and disp_h > 0 :
                    resized_img = cv2.resize(display_img_for_preview,(disp_w,disp_h))
                else: 
                     resized_img = display_img_for_preview 

                img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                self.photo_image = ImageTk.PhotoImage(image=pil_img) 
                self.preview_label.config(image=self.photo_image, text="")

                self.label_info_text.config(state=tk.NORMAL)
                self.label_info_text.delete(1.0, tk.END)
                self.label_info_text.insert(tk.END, "\n".join(label_info_str_parts))
                self.label_info_text.config(state=tk.DISABLED)
            except Exception as e:
                print(f"미리보기 이미지 로드/표시 오류: {e}")
                self.preview_label.config(image=None, text="미리보기 로드 오류")
                self.photo_image = None
        else:
            self.preview_label.config(image=None, text="이미지 파일을 찾을 수 없음" if image_path else "이미지 선택 시 미리보기")
            self.photo_image = None
            self.label_info_text.config(state=tk.NORMAL)
            self.label_info_text.delete(1.0, tk.END)
            if not image_path: self.label_info_text.insert(tk.END, "표시할 라벨 정보 없음")
            self.label_info_text.config(state=tk.DISABLED)
