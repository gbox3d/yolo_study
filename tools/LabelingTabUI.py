import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

from SegmentationProcessor import SegmentationProcessor
from LabelManager import LabelManager 

class LabelingTab:
    
    # Constants can be defined here and accessed by tab UI classes via self.app
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 640
    HANDLE_SIZE = 8 
    HANDLE_COLOR = (255, 165, 0) # BGR
    ACTIVE_BBOX_COLOR = (0, 255, 255) # BGR
    SAVED_BBOX_COLOR = (255, 0, 255)   # BGR
    
    def __init__(self, parent_tab_frame, app_controller):
        self.parent_tab = parent_tab_frame
        self.app = app_controller # Reference to the main AutoLabelerApp instance
        
        # --- Initialize Managers ---
        
        self.segmentation_processor = SegmentationProcessor(
            model_path="sam2_s.pt", 
            processing_width=self.PROCESSING_WIDTH,
            processing_height=self.PROCESSING_HEIGHT
        )
        
        
        if not self.segmentation_processor.is_model_loaded():
            messagebox.showwarning("모델 로드 실패", "SAM 모델 로드에 실패했습니다. 세그멘테이션 기능이 제한될 수 있습니다.")
        
        # --- Core Application State (shared or controlled at app level) ---
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.current_frame_number = 0
        self.original_video_width = 0
        self.original_video_height = 0
        self.video_fps = 0
        self.is_video_loaded = False

        # Tab-specific state
        self.photo_image = None 
        self.current_active_mask_processed = None 
        self.current_active_polygons_processed = [] 
        self.current_active_bbox_processed = None   
        self.selected_label_id_from_list = None 
        self.dragging_handle = None
        self.drag_start_mouse_pos = None
        self.drag_start_bbox_processed = None

        # --- UI Elements ---
        top_frame = ttk.Frame(self.parent_tab)
        top_frame.pack(fill=tk.BOTH, expand=True)

        self.video_frame_container = ttk.LabelFrame(top_frame, text="비디오 (라벨링)", width=800, height=700)
        self.video_frame_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.video_frame_container.pack_propagate(False)

        self.video_label = ttk.Label(self.video_frame_container, background="black", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.video_label.bind("<ButtonPress-1>", self.on_mouse_press)
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.video_label.bind("<Button-3>", self.on_right_click_clear_active_bbox)

        control_panel = ttk.Frame(top_frame, width=380)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        control_panel.pack_propagate(False)

        file_info_frame = ttk.LabelFrame(control_panel, text="파일 & 정보", padding="10")
        file_info_frame.pack(fill=tk.X, pady=5)
        self.btn_open = ttk.Button(file_info_frame, text="파일 열기", command=self.open_file)
        self.btn_open.pack(fill=tk.X, pady=2)
        self.lbl_file_name = ttk.Label(file_info_frame, text="파일: 없음", wraplength=340)
        self.lbl_file_name.pack(fill=tk.X, pady=2)
        self.lbl_frame_info = ttk.Label(file_info_frame, text="프레임: N/A / N/A (FPS: N/A)")
        self.lbl_frame_info.pack(fill=tk.X, pady=2)
        
        nav_frame = ttk.LabelFrame(control_panel, text="프레임 이동", padding="10")
        nav_frame.pack(fill=tk.X, pady=5)
        frame_nav_buttons_frame = ttk.Frame(nav_frame)
        frame_nav_buttons_frame.pack(fill=tk.X, pady=2)
        self.btn_prev_frame = ttk.Button(frame_nav_buttons_frame, text="<< 이전", command=self.prev_frame, state=tk.DISABLED)
        self.btn_prev_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        self.btn_next_frame = ttk.Button(frame_nav_buttons_frame, text="다음 >>", command=self.next_frame, state=tk.DISABLED)
        self.btn_next_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2,0))
        goto_frame_container = ttk.Frame(nav_frame)
        goto_frame_container.pack(fill=tk.X, pady=2)
        self.lbl_goto = ttk.Label(goto_frame_container, text="프레임 이동:")
        self.lbl_goto.pack(side=tk.LEFT, padx=(0,5))
        self.entry_goto_frame = ttk.Entry(goto_frame_container, width=8)
        self.entry_goto_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.entry_goto_frame.bind("<Return>", self.go_to_frame_event) 
        self.btn_goto_frame = ttk.Button(goto_frame_container, text="이동", command=self.go_to_frame, state=tk.DISABLED, width=5) 
        self.btn_goto_frame.pack(side=tk.RIGHT, padx=(5,0))

        labeling_ops_frame = ttk.LabelFrame(control_panel, text="라벨링 작업", padding="10")
        labeling_ops_frame.pack(fill=tk.X, pady=5)
        ttk.Label(labeling_ops_frame, text="클래스 선택:").pack(anchor=tk.W)
        self.class_var = tk.StringVar()
        self.class_combobox = ttk.Combobox(labeling_ops_frame, textvariable=self.class_var, 
                                           values=list(self.app.class_names.values()), state="readonly")
        if self.app.class_names: self.class_combobox.current(0)
        
        
        self.class_combobox.pack(fill=tk.X, pady=2)
        self.btn_add_label = ttk.Button(labeling_ops_frame, text="라벨 추가/업데이트", command=self.add_or_update_label, state=tk.DISABLED)
        self.btn_add_label.pack(fill=tk.X, pady=5)
        self.btn_clear_active = ttk.Button(labeling_ops_frame, text="활성 세그멘테이션 지우기", command=self.clear_active_segmentation_data_ui_action, state=tk.DISABLED)
        self.btn_clear_active.pack(fill=tk.X, pady=2)

        labels_list_frame = ttk.LabelFrame(control_panel, text="현재 프레임 라벨 목록", padding="10")
        labels_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.labels_listbox = tk.Listbox(labels_list_frame, height=6)
        self.labels_listbox.pack(fill=tk.BOTH, expand=True, pady=2)
        self.labels_listbox.bind("<<ListboxSelect>>", self.on_label_select_from_list)
        self.btn_delete_label = ttk.Button(labels_list_frame, text="선택된 라벨 삭제", command=self.delete_selected_label, state=tk.DISABLED)
        self.btn_delete_label.pack(fill=tk.X, pady=5)

        save_frame = ttk.LabelFrame(control_panel, text="저장", padding="10")
        save_frame.pack(fill=tk.X, pady=5)
        ttk.Label(save_frame, text="파일 이름 접두사:").pack(anchor=tk.W)
        self.filename_prefix_var = tk.StringVar(value="frame")
        self.entry_filename_prefix = ttk.Entry(save_frame, textvariable=self.filename_prefix_var)
        self.entry_filename_prefix.pack(fill=tk.X, pady=2)
        self.btn_save_labels = ttk.Button(save_frame, text="현재 프레임 라벨 저장", command=self.save_current_frame_and_labels_ui, state=tk.DISABLED)
        self.btn_save_labels.pack(fill=tk.X, pady=5)
        
        self.btn_about = ttk.Button(control_panel, text="정보 (About)", command=self.show_about)
        self.btn_about.pack(fill=tk.X, pady=10, side=tk.BOTTOM)

    # --- Coordinate Conversion Helpers ---
    def _get_click_coords_on_displayed_image(self, event_x, event_y):
        if self.original_video_width == 0 or self.original_video_height == 0: return None
        label_display_width = self.video_label.winfo_width()
        label_display_height = self.video_label.winfo_height()
        if label_display_width <= 1 or label_display_height <= 1 : return None
        scale_for_display_w = label_display_width / self.original_video_width
        scale_for_display_h = label_display_height / self.original_video_height
        scale_for_display = min(scale_for_display_w, scale_for_display_h)
        if scale_for_display <= 0: return None
        displayed_img_actual_width = int(self.original_video_width * scale_for_display)
        displayed_img_actual_height = int(self.original_video_height * scale_for_display)
        if displayed_img_actual_width <= 0 or displayed_img_actual_height <=0: return None
        pad_x = (label_display_width - displayed_img_actual_width) / 2
        pad_y = (label_display_height - displayed_img_actual_height) / 2
        click_x_on_img = event_x - pad_x
        click_y_on_img = event_y - pad_y
        if not (0 <= click_x_on_img < displayed_img_actual_width and \
                0 <= click_y_on_img < displayed_img_actual_height):
            return None
        return click_x_on_img, click_y_on_img, displayed_img_actual_width, displayed_img_actual_height

    def _convert_coords_display_to_processed(self, display_x, display_y, displayed_img_actual_width, displayed_img_actual_height):
        if displayed_img_actual_width == 0 or displayed_img_actual_height == 0 or \
           self.original_video_width == 0 or self.original_video_height == 0: return None
        original_x = (display_x / displayed_img_actual_width) * self.original_video_width
        original_y = (display_y / displayed_img_actual_height) * self.original_video_height
        processed_x = int(original_x * (self.PROCESSING_WIDTH / self.original_video_width))
        processed_y = int(original_y * (self.PROCESSING_HEIGHT / self.original_video_height))
        return processed_x, processed_y

    def _get_bbox_handle_rects_display(self): 
        if self.current_active_bbox_processed is None or self.original_video_width == 0 or self.original_video_height == 0: return {}
        x_proc, y_proc, w_proc, h_proc = self.current_active_bbox_processed
        corners_proc = {'tl':(x_proc,y_proc),'tr':(x_proc+w_proc,y_proc),'bl':(x_proc,y_proc+h_proc),'br':(x_proc+w_proc,y_proc+h_proc)}
        label_display_width = self.video_label.winfo_width()
        label_display_height = self.video_label.winfo_height()
        if label_display_width <= 1 or label_display_height <= 1: return {}
        scale_w_orig_to_disp = label_display_width / self.original_video_width
        scale_h_orig_to_disp = label_display_height / self.original_video_height
        scale_orig_to_disp = min(scale_w_orig_to_disp, scale_h_orig_to_disp)
        if scale_orig_to_disp <= 0: return {}
        displayed_img_w = int(self.original_video_width * scale_orig_to_disp)
        displayed_img_h = int(self.original_video_height * scale_orig_to_disp)
        pad_x = (label_display_width - displayed_img_w) / 2
        pad_y = (label_display_height - displayed_img_h) / 2
        handle_rects_display = {}
        for key, (px, py) in corners_proc.items():
            orig_x = px * (self.original_video_width / self.PROCESSING_WIDTH)
            orig_y = py * (self.original_video_height / self.PROCESSING_HEIGHT)
            disp_x = (orig_x * scale_orig_to_disp) + pad_x
            disp_y = (orig_y * scale_orig_to_disp) + pad_y
            handle_rects_display[key] = (disp_x - self.HANDLE_SIZE // 2, disp_y - self.HANDLE_SIZE // 2, self.HANDLE_SIZE, self.HANDLE_SIZE)
        return handle_rects_display

    # --- Tab-Specific Event Handlers & Logic ---
    def on_mouse_press(self, event):
        if not self.is_video_loaded: return
        click_info = self._get_click_coords_on_displayed_image(event.x, event.y)
        if click_info is None: self.dragging_handle = None; return

        if self.current_active_bbox_processed: 
            handle_rects = self._get_bbox_handle_rects_display()
            for handle_name, (rx, ry, rw, rh) in handle_rects.items():
                if rx <= event.x < rx + rw and ry <= event.y < ry + rh:
                    self.dragging_handle = handle_name
                    self.drag_start_mouse_pos = (event.x, event.y)
                    self.drag_start_bbox_processed = list(self.current_active_bbox_processed)
                    return 
        
        self.dragging_handle = None
        if self.selected_label_id_from_list is None:
            self.run_segmentation(event) 
        else:
            print("LabelingTab: 기존 라벨 편집 모드 (클릭으로 새 SAM 실행 안함)")

    def on_mouse_drag(self, event):
        if self.dragging_handle and self.is_video_loaded and self.drag_start_bbox_processed:
            click_info = self._get_click_coords_on_displayed_image(event.x, event.y)
            if click_info is None: return
            current_click_on_img_x, current_click_on_img_y, disp_img_w, disp_img_h = click_info
            current_mouse_proc_x, current_mouse_proc_y = self._convert_coords_display_to_processed(
                current_click_on_img_x, current_click_on_img_y, disp_img_w, disp_img_h
            )
            if current_mouse_proc_x is None: return

            x, y, w, h = self.drag_start_bbox_processed
            new_x, new_y, new_w, new_h = x, y, w, h

            if self.dragging_handle == 'tl':
                new_x=min(max(0,current_mouse_proc_x),x+w-1); new_y=min(max(0,current_mouse_proc_y),y+h-1)
                new_w=(x+w)-new_x; new_h=(y+h)-new_y
            elif self.dragging_handle == 'br':
                new_w=max(1,current_mouse_proc_x-x); new_h=max(1,current_mouse_proc_y-y)
            elif self.dragging_handle == 'tr':
                new_y=min(max(0,current_mouse_proc_y),y+h-1); new_w=max(1,current_mouse_proc_x-x)
                new_h=(y+h)-new_y
            elif self.dragging_handle == 'bl':
                new_x=min(max(0,current_mouse_proc_x),x+w-1); new_w=(x+w)-new_x
                new_h=max(1,current_mouse_proc_y-y)

            new_x=max(0,min(new_x,self.PROCESSING_WIDTH-1)); new_y=max(0,min(new_y,self.PROCESSING_HEIGHT-1))
            new_w=max(1,min(new_w,self.PROCESSING_WIDTH-new_x)); new_h=max(1,min(new_h,self.PROCESSING_HEIGHT-new_y))

            if new_w > 0 and new_h > 0 : 
                self.current_active_bbox_processed = [new_x, new_y, new_w, new_h]
                self.current_active_mask_processed = None 
                self.current_active_polygons_processed = []    
                self.show_frame_and_labels(self.current_frame_number) 
            else: 
                self.current_active_bbox_processed = list(self.drag_start_bbox_processed)
    
    def on_mouse_release(self, event):
        if self.dragging_handle:
            if self.selected_label_id_from_list and self.current_active_bbox_processed:
                self.set_add_update_button_text("선택된 라벨 업데이트") 
            elif self.current_active_bbox_processed:
                 self.set_add_update_button_text("새 라벨 추가")
        self.dragging_handle = None
        self.drag_start_mouse_pos = None
        self.drag_start_bbox_processed = None
        self.update_ui_state()

    def on_right_click_clear_active_bbox(self, event):
        if self.is_video_loaded:
            self.clear_active_segmentation_data_internally() 
            self.show_frame_and_labels(self.current_frame_number)

    def run_segmentation(self, event): 
        if not self.is_video_loaded or not self.segmentation_processor.is_model_loaded() or \
           self.original_video_width == 0 or self.original_video_height == 0:
            if self.is_video_loaded and not self.segmentation_processor.is_model_loaded():
                 messagebox.showwarning("SAM 모델 오류", "SAM 모델이 로드되지 않았습니다.")
            return

        click_info = self._get_click_coords_on_displayed_image(event.x, event.y)
        if click_info is None: return
        click_x_on_img, click_y_on_img, disp_img_w, disp_img_h = click_info
        
        sam_input_click_x, sam_input_click_y = self._convert_coords_display_to_processed(
            click_x_on_img, click_y_on_img, disp_img_w, disp_img_h
        )
        if sam_input_click_x is None : return

        processed_bgr_frame = self.get_current_frame_resized() 
        if processed_bgr_frame is None: messagebox.showerror("오류", "SAM 처리를 위한 프레임을 가져올 수 없습니다."); return
        
        self.clear_active_segmentation_data_internally() 

        mask, polys, bbox = self.segmentation_processor.process_image(
            processed_bgr_frame, [sam_input_click_x, sam_input_click_y]
        )
        self.current_active_mask_processed = mask
        self.current_active_polygons_processed = polys if polys else [] 
        self.current_active_bbox_processed = bbox
        
        self.selected_label_id_from_list = None 
        self.clear_listbox_selection()
        self.set_add_update_button_text("새 라벨 추가")

        self.show_frame_and_labels(self.current_frame_number)

    def clear_active_segmentation_data_internally(self): 
        self.current_active_mask_processed = None
        self.current_active_polygons_processed = [] 
        self.current_active_bbox_processed = None

    def clear_active_segmentation_data_ui_action(self): 
        self.clear_active_segmentation_data_internally()
        self.selected_label_id_from_list = None 
        self.clear_listbox_selection()
        self.set_add_update_button_text("새 라벨 추가")
        self.show_frame_and_labels(self.current_frame_number)

    def add_or_update_label(self):
        if not self.current_active_bbox_processed:
            messagebox.showwarning("라벨 추가 오류", "추가할 바운딩 박스가 없습니다.")
            return
        
        selected_class_name = self.class_var.get()
        if not selected_class_name:
            messagebox.showwarning("라벨 추가 오류", "클래스를 선택해주세요.")
            return
        
        selected_class_id = self.app.class_ids.get(selected_class_name)
        if selected_class_id is None: 
            messagebox.showerror("오류", "선택된 클래스 ID를 찾을 수 없습니다.")
            return

        current_label_manager = self.get_current_label_manager()

        if self.selected_label_id_from_list: 
            success = current_label_manager.update_label(
                self.selected_label_id_from_list,
                selected_class_id,
                selected_class_name,
                self.current_active_bbox_processed
            )
            if not success: messagebox.showerror("오류", "라벨 업데이트에 실패했습니다.")
        else: 
            new_id = current_label_manager.add_label(
                selected_class_id,
                selected_class_name,
                self.current_active_bbox_processed
            )
            if not new_id: messagebox.showerror("오류", "라벨 추가에 실패했습니다.")
        
        self.clear_active_segmentation_data_internally()
        self.selected_label_id_from_list = None
        self.clear_listbox_selection()
        self.set_add_update_button_text("새 라벨 추가")
        self.show_frame_and_labels(self.current_frame_number) 

    def on_label_select_from_list(self, event):
        if not self.labels_listbox.curselection(): 
            self.selected_label_id_from_list = None
            self.set_add_update_button_text("새 라벨 추가")
            self.update_ui_state()
            return

        selected_index = self.labels_listbox.curselection()[0]
        current_label_manager = self.get_current_label_manager()
        frame_labels = current_label_manager.get_labels()

        if 0 <= selected_index < len(frame_labels):
            selected_label = frame_labels[selected_index]
            self.selected_label_id_from_list = selected_label.get('id')
            
            self.current_active_bbox_processed = list(selected_label['bbox_processed']) 
            self.current_active_mask_processed = None 
            self.current_active_polygons_processed = [] 
            
            self.set_class_combobox(selected_label['class_name'])
            self.set_add_update_button_text("선택된 라벨 업데이트")
            
            self.show_frame_and_labels(self.current_frame_number) 
        self.update_ui_state()

    def delete_selected_label(self):
        if not self.selected_label_id_from_list:
            messagebox.showwarning("삭제 오류", "삭제할 라벨이 리스트에서 선택되지 않았습니다.")
            return

        current_label_manager = self.get_current_label_manager()
        success = current_label_manager.delete_label(self.selected_label_id_from_list)

        if success:
            self.clear_active_segmentation_data_internally()
            self.selected_label_id_from_list = None
            self.set_add_update_button_text("새 라벨 추가")
            self.show_frame_and_labels(self.current_frame_number) 
        else:
            messagebox.showerror("오류", "라벨 삭제에 실패했습니다.")
        self.update_ui_state()

    # --- UI Update Methods (called by app_controller or self) ---
    def update_ui_state(self): 
        video_loaded = self.is_video_loaded
        self.btn_open.config(state=tk.NORMAL) # Open button is always active
        self.btn_prev_frame.config(state=tk.NORMAL if video_loaded and self.current_frame_number > 0 else tk.DISABLED)
        self.btn_next_frame.config(state=tk.NORMAL if video_loaded and self.current_frame_number < self.total_frames - 1 else tk.DISABLED)
        self.btn_goto_frame.config(state=tk.NORMAL if video_loaded else tk.DISABLED)
        self.entry_goto_frame.config(state=tk.NORMAL if video_loaded else tk.DISABLED)

        can_add_or_update = video_loaded and self.current_active_bbox_processed is not None
        self.btn_add_label.config(state=tk.NORMAL if can_add_or_update else tk.DISABLED)
        
        can_clear_active = video_loaded and (self.current_active_bbox_processed is not None or self.current_active_mask_processed is not None)
        self.btn_clear_active.config(state=tk.NORMAL if can_clear_active else tk.DISABLED)
        
        can_delete_selected = video_loaded and self.selected_label_id_from_list is not None
        self.btn_delete_label.config(state=tk.NORMAL if can_delete_selected else tk.DISABLED)
        
        self.class_combobox.config(state="readonly" if video_loaded else tk.DISABLED, values=list(self.app.class_names.values()))
        if video_loaded and not self.class_var.get() and self.app.class_names: 
             self.class_combobox.current(0)

        self.entry_filename_prefix.config(state=tk.NORMAL if video_loaded else tk.DISABLED)
        self.btn_save_labels.config(state=tk.NORMAL if video_loaded else tk.DISABLED)

    def update_frame_info(self, frame_num_text, file_name_text):
        self.lbl_frame_info.config(text=frame_num_text)
        self.lbl_file_name.config(text=file_name_text)

    def update_video_display(self, photo_image):
        self.video_label.config(image=photo_image)
        self.photo_image = photo_image 

    def update_labels_listbox_display(self, labels_data, selected_id):
        self.labels_listbox.delete(0, tk.END)
        for i, label in enumerate(labels_data):
            bbox_str = f"Box({label['bbox_processed'][0]},{label['bbox_processed'][1]}..)"
            display_text = f"{i+1}. {label['class_name']} - {bbox_str}"
            self.labels_listbox.insert(tk.END, display_text)
            if label.get('id') == selected_id:
                self.labels_listbox.selection_set(i)
                self.labels_listbox.activate(i)
    
    def set_class_combobox(self, class_name):
        self.class_var.set(class_name)

    def get_filename_prefix(self):
        return self.filename_prefix_var.get()

    def set_add_update_button_text(self, text):
        self.btn_add_label.config(text=text)

    def clear_goto_entry(self):
        self.entry_goto_frame.delete(0, tk.END)

    def focus_prefix_entry(self):
        self.entry_filename_prefix.focus()

    def clear_listbox_selection(self):
        self.labels_listbox.selection_clear(0, tk.END)
        self.selected_label_id_from_list = None 
        self.set_add_update_button_text("새 라벨 추가")

    # --- Global Application Actions (called by Tab UI or self) ---
    def open_file(self):
        filepath = filedialog.askopenfilename(title="MP4 비디오 파일을 선택하세요", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))
        if not filepath: return
        if self.cap: self.cap.release()
        
        self.frame_label_managers = {} 
        if hasattr(self, 'clear_active_segmentation_data_internally'): 
            self.clear_active_segmentation_data_internally() 
            self.selected_label_id_from_list = None # Reset selection in tab
            self.clear_listbox_selection()


        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened(): messagebox.showerror("오류", f"파일을 열 수 없습니다: {filepath}"); self.cap = None; return
        self.video_path = filepath
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps == 0: self.video_fps = 30.0
        if self.original_video_width == 0 or self.original_video_height == 0:
            messagebox.showerror("오류", "비디오 해상도 정보를 가져올 수 없습니다."); self.cap.release(); self.cap = None; return
        self.current_frame_number = 0; self.is_video_loaded = True
        
        self.show_frame_and_labels(self.current_frame_number) 
        self.update_ui_state() # Update UI state after loading video
        
    def get_current_label_manager(self): 
        if self.current_frame_number not in self.frame_label_managers:
            self.frame_label_managers[self.current_frame_number] = LabelManager()
        return self.frame_label_managers[self.current_frame_number]

    def show_frame_and_labels(self, frame_number): 
        if not self.is_video_loaded or not self.cap:
            if hasattr(self, 'video_label'): 
                black_img = Image.new('RGB', (self.video_label.winfo_width() if self.video_label.winfo_width() > 1 else 300, 
                                            self.video_label.winfo_height() if self.video_label.winfo_height() > 1 else 200), color='black')
                photo_img = ImageTk.PhotoImage(image=black_img)
                self.update_video_display(photo_img)
            return

        if not (0 <= frame_number < self.total_frames):
            if frame_number < 0: self.current_frame_number = 0
            elif frame_number >= self.total_frames: self.current_frame_number = self.total_frames - 1 if self.total_frames > 0 else 0
        else: self.current_frame_number = frame_number
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, original_frame_bgr = self.cap.read()
        if not ret: self.is_video_loaded = False; self.update_ui_state_all_tabs(); return
        
        display_frame = original_frame_bgr.copy()
        
        current_label_manager = self.get_current_label_manager()
        frame_labels = current_label_manager.get_labels()

        # Draw saved labels for the current frame
        for label in frame_labels:
            if hasattr(self, 'selected_label_id_from_list') and \
               label['id'] == self.selected_label_id_from_list and \
               hasattr(self, 'current_active_bbox_processed') and \
               self.current_active_bbox_processed:
                continue 
            
            x_p, y_p, w_p, h_p = label['bbox_processed']
            ox = int(x_p * (self.original_video_width / self.PROCESSING_WIDTH))
            oy = int(y_p * (self.original_video_height / self.PROCESSING_HEIGHT))
            ow = int(w_p * (self.original_video_width / self.PROCESSING_WIDTH))
            oh = int(h_p * (self.original_video_height / self.PROCESSING_HEIGHT))
            cv2.rectangle(display_frame, (ox, oy), (ox + ow, oy + oh), self.SAVED_BBOX_COLOR, 2)
            cv2.putText(display_frame, label['class_name'], (ox, oy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.SAVED_BBOX_COLOR, 1)

        # Draw active (currently being worked on) elements from LabelingTabUI's state
        if hasattr(self, 'current_active_bbox_processed') and self.current_active_bbox_processed is not None:
            if self.current_active_polygons_processed: 
                 for poly_processed in self.current_active_polygons_processed:
                    scaled_poly = poly_processed.astype(np.float32)
                    scaled_poly[:, 0, 0] = scaled_poly[:, 0, 0] * (self.original_video_width / self.PROCESSING_WIDTH)
                    scaled_poly[:, 0, 1] = scaled_poly[:, 0, 1] * (self.original_video_height / self.PROCESSING_HEIGHT)
                    cv2.drawContours(display_frame, [scaled_poly.astype(np.int32)], -1, (0, 0, 255), 2) 

            x_proc, y_proc, w_proc, h_proc = self.current_active_bbox_processed
            orig_x = int(x_proc * (self.original_video_width / self.PROCESSING_WIDTH))
            orig_y = int(y_proc * (self.original_video_height / self.PROCESSING_HEIGHT))
            orig_w = int(w_proc * (self.original_video_width / self.PROCESSING_WIDTH))
            orig_h = int(h_proc * (self.original_video_height / self.PROCESSING_HEIGHT))
            cv2.rectangle(display_frame, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), self.ACTIVE_BBOX_COLOR, 2)
            
            selected_class_name = self.class_var.get() if hasattr(self, 'class_var') else ""
            if selected_class_name:
                 cv2.putText(display_frame, selected_class_name, (orig_x, orig_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ACTIVE_BBOX_COLOR, 1)

            handle_coords_orig = {
                'tl':(orig_x,orig_y),'tr':(orig_x+orig_w,orig_y),'bl':(orig_x,orig_y+orig_h),'br':(orig_x+orig_w,orig_y+orig_h)
            }
            for _, (hx, hy) in handle_coords_orig.items():
                cv2.rectangle(display_frame,(hx-self.HANDLE_SIZE//2,hy-self.HANDLE_SIZE//2),(hx+self.HANDLE_SIZE//2,hy+self.HANDLE_SIZE//2),self.HANDLE_COLOR,-1)
        
        elif hasattr(self, 'current_active_mask_processed') and self.current_active_mask_processed is not None: 
            mask_original_res = cv2.resize(self.current_active_mask_processed.astype(np.uint8),
                                           (self.original_video_width, self.original_video_height),
                                           interpolation=cv2.INTER_NEAREST)
            color_mask_overlay = np.zeros_like(original_frame_bgr, dtype=np.uint8)
            color_mask_overlay[mask_original_res == 1] = (0, 255, 0) 
            cv2.addWeighted(color_mask_overlay, 0.3, display_frame, 0.7, 0, dst=display_frame)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        if hasattr(self, 'video_label'):
            video_label_widget = self.video_label
            label_width = video_label_widget.winfo_width(); label_height = video_label_widget.winfo_height()
            if label_width > 1 and label_height > 1 : 
                img_width, img_height = pil_image.size
                scale_w = label_width / img_width; scale_h = label_height / img_height
                scale = min(scale_w, scale_h) if img_width > 0 and img_height > 0 else 1.0
                new_width = int(img_width * scale); new_height = int(img_height * scale)
                if new_width > 0 and new_height > 0: pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Store the PhotoImage reference in AutoLabelerApp to prevent garbage collection in LabelingTabUI
            self.photo_image_labeling_tab = ImageTk.PhotoImage(image=pil_image) 
            self.update_video_display(self.photo_image_labeling_tab) 
        
        
            self.update_frame_info(
                f"프레임: {self.current_frame_number} / {self.total_frames -1 if self.total_frames > 0 else 0} (FPS: {self.video_fps:.2f})",
                f"파일: {os.path.basename(self.video_path) if self.video_path else '없음'}"
            )
            self.update_labels_listbox_display(frame_labels, self.selected_label_id_from_list) 
            
        #self.update_ui_state_all_tabs()
        self.update_ui_state() # Update the state of the current tab UI

    def save_current_frame_and_labels_ui(self): # This method is now correctly defined
        if not self.is_video_loaded:
            messagebox.showwarning("저장 오류", "비디오가 로드되지 않았습니다.")
            return

        current_label_manager = self.get_current_label_manager()
        
        prefix = self.get_filename_prefix() 
        if not prefix:
            messagebox.showwarning("저장 오류", "파일 이름 접두사를 입력해주세요.")
            self.focus_prefix_entry()
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, original_frame = self.cap.read()
        if not ret:
            messagebox.showerror("저장 오류", "현재 프레임의 이미지를 읽을 수 없습니다.")
            return

        yolo_strings = current_label_manager.to_yolo_format_strings(self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT)
        
        if not yolo_strings: 
             if not messagebox.askyesno("저장 확인", "현재 프레임에 라벨이 없습니다. 빈 라벨 파일과 이미지를 저장하시겠습니까?"):
                return

        success, img_path, txt_path, message = self.app.data_manager.save_image_and_yolo_labels(
            original_frame, yolo_strings, prefix, f"{self.current_frame_number:05d}"
        )
        if success:
            messagebox.showinfo("저장 완료", f"{message}\n이미지: {img_path}\n라벨: {txt_path}")
            
        else:
            messagebox.showerror("저장 실패", message)

    def get_current_frame_resized(self):
        if not self.is_video_loaded or not self.cap: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame_bgr = self.cap.read()
        if not ret: return None
        return cv2.resize(frame_bgr, (self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT))
   
    def next_frame(self):
        if self.is_video_loaded and self.current_frame_number < self.total_frames - 1:
            if hasattr(self, 'clear_active_segmentation_data_internally'):
                self.clear_active_segmentation_data_internally()
                self.selected_label_id_from_list = None 
                self.clear_listbox_selection()
            self.show_frame_and_labels(self.current_frame_number + 1)

    def prev_frame(self):
        if self.is_video_loaded and self.current_frame_number > 0:
            if hasattr(self, 'clear_active_segmentation_data_internally'):
                self.clear_active_segmentation_data_internally()
                self.selected_label_id_from_list = None
                self.clear_listbox_selection()
            self.show_frame_and_labels(self.current_frame_number - 1)
            
    def go_to_frame(self): 
        if not self.is_video_loaded: return
        if not hasattr(self, 'entry_goto_frame'): return 
        try:
            target_frame_str = self.entry_goto_frame.get()
            if not target_frame_str: return 
            target_frame = int(target_frame_str)
            if 0 <= target_frame < self.total_frames:
                if hasattr(self, 'clear_active_segmentation_data_internally'):
                    self.clear_active_segmentation_data_internally()
                    self.selected_label_id_from_list = None
                    self.clear_listbox_selection()
                self.show_frame_and_labels(target_frame)
            else:
                messagebox.showwarning("입력 오류", f"프레임 번호는 0에서 {self.total_frames - 1} 사이여야 합니다.")
        except ValueError:
            messagebox.showerror("입력 오류", "유효한 숫자를 입력하세요.")
        finally:
            if hasattr(self, 'entry_goto_frame'): self.clear_goto_entry()
            
    def show_about(self):
        messagebox.showinfo("자동 라벨링 앱 정보",
                            f"자동 라벨링 앱 v2.1 (모듈화, {self.PROCESSING_WIDTH}x{self.PROCESSING_HEIGHT} 처리)\n\n"
                            "OpenCV, Tkinter, Ultralytics SAM, PyYAML을 사용하여 제작되었습니다.\n"
                            "비디오 프레임의 객체에 대해 BBox 라벨링 및 YOLO 형식 저장을 지원하며, 저장된 데이터를 관리합니다.")

    def go_to_frame_event(self, event=None): 
        self.go_to_frame()