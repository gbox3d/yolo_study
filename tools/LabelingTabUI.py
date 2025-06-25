import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

import threading

from SegmentationProcessor import SegmentationProcessor
from LabelManager import LabelManager 
from VideoFrameRenderer import VideoFrameRenderer

class LabelingTab:
    
    # Constants
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 640
    HANDLE_SIZE = 8 
    HANDLE_COLOR = (255, 165, 0) # BGR
    ACTIVE_BBOX_COLOR = (0, 255, 255) # BGR
    SAVED_BBOX_COLOR = (255, 0, 255)   # BGR
    
    def __init__(self, parent_tab_frame, app_controller):
        self.parent_tab = parent_tab_frame
        self.app = app_controller
        
        # Initialize processors
        self.segmentation_processor = SegmentationProcessor(model_path="sam2_s.pt")
        self.frame_renderer = VideoFrameRenderer(self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT)
        
        if not self.segmentation_processor.is_model_loaded():
            messagebox.showwarning("모델 로드 실패", "SAM 모델 로드에 실패했습니다. 세그멘테이션 기능이 제한될 수 있습니다.")
        
        # Video state
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.current_frame_number = 0
        self.video_fps = 0
        self.is_video_loaded = False

        # Padding state (for aspect ratio preservation)
        self.current_scale = 1.0
        self.current_pad_x = 0
        self.current_pad_y = 0
        self.current_resized_w = 640
        self.current_resized_h = 640

        # Labeling state
        self.photo_image = None 
        self.current_active_mask_processed = None 
        self.current_active_polygons_processed = [] 
        self.current_active_bbox_processed = None   
        self.selected_label_id_from_list = None 
        self.dragging_handle = None
        self.drag_start_mouse_pos = None
        self.drag_start_bbox_processed = None
        
        # UI 요소
        self.wait_message_label = None 
        

        self._setup_ui()

    def _setup_ui(self):
        
        """UI 구성"""
        top_frame = ttk.Frame(self.parent_tab)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # 좌측: 비디오 영역 (고정 크기)
        left_container = ttk.Frame(top_frame)
        left_container.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_frame_container = ttk.LabelFrame(left_container, text="비디오 (라벨링)", width=650, height=650)
        self.video_frame_container.pack()
        self.video_frame_container.pack_propagate(False)

        self.video_label = ttk.Label(self.video_frame_container, background="black", anchor=tk.CENTER)
        self.video_label.place(x=5, y=5, width=640, height=640)
        
        # 마우스 이벤트 바인딩
        self.video_label.bind("<ButtonPress-1>", self.on_mouse_press)
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.video_label.bind("<Button-3>", self.on_right_click_clear_active_bbox)

        # 우측: 스크롤 가능한 컨트롤 패널
        control_container = ttk.Frame(top_frame, width=450)
        control_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)
        control_container.pack_propagate(False)

        # 스크롤바가 있는 Canvas 생성
        self.control_canvas = tk.Canvas(control_container, highlightthickness=0)
        self.control_scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)

        # 스크롤바와 캔버스 배치
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 스크롤 가능한 프레임 생성
        self.scrollable_frame = ttk.Frame(self.control_canvas)
        self.canvas_window = self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # 스크롤 영역 업데이트 함수
        def configure_scroll_region(event=None):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
            canvas_width = self.control_canvas.winfo_width()
            if canvas_width > 1:
                self.control_canvas.itemconfig(self.canvas_window, width=canvas_width)

        def configure_canvas_width(event):
            canvas_width = event.width
            self.control_canvas.itemconfig(self.canvas_window, width=canvas_width)

        # 이벤트 바인딩
        self.scrollable_frame.bind("<Configure>", configure_scroll_region)
        self.control_canvas.bind("<Configure>", configure_canvas_width)

        # 마우스 휠 스크롤 지원
        def on_mousewheel(event):
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def bind_mousewheel(event):
            self.control_canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            self.control_canvas.unbind_all("<MouseWheel>")

        self.control_canvas.bind('<Enter>', bind_mousewheel)
        self.control_canvas.bind('<Leave>', unbind_mousewheel)
        
        # 컨트롤 패널 내용을 스크롤 가능한 프레임에 추가
        self._setup_control_panels(self.scrollable_frame)
        
        self.wait_message_label = ttk.Label(self.video_label, text="처리 중...",
            font=("TkDefaultFont", 16, "bold"),
            background="grey", foreground="white",
            anchor=tk.CENTER, relief=tk.RAISED)
        
        
        # 핫키 설정
        self._setup_hotkeys()

    def _setup_control_panels(self, parent):
        """컨트롤 패널 구성"""
        # 파일 정보
        file_info_frame = ttk.LabelFrame(parent, text="파일 & 정보", padding="8")
        file_info_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.btn_open = ttk.Button(file_info_frame, text="파일 열기", command=self.open_file)
        self.btn_open.pack(fill=tk.X, pady=(0, 5))
        
        self.lbl_file_name = ttk.Label(file_info_frame, text="파일: 없음", wraplength=480)
        self.lbl_file_name.pack(fill=tk.X, pady=(0, 3))
        
        self.lbl_frame_info = ttk.Label(file_info_frame, text="프레임: N/A / N/A (FPS: N/A)")
        self.lbl_frame_info.pack(fill=tk.X, pady=(0, 3))
        
        # 핫키 힌트 추가 (자동 세그멘테이션 기능 포함)
        hotkey_hint = ttk.Label(file_info_frame, text="핫키: ← → (프레임), ↑↓ (10프레임), Space, Del, Esc, Ctrl+S\nShift+← → (자동 세그멘테이션)", 
                               font=("TkDefaultFont", 8), foreground="gray")
        hotkey_hint.pack(fill=tk.X)
        
        # 프레임 네비게이션
        nav_frame = ttk.LabelFrame(parent, text="프레임 이동", padding="8")
        nav_frame.pack(fill=tk.X, pady=(0, 8))
        
        frame_nav_buttons_frame = ttk.Frame(nav_frame)
        frame_nav_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_prev_frame = ttk.Button(frame_nav_buttons_frame, text="<< 이전", command=self.prev_frame, state=tk.DISABLED)
        self.btn_prev_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        
        self.btn_next_frame = ttk.Button(frame_nav_buttons_frame, text="다음 >>", command=self.next_frame, state=tk.DISABLED)
        self.btn_next_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(3, 0))
        
        goto_frame_container = ttk.Frame(nav_frame)
        goto_frame_container.pack(fill=tk.X)
        
        ttk.Label(goto_frame_container, text="프레임 이동:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.entry_goto_frame = ttk.Entry(goto_frame_container, width=10)
        self.entry_goto_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.entry_goto_frame.bind("<Return>", self.go_to_frame_event)
        
        self.btn_goto_frame = ttk.Button(goto_frame_container, text="이동", command=self.go_to_frame, state=tk.DISABLED, width=8)
        self.btn_goto_frame.pack(side=tk.RIGHT)

        # 라벨링 작업
        labeling_ops_frame = ttk.LabelFrame(parent, text="라벨링 작업", padding="8")
        labeling_ops_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(labeling_ops_frame, text="클래스 선택:").pack(anchor=tk.W, pady=(0, 3))
        
        self.class_var = tk.StringVar()
        self.class_combobox = ttk.Combobox(labeling_ops_frame, textvariable=self.class_var, 
                                           values=list(self.app.class_names.values()), state="readonly")
        if self.app.class_names: 
            self.class_combobox.current(0)
        self.class_combobox.pack(fill=tk.X, pady=(0, 8))
        
        self.btn_add_label = ttk.Button(labeling_ops_frame, text="라벨 추가/업데이트", command=self.add_or_update_label, state=tk.DISABLED)
        self.btn_add_label.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_clear_active = ttk.Button(labeling_ops_frame, text="활성 세그멘테이션 지우기", command=self.clear_active_segmentation_data_ui_action, state=tk.DISABLED)
        self.btn_clear_active.pack(fill=tk.X)

        # 라벨 목록 (더 큰 공간 할당)
        labels_list_frame = ttk.LabelFrame(parent, text="현재 프레임 라벨 목록", padding="8")
        labels_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        self.labels_listbox = tk.Listbox(labels_list_frame, height=10)
        self.labels_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.labels_listbox.bind("<<ListboxSelect>>", self.on_label_select_from_list)
        
        self.btn_delete_label = ttk.Button(labels_list_frame, text="선택된 라벨 삭제", command=self.delete_selected_label, state=tk.DISABLED)
        self.btn_delete_label.pack(fill=tk.X)

        # 저장
        save_frame = ttk.LabelFrame(parent, text="저장", padding="8")
        save_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(save_frame, text="파일 이름 접두사:").pack(anchor=tk.W, pady=(0, 3))
        
        self.filename_prefix_var = tk.StringVar(value="frame")
        self.entry_filename_prefix = ttk.Entry(save_frame, textvariable=self.filename_prefix_var)
        self.entry_filename_prefix.pack(fill=tk.X, pady=(0, 5))
        
        # 현재 프레임 저장
        self.btn_save_labels = ttk.Button(save_frame, text="현재 프레임 라벨 저장", command=self.save_current_frame_and_labels_ui, state=tk.DISABLED)
        self.btn_save_labels.pack(fill=tk.X, pady=(0, 3))
        
        # 전체 관리 버튼들
        batch_frame = ttk.Frame(save_frame)
        batch_frame.pack(fill=tk.X, pady=(3, 0))
        
        self.btn_save_all_labels = ttk.Button(batch_frame, text="전체 라벨 JSON 저장", command=self.save_all_labels_to_json, state=tk.DISABLED)
        self.btn_save_all_labels.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        self.btn_load_all_labels = ttk.Button(batch_frame, text="전체 라벨 JSON 불러오기", command=self.load_all_labels_from_json, state=tk.DISABLED)
        self.btn_load_all_labels.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))
        
        # 일괄 내보내기 버튼들
        export_frame = ttk.Frame(save_frame)
        export_frame.pack(fill=tk.X, pady=(3, 0))
        
        self.btn_export_yolo_batch = ttk.Button(export_frame, text="YOLO 일괄 내보내기", command=self.export_all_frames_to_yolo, state=tk.DISABLED)
        self.btn_export_yolo_batch.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        self.btn_show_label_stats = ttk.Button(export_frame, text="라벨 통계", command=self.show_all_labels_statistics, state=tk.DISABLED)
        self.btn_show_label_stats.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))
        
        # About (하단 고정)
        self.btn_about = ttk.Button(parent, text="정보 (About)", command=self.show_about)
        self.btn_about.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))

    def _setup_hotkeys(self):
        """핫키 설정"""
        root = self.app.master
        
        # 키보드 이벤트 바인딩
        root.bind('<Key>', self.on_key_press)
        root.bind('<KeyPress-Left>', self.on_key_left)
        root.bind('<KeyPress-Right>', self.on_key_right)
        root.bind('<KeyPress-Up>', self.on_key_up)
        root.bind('<KeyPress-Down>', self.on_key_down)
        root.bind('<KeyPress-space>', self.on_key_space)
        root.bind('<KeyPress-Delete>', self.on_key_delete)
        root.bind('<KeyPress-Escape>', self.on_key_escape)
        root.bind('<Control-s>', self.on_key_save)
        
        # Shift + 좌우 방향키 (자동 세그멘테이션)
        root.bind('<Shift-Left>', self.on_shift_key_left)
        root.bind('<Shift-Right>', self.on_shift_key_right)
        
        root.focus_set()
        
        print("핫키 설정 완료:")
        print("  ← → : 이전/다음 프레임")
        print("  ↑ ↓ : 10프레임씩 이동")
        print("  Space : 다음 프레임")
        print("  Delete : 선택된 라벨 삭제")
        print("  Esc : 활성 세그멘테이션 지우기")
        print("  Ctrl+S : 현재 프레임 저장")
        print("  Shift+← → : 자동 세그멘테이션 (이전/다음 프레임 bbox 참조)")

    #--- wait message ---
    def _show_wait_message(self, message="세그멘테이션 중..."):
        if self.wait_message_label:
            self.wait_message_label.config(text=message)
            self.wait_message_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.7, relheight=0.2)
            self.wait_message_label.lift()
            self.video_label.update_idletasks() # 메시지가 즉시 보이도록 강제 업데이트

    def _hide_wait_message(self):
        if self.wait_message_label:
            self.wait_message_label.place_forget()

    # === 핫키 이벤트 처리 ===
    def on_key_press(self, event):
        """일반 키 이벤트 처리"""
        focused_widget = self.app.master.focus_get()
        if isinstance(focused_widget, (tk.Entry, tk.Text)):
            return
        if not self.is_video_loaded:
            return

    def on_key_left(self, event):
        """왼쪽 화살표: 이전 프레임"""
        if not self.is_video_loaded:
            return
        self.prev_frame()
        return "break"

    def on_key_right(self, event):
        """오른쪽 화살표: 다음 프레임"""
        if not self.is_video_loaded:
            return
        self.next_frame()
        return "break"

    def on_key_up(self, event):
        """위쪽 화살표: 10프레임 뒤로"""
        if not self.is_video_loaded:
            return
        target_frame = max(0, self.current_frame_number - 10)
        self._jump_to_frame(target_frame)
        return "break"

    def on_key_down(self, event):
        """아래쪽 화살표: 10프레임 앞으로"""
        if not self.is_video_loaded:
            return
        target_frame = min(self.total_frames - 1, self.current_frame_number + 10)
        self._jump_to_frame(target_frame)
        return "break"

    def on_key_space(self, event):
        """스페이스바: 다음 프레임"""
        if not self.is_video_loaded:
            return
        self.next_frame()
        return "break"

    def on_key_delete(self, event):
        """Delete키: 선택된 라벨 삭제"""
        if not self.is_video_loaded:
            return
        if self.selected_label_id_from_list:
            self.delete_selected_label()
        return "break"

    def on_key_escape(self, event):
        """Esc키: 활성 세그멘테이션 지우기"""
        if not self.is_video_loaded:
            return
        self.clear_active_segmentation_data_ui_action()
        return "break"

    def on_key_save(self, event):
        """Ctrl+S: 현재 프레임 저장"""
        if not self.is_video_loaded:
            return
        self.save_current_frame_and_labels_ui()
        return "break"

    def on_shift_key_left(self, event):
        """Shift+Left: 이전 프레임 bbox 참조해서 자동 세그멘테이션"""
        if not self.is_video_loaded:
            return
        self._auto_segmentation_with_reference(-1)  # 이전 프레임
        return "break"

    def on_shift_key_right(self, event):
        """Shift+Right: 다음 프레임 bbox 참조해서 자동 세그멘테이션"""
        if not self.is_video_loaded:
            return
        
        threading.Thread(target=self._auto_segmentation_with_reference, args=(1,), daemon=True).start()
        # self._auto_segmentation_with_reference(1)   # 다음 프레임
        return "break"
        
    def _auto_segmentation_with_reference(self, frame_offset):
        """
        참조 프레임의 bbox를 기반으로 자동 세그멘테이션 수행
        
        Args:
            frame_offset: 참조할 프레임의 오프셋 (-1: 이전, 1: 다음)
        """
        if not self.segmentation_processor.is_model_loaded():
            messagebox.showwarning("SAM 모델 오류", "SAM 모델이 로드되지 않았습니다.")
            return

        # 참조 프레임 번호 계산
        reference_frame = self.current_frame_number 
        self.current_frame_number = reference_frame + frame_offset
        
        # 유효한 프레임 범위 확인
        if not (0 <= self.current_frame_number < self.total_frames):
            direction_text = "이전" if frame_offset < 0 else "다음"
            messagebox.showinfo("자동 세그멘테이션", f"{direction_text} 프레임이 존재하지 않습니다.")
            self.current_frame_number = reference_frame  # 원래 프레임으로 복구
            return

        # 참조 프레임의 라벨 매니저 가져오기
        if reference_frame not in self.app.frame_label_managers:
            direction_text = "이전" if frame_offset < 0 else "다음"
            messagebox.showinfo("자동 세그멘테이션", f"{direction_text} 프레임에 라벨이 없습니다.")
            self.current_frame_number = reference_frame  # 원래 프레임으로 복구
            return

        reference_label_manager = self.app.frame_label_managers[reference_frame]
        reference_labels = reference_label_manager.get_labels()
        
        if not reference_labels:
            direction_text = "이전" if frame_offset < 0 else "다음"
            messagebox.showinfo("자동 세그멘테이션", f"{direction_text} 프레임에 라벨이 없습니다.")
            self.current_frame_number = reference_frame  # 원래 프레임으로 복구
            return

        # 현재 프레임의 처리 해상도 이미지 가져오기
        frame_processed = self.get_current_frame_processed()
        if frame_processed is None:
            messagebox.showerror("오류", "현재 프레임을 가져올 수 없습니다.")
            self.current_frame_number = reference_frame  # 원래 프레임으로 복구
            return

        # 기존 활성 세그멘테이션 데이터 초기화
        self.clear_active_segmentation_data_internally()

        # 각 참조 bbox에 대해 자동 세그멘테이션 수행
        current_label_manager = self.get_current_label_manager()
        successful_count = 0
        # total_count = len(reference_labels)

        self._show_wait_message("자동 세그멘테이션 중...")
        for ref_label in reference_labels:
            try:
                ref_bbox = ref_label['bbox_processed']
                ref_class_id = ref_label['class_id']
                ref_class_name = ref_label['class_name']
                
                # bbox 중심점 계산
                x, y, w, h = ref_bbox
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 중심점이 유효한 범위 내에 있는지 확인
                if not (0 <= center_x < self.PROCESSING_WIDTH and 0 <= center_y < self.PROCESSING_HEIGHT):
                    print(f"Warning: 참조 bbox 중심점이 범위를 벗어남: ({center_x}, {center_y})")
                    continue

                # SAM 세그멘테이션 수행
                mask, polygons, bbox = self.segmentation_processor.process_image_sync(
                    frame_processed, prompt_data=ref_bbox, prompt_type="bbox"
                )
                
                if bbox is not None:
                    # 새 라벨 추가 (기존 클래스 정보 유지)
                    new_id = current_label_manager.add_label(
                        ref_class_id, ref_class_name, bbox
                    )
                    if new_id:
                        successful_count += 1
                        print(f"자동 라벨 추가 성공: {ref_class_name} at {bbox}")
                    else:
                        print(f"라벨 추가 실패: {ref_class_name}")
                else:
                    print(f"세그멘테이션 실패: {ref_class_name} at ({center_x}, {center_y})")

            except Exception as e:
                print(f"자동 세그멘테이션 오류: {e}")
                continue

        self._hide_wait_message()
        
        # 결과 표시
        direction_text = "이전" if frame_offset < 0 else "다음"
        if successful_count > 0:
            self.show_frame_and_labels(self.current_frame_number)
            
        else:
            messagebox.showwarning(
                "자동 세그멘테이션 실패", 
                f"{direction_text} 프레임의 bbox를 참조했으나 세그멘테이션에 실패했습니다."
            )

        # UI 상태 업데이트
        self.selected_label_id_from_list = None
        self.clear_listbox_selection()
        self.set_add_update_button_text("새 라벨 추가")
        self.update_ui_state()

    def _jump_to_frame(self, target_frame):
        """특정 프레임으로 점프 (핫키용)"""
        if 0 <= target_frame < self.total_frames:
            self.clear_active_segmentation_data_internally()
            self.selected_label_id_from_list = None
            self.clear_listbox_selection()
            self.show_frame_and_labels(target_frame)

    # === 좌표 변환 ===
    def _get_click_coords_processed(self, event_x, event_y):
        """UI 클릭을 처리 해상도 좌표로 변환 (패딩 고려)"""
        if not hasattr(self, 'current_scale'):
            return None
            
        label_display_width = self.video_label.winfo_width()
        label_display_height = self.video_label.winfo_height()
        if label_display_width <= 1 or label_display_height <= 1: 
            return None
        
        # 처리 해상도가 UI에 맞게 스케일된 크기 계산
        ui_scale = min(label_display_width / self.PROCESSING_WIDTH, label_display_height / self.PROCESSING_HEIGHT)
        if ui_scale <= 0: 
            return None
        
        displayed_width = int(self.PROCESSING_WIDTH * ui_scale)
        displayed_height = int(self.PROCESSING_HEIGHT * ui_scale)
        
        # UI에서의 중앙 정렬 패딩
        ui_pad_x = (label_display_width - displayed_width) / 2
        ui_pad_y = (label_display_height - displayed_height) / 2
        
        # 클릭이 이미지 영역 내인지 확인
        click_x = event_x - ui_pad_x
        click_y = event_y - ui_pad_y
        
        if not (0 <= click_x < displayed_width and 0 <= click_y < displayed_height):
            return None
        
        # 처리 해상도 좌표로 변환
        coord_x = (click_x / displayed_width) * self.PROCESSING_WIDTH
        coord_y = (click_y / displayed_height) * self.PROCESSING_HEIGHT
        
        # 패딩 영역 클릭인지 확인
        if (coord_x < self.current_pad_x or 
            coord_x >= self.current_pad_x + self.current_resized_w or
            coord_y < self.current_pad_y or 
            coord_y >= self.current_pad_y + self.current_resized_h):
            return None  # 패딩 영역(검은 부분) 클릭
        
        return int(coord_x), int(coord_y)

    def _get_bbox_handle_rects_display(self):
        """바운딩 박스 핸들의 UI 표시 좌표 계산 (패딩 고려)"""
        if self.current_active_bbox_processed is None or not hasattr(self, 'current_scale'):
            return {}
        
        x_proc, y_proc, w_proc, h_proc = self.current_active_bbox_processed
        
        label_display_width = self.video_label.winfo_width()
        label_display_height = self.video_label.winfo_height()
        if label_display_width <= 1 or label_display_height <= 1:
            return {}
        
        # 처리 해상도가 UI에 맞게 스케일된 비율
        ui_scale = min(label_display_width / self.PROCESSING_WIDTH, label_display_height / self.PROCESSING_HEIGHT)
        if ui_scale <= 0:
            return {}
        
        displayed_width = int(self.PROCESSING_WIDTH * ui_scale)
        displayed_height = int(self.PROCESSING_HEIGHT * ui_scale)
        ui_pad_x = (label_display_width - displayed_width) / 2
        ui_pad_y = (label_display_height - displayed_height) / 2
        
        # 핸들 위치 계산
        corners = {
            'tl': (x_proc, y_proc),
            'tr': (x_proc + w_proc, y_proc),
            'bl': (x_proc, y_proc + h_proc),
            'br': (x_proc + w_proc, y_proc + h_proc)
        }
        
        handle_rects = {}
        for key, (px, py) in corners.items():
            # 처리 해상도 좌표를 UI 좌표로 변환
            disp_x = (px / self.PROCESSING_WIDTH) * displayed_width + ui_pad_x
            disp_y = (py / self.PROCESSING_HEIGHT) * displayed_height + ui_pad_y
            
            handle_rects[key] = (
                disp_x - self.HANDLE_SIZE // 2, 
                disp_y - self.HANDLE_SIZE // 2, 
                self.HANDLE_SIZE, 
                self.HANDLE_SIZE
            )
        
        return handle_rects

    # === 마우스 이벤트 ===
    def on_mouse_press(self, event):
        if not self.is_video_loaded: 
            return
        
        # 핸들 드래그 체크
        if self.current_active_bbox_processed:
            handle_rects = self._get_bbox_handle_rects_display()
            for handle_name, (rx, ry, rw, rh) in handle_rects.items():
                if rx <= event.x < rx + rw and ry <= event.y < ry + rh:
                    self.dragging_handle = handle_name
                    self.drag_start_mouse_pos = (event.x, event.y)
                    self.drag_start_bbox_processed = list(self.current_active_bbox_processed)
                    return
        
        self.dragging_handle = None
        
        # 새 세그멘테이션 실행 (기존 라벨 편집 모드가 아닐 때만)
        if self.selected_label_id_from_list is None:
            self.run_segmentation(event)

    def on_mouse_drag(self, event):
        if not (self.dragging_handle and self.is_video_loaded and self.drag_start_bbox_processed):
            return
        
        coords = self._get_click_coords_processed(event.x, event.y)
        if coords is None:
            return
        
        current_x, current_y = coords
        x, y, w, h = self.drag_start_bbox_processed
        
        # 핸들에 따른 바운딩 박스 조정
        if self.dragging_handle == 'tl':
            new_x = min(max(0, current_x), x + w - 1)
            new_y = min(max(0, current_y), y + h - 1)
            new_w = (x + w) - new_x
            new_h = (y + h) - new_y
        elif self.dragging_handle == 'br':
            new_x, new_y = x, y
            new_w = max(1, current_x - x)
            new_h = max(1, current_y - y)
        elif self.dragging_handle == 'tr':
            new_x = x
            new_y = min(max(0, current_y), y + h - 1)
            new_w = max(1, current_x - x)
            new_h = (y + h) - new_y
        elif self.dragging_handle == 'bl':
            new_x = min(max(0, current_x), x + w - 1)
            new_y = y
            new_w = (x + w) - new_x
            new_h = max(1, current_y - y)
        else:
            return
        
        # 경계값 확인
        new_x = max(0, min(new_x, self.PROCESSING_WIDTH - 1))
        new_y = max(0, min(new_y, self.PROCESSING_HEIGHT - 1))
        new_w = max(1, min(new_w, self.PROCESSING_WIDTH - new_x))
        new_h = max(1, min(new_h, self.PROCESSING_HEIGHT - new_y))
        
        if new_w > 0 and new_h > 0:
            self.current_active_bbox_processed = [new_x, new_y, new_w, new_h]
            self.current_active_mask_processed = None
            self.current_active_polygons_processed = []
            self.show_frame_and_labels(self.current_frame_number)

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


    # === 세그멘테이션 처리 ===
    def _on_segmentation_complete(self,  mask, polygons, bbox, error, call_context):
        print("callCackSegmentation called")
        
        self.current_active_mask_processed = mask
        self.current_active_polygons_processed = polygons if polygons else []
        self.current_active_bbox_processed = bbox
        
        self.selected_label_id_from_list = None
        self.clear_listbox_selection()
        self.set_add_update_button_text("새 라벨 추가")
        
        self.show_frame_and_labels(self.current_frame_number)
        
        #wait 대화창 닫기
        self._hide_wait_message()
        
        if error:
            print(f"세그멘테이션 오류: {error}")
            messagebox.showerror("세그멘테이션 오류", f"오류 발생: {error}")
        
    
    def run_segmentation(self, event):
        if not self.is_video_loaded or not self.segmentation_processor.is_model_loaded():
            if self.is_video_loaded and not self.segmentation_processor.is_model_loaded():
                messagebox.showwarning("SAM 모델 오류", "SAM 모델이 로드되지 않았습니다.")
            return

        # 클릭 좌표를 처리 해상도 기준으로 변환
        coords = self._get_click_coords_processed(event.x, event.y)
        if coords is None:
            return
        
        click_x, click_y = coords
        
        # 처리 해상도 프레임 가져오기
        frame_processed = self.get_current_frame_processed()
        if frame_processed is None:
            messagebox.showerror("오류", "SAM 처리를 위한 프레임을 가져올 수 없습니다.")
            return
        
        self.clear_active_segmentation_data_internally()
        
        # SAM 처리
        #mask, polygons, bbox = self.segmentation_processor.process_image_sync(frame_processed, [click_x, click_y])
        
        self.segmentation_processor.process_image_async(
            frame_processed, # 복사는 async 메서드 내부에서 처리
            [click_x, click_y], 
            "point",
            self._on_segmentation_complete, # 콜백 함수
            # 콜백에 전달될 컨텍스트 정보
            call_context={'type': 'single_click', 'target_frame_num': self.current_frame_number}
        )
        
        # WAIT  대화창 
        
        self._show_wait_message("세그멘테이션 중...")
        
    def clear_active_segmentation_data_internally(self):
        """활성 세그멘테이션 데이터 초기화"""
        self.current_active_mask_processed = None
        self.current_active_polygons_processed = []
        self.current_active_bbox_processed = None

    def clear_active_segmentation_data_ui_action(self):
        """UI 액션: 활성 세그멘테이션 데이터 지우기"""
        self.clear_active_segmentation_data_internally()
        self.selected_label_id_from_list = None
        self.clear_listbox_selection()
        self.set_add_update_button_text("새 라벨 추가")
        self.show_frame_and_labels(self.current_frame_number)

    # === 프레임 처리 ===
    def get_current_frame_processed(self):
        """현재 프레임을 처리 해상도에 원본 비율 유지하면서 패딩 추가"""
        if not self.is_video_loaded or not self.cap:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return self._resize_with_padding(frame, self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT)

    def _resize_with_padding(self, image, target_width, target_height):
        """렌더러를 사용한 패딩 처리"""
        padded, scale, pad_x, pad_y, resized_w, resized_h = self.frame_renderer.resize_with_padding(
            image, target_width, target_height
        )
        
        # 패딩 정보 저장 (좌표 변환에 필요)
        self.current_scale = scale
        self.current_pad_x = pad_x
        self.current_pad_y = pad_y
        self.current_resized_w = resized_w
        self.current_resized_h = resized_h
        
        return padded

    def show_frame_and_labels(self, frame_number):
        """프레임과 라벨 표시 (렌더러 사용)"""
        if not self.is_video_loaded or not self.cap:
            self._show_black_image()
            return

        # 프레임 번호 유효성 검사
        if not (0 <= frame_number < self.total_frames):
            if frame_number < 0:
                self.current_frame_number = 0
            elif frame_number >= self.total_frames:
                self.current_frame_number = max(0, self.total_frames - 1)
        else:
            self.current_frame_number = frame_number
        
        # 처리 해상도 프레임 가져오기
        frame_processed = self.get_current_frame_processed()
        if frame_processed is None:
            self.is_video_loaded = False
            self.update_ui_state()
            return
        
        # 라벨 매니저 가져오기
        current_label_manager = self.get_current_label_manager()
        frame_labels = current_label_manager.get_labels()

        # 활성 요소들 준비
        active_elements = {
            'bbox': self.current_active_bbox_processed,
            'mask': self.current_active_mask_processed,
            'polygons': self.current_active_polygons_processed,
            'selected_id': getattr(self, 'selected_label_id_from_list', None)
        }
        
        # 선택된 클래스 이름
        selected_class_name = self.class_var.get() if hasattr(self, 'class_var') else ""
        
        # 렌더러를 사용해서 프레임 렌더링
        display_frame = self.frame_renderer.render_frame_with_labels(
            frame_processed, 
            frame_labels, 
            active_elements, 
            selected_class_name
        )

        # PIL 이미지로 변환하여 표시
        self._display_image(display_frame)
        
        # UI 정보 업데이트
        self.update_frame_info(
            f"프레임: {self.current_frame_number} / {max(0, self.total_frames - 1)} (FPS: {self.video_fps:.2f})",
            f"파일: {os.path.basename(self.video_path) if self.video_path else '없음'}"
        )
        self.update_labels_listbox_display(frame_labels, self.selected_label_id_from_list)
        self.update_ui_state()

    def _show_black_image(self):
        """검은 이미지 표시"""
        if hasattr(self, 'video_label'):
            width = max(300, self.video_label.winfo_width())
            height = max(200, self.video_label.winfo_height())
            black_img = Image.new('RGB', (width, height), color='black')
            photo_img = ImageTk.PhotoImage(image=black_img)
            self.update_video_display(photo_img)

    def _display_image(self, cv_image):
        """렌더러를 사용한 이미지 표시"""
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        # 기본값 설정
        if label_width <= 1:
            label_width = 640
        if label_height <= 1:
            label_height = 640
        
        # 렌더러를 사용해서 PhotoImage 생성
        self.photo_image_labeling_tab = self.frame_renderer.frame_to_photoimage(
            cv_image, label_width, label_height
        )
        self.update_video_display(self.photo_image_labeling_tab)

    # === 라벨 관리 ===
    def get_current_label_manager(self):
        """현재 프레임의 라벨 매니저 반환"""
        if self.current_frame_number not in self.app.frame_label_managers:
            self.app.frame_label_managers[self.current_frame_number] = LabelManager()
        return self.app.frame_label_managers[self.current_frame_number]

    def add_or_update_label(self):
        """라벨 추가 또는 업데이트"""
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

        # 업데이트 또는 새로 추가
        if self.selected_label_id_from_list:
            success = current_label_manager.update_label(
                self.selected_label_id_from_list,
                selected_class_id,
                selected_class_name,
                self.current_active_bbox_processed
            )
            if not success:
                messagebox.showerror("오류", "라벨 업데이트에 실패했습니다.")
        else:
            new_id = current_label_manager.add_label(
                selected_class_id,
                selected_class_name,
                self.current_active_bbox_processed
            )
            if not new_id:
                messagebox.showerror("오류", "라벨 추가에 실패했습니다.")
        
        # 상태 초기화
        self.clear_active_segmentation_data_internally()
        self.selected_label_id_from_list = None
        self.clear_listbox_selection()
        self.set_add_update_button_text("새 라벨 추가")
        self.show_frame_and_labels(self.current_frame_number)

    def on_label_select_from_list(self, event):
        """라벨 목록에서 선택 시 처리"""
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
            
            # 선택된 라벨의 바운딩 박스를 활성화
            self.current_active_bbox_processed = list(selected_label['bbox_processed'])
            self.current_active_mask_processed = None
            self.current_active_polygons_processed = []
            
            # 클래스 콤보박스 설정
            self.set_class_combobox(selected_label['class_name'])
            self.set_add_update_button_text("선택된 라벨 업데이트")
            
            self.show_frame_and_labels(self.current_frame_number)
        
        self.update_ui_state()

    def delete_selected_label(self):
        """선택된 라벨 삭제"""
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

    # === 파일 관리 ===
    def open_file(self):
        """비디오 파일 열기"""
        filepath = filedialog.askopenfilename(
            title="비디오 파일을 선택하세요", 
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        )
        if not filepath:
            return
        
        # 기존 캡처 해제
        if self.cap:
            self.cap.release()
        
        # 상태 초기화
        self.app.frame_label_managers = {}
        self.clear_active_segmentation_data_internally()
        self.selected_label_id_from_list = None
        self.clear_listbox_selection()

        # 새 비디오 열기
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            messagebox.showerror("오류", f"파일을 열 수 없습니다: {filepath}")
            self.cap = None
            return
        
        # 비디오 정보 가져오기
        self.video_path = filepath
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps == 0:
            self.video_fps = 30.0
        
        self.current_frame_number = 0
        self.is_video_loaded = True
        
        print(f"비디오 로드 완료: {filepath}")
        print(f"총 프레임: {self.total_frames}, FPS: {self.video_fps}")
        
        self.show_frame_and_labels(self.current_frame_number)
        self.update_ui_state()

    def save_current_frame_and_labels_ui(self):
        """현재 프레임과 라벨 저장"""
        if not self.is_video_loaded:
            messagebox.showwarning("저장 오류", "비디오가 로드되지 않았습니다.")
            return

        prefix = self.get_filename_prefix()
        if not prefix:
            messagebox.showwarning("저장 오류", "파일 이름 접두사를 입력해주세요.")
            self.focus_prefix_entry()
            return

        # 처리 해상도 프레임 가져오기
        frame_processed = self.get_current_frame_processed()
        if frame_processed is None:
            messagebox.showerror("저장 오류", "현재 프레임의 이미지를 읽을 수 없습니다.")
            return

        # YOLO 형식 라벨 생성
        current_label_manager = self.get_current_label_manager()
        yolo_strings = current_label_manager.to_yolo_format_strings(self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT)
        
        if not yolo_strings:
            if not messagebox.askyesno("저장 확인", "현재 프레임에 라벨이 없습니다. 빈 라벨 파일과 이미지를 저장하시겠습니까?"):
                return

        # 저장 실행
        success, img_path, txt_path, message = self.app.data_manager.save_image_and_yolo_labels(
            frame_processed, yolo_strings, prefix, f"{self.current_frame_number:05d}"
        )
        
        if success:
            messagebox.showinfo("저장 완료", f"{message}\n이미지: {img_path}\n라벨: {txt_path}")
        else:
            messagebox.showerror("저장 실패", message)

    # === 전체 라벨 관리 기능 ===
    def save_all_labels_to_json(self):
        """전체 프레임의 라벨 데이터를 JSON 파일로 저장"""
        if not self.app.frame_label_managers:
            messagebox.showwarning("저장 오류", "저장할 라벨 데이터가 없습니다.")
            return

        filepath = filedialog.asksaveasfilename(
            title="전체 라벨 데이터 저장",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            # 전체 라벨 데이터 수집
            all_labels_data = {
                'metadata': {
                    'version': '1.0',
                    'created_at': __import__('datetime').datetime.now().isoformat(),
                    'video_info': {
                        'total_frames': self.total_frames,
                        'fps': self.video_fps,
                        'video_path': self.video_path,
                        'processing_resolution': [self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT]
                    },
                    'class_info': {
                        'class_names': self.app.class_names,
                        'class_ids': self.app.class_ids
                    }
                },
                'frame_labels': {}
            }

            # 각 프레임의 라벨 데이터 추가
            total_labels = 0
            for frame_num, label_manager in self.app.frame_label_managers.items():
                labels = label_manager.get_labels()
                if labels:  # 라벨이 있는 프레임만 저장
                    all_labels_data['frame_labels'][str(frame_num)] = labels
                    total_labels += len(labels)

            all_labels_data['metadata']['total_labeled_frames'] = len(all_labels_data['frame_labels'])
            all_labels_data['metadata']['total_labels'] = total_labels

            # JSON 파일로 저장
            import json
            import os
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_labels_data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo(
                "저장 완료", 
                f"전체 라벨 데이터가 저장되었습니다.\n"
                f"파일: {filepath}\n"
                f"라벨된 프레임: {len(all_labels_data['frame_labels'])}개\n"
                f"총 라벨 수: {total_labels}개"
            )
            
        except Exception as e:
            messagebox.showerror("저장 실패", f"라벨 데이터 저장 중 오류 발생:\n{e}")

    def load_all_labels_from_json(self):
        """JSON 파일에서 전체 라벨 데이터를 불러오기"""
        if not self.is_video_loaded:
            messagebox.showwarning("불러오기 오류", "먼저 비디오를 로드해주세요.")
            return

        # 수정 제안 코드 (실제 라벨 데이터가 하나라도 있는지 체크)
        if self.app.frame_label_managers and any(manager.get_labels() for manager in self.app.frame_label_managers.values()):
            if not messagebox.askyesno("기존 데이터 확인", "실제로 저장된 라벨 데이터가 있습니다. 덮어쓰시겠습니까?"): # 메시지 내용도 조금 더 명확하게 수정 가능
                return

        filepath = filedialog.askopenfilename(
            title="전체 라벨 데이터 불러오기",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # 데이터 구조 검증
            if 'frame_labels' not in loaded_data:
                messagebox.showerror("불러오기 오류", "올바른 라벨 데이터 파일이 아닙니다.")
                return

            # 기존 데이터 초기화
            self.app.frame_label_managers = {}
            
            # 라벨 데이터 로드
            loaded_frames = 0
            loaded_labels = 0
            
            for frame_str, frame_labels in loaded_data['frame_labels'].items():
                try:
                    frame_num = int(frame_str)
                    
                    # 유효한 프레임 범위 확인
                    if not (0 <= frame_num < self.total_frames):
                        print(f"Warning: 프레임 {frame_num}이 현재 비디오 범위를 벗어남")
                        continue

                    # 라벨 매니저 생성 및 라벨 추가
                    from LabelManager import LabelManager
                    label_manager = LabelManager()
                    
                    for label_data in frame_labels:
                        # 필수 필드 검증
                        if all(field in label_data for field in ['class_id', 'class_name', 'bbox_processed']):
                            # ID가 없으면 새로 생성
                            if 'id' not in label_data:
                                label_data['id'] = __import__('uuid').uuid4().hex
                            
                            label_manager.labels.append(label_data)
                            loaded_labels += 1
                    
                    if label_manager.get_labels():  # 라벨이 있는 경우만 추가
                        self.app.frame_label_managers[frame_num] = label_manager
                        loaded_frames += 1
                        
                except (ValueError, KeyError) as e:
                    print(f"Warning: 프레임 {frame_str} 데이터 로드 실패: {e}")
                    continue

            # 현재 프레임 새로고침
            self.clear_active_segmentation_data_internally()
            self.selected_label_id_from_list = None
            self.clear_listbox_selection()
            self.show_frame_and_labels(self.current_frame_number)

            # 결과 표시
            metadata = loaded_data.get('metadata', {})
            original_frames = metadata.get('total_labeled_frames', loaded_frames)
            original_labels = metadata.get('total_labels', loaded_labels)
            
            messagebox.showinfo(
                "불러오기 완료",
                f"라벨 데이터가 로드되었습니다.\n"
                f"로드된 프레임: {loaded_frames}개 (원본: {original_frames}개)\n"
                f"로드된 라벨: {loaded_labels}개 (원본: {original_labels}개)"
            )
            
        except Exception as e:
            messagebox.showerror("불러오기 실패", f"라벨 데이터 불러오기 중 오류 발생:\n{e}")

    def export_all_frames_to_yolo(self):
        """전체 프레임을 YOLO 형식으로 일괄 내보내기 (라벨이 있는 프레임만)"""
        if not self.app.frame_label_managers:
            messagebox.showwarning("내보내기 오류", "내보낼 라벨 데이터가 (전체적으로) 없습니다.")
            return

        # 실제로 라벨이 있는 프레임만 필터링
        labeled_frame_numbers = []
        for frame_num, manager in self.app.frame_label_managers.items():
            if manager and manager.get_labels(): # LabelManager가 존재하고, 라벨 목록이 비어있지 않은 경우
                labeled_frame_numbers.append(frame_num)

        if not labeled_frame_numbers:
            messagebox.showwarning("내보내기 오류", "라벨이 지정된 프레임이 없습니다.")
            return

        # 출력 폴더 선택
        output_dir = filedialog.askdirectory(title="YOLO 파일 저장 폴더 선택")
        if not output_dir:
            return

        # 옵션 선택
        from tkinter import simpledialog # 위치는 함수 상단이나 클래스 임포트 쪽으로 옮겨도 무방
        include_images = messagebox.askyesno(
            "이미지 포함 여부",
            "이미지 파일도 함께 내보내시겠습니까?\n\n"
            "예: 이미지 + 라벨 파일\n"
            "아니오: 라벨 파일만"
        )

        # 파일명 접두사 입력
        prefix = simpledialog.askstring(
            "파일명 접두사",
            "파일명 접두사를 입력하세요:",
            initialvalue=self.get_filename_prefix()
        )
        if not prefix: # 사용자가 취소하거나 빈 문자열 입력 시
            return

        # 진행 상황 표시 다이얼로그
        progress_window = tk.Toplevel(self.app.master)
        progress_window.title("YOLO 일괄 내보내기...")
        progress_window.geometry("450x150")
        progress_window.transient(self.app.master) # 메인 창 위에 항상 표시
        progress_window.grab_set() # 다른 창 비활성화 (모달처럼)
        
        progress_label = ttk.Label(progress_window, text="내보내기 준비 중...")
        progress_label.pack(pady=10)
        
        # 프로그레스바 최대값을 라벨이 있는 프레임 수로 설정
        progress_bar = ttk.Progressbar(progress_window, mode='determinate', maximum=len(labeled_frame_numbers))
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        # 취소 버튼 추가 (선택적이지만 좋은 UX)
        # 이 취소를 실제로 동작하게 하려면, for 루프 내에서 플래그를 확인해야 합니다.
        # 여기서는 progress_window.destroy()가 호출되면 winfo_exists()로 감지합니다.
        cancel_button = ttk.Button(progress_window, text="취소", command=progress_window.destroy)
        cancel_button.pack(pady=5)

        try:
            import os # os 모듈 임포트 위치 확인 (보통 파일 상단)
            labels_dir = os.path.join(output_dir, "labels")
            os.makedirs(labels_dir, exist_ok=True)
            
            images_dir = None
            if include_images:
                images_dir = os.path.join(output_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

            exported_count = 0
            failed_count = 0
            
            # 정렬된 라벨 프레임 번호 목록 사용
            frame_numbers_to_export = sorted(labeled_frame_numbers)

            for i, frame_num in enumerate(frame_numbers_to_export):
                if not progress_window.winfo_exists(): # 사용자가 진행률 창을 닫으면 중단
                    messagebox.showinfo("취소됨", "내보내기 작업이 사용자에 의해 취소되었습니다.")
                    # 원래 프레임으로 복구하는 로직은 finally 블록이나 여기서 명시적 호출 가능
                    self.show_frame_and_labels(self.current_frame_number)
                    return # 함수 종료

                progress_label.config(text=f"프레임 {frame_num} 내보내는 중... ({i+1}/{len(frame_numbers_to_export)})")
                progress_bar.config(value=i + 1) # 0부터 시작하는 인덱스이므로 +1
                progress_window.update_idletasks() # UI 강제 업데이트

                try:
                    label_manager = self.app.frame_label_managers[frame_num]
                    
                    # YOLO 형식 변환 (get_labels()가 비어있지 않음은 위에서 보장됨)
                    yolo_strings = label_manager.to_yolo_format_strings(
                        self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT
                    )
                    
                    # 라벨 파일 저장 (yolo_strings가 비어 있을 수 없음 - 라벨이 있는 프레임만 처리하므로)
                    # 하지만 안전을 위해 비어있는 경우 빈 파일을 생성하도록 유지
                    filename_base = f"{prefix}_{frame_num:05d}" # 공통 파일 이름 (확장자 제외)
                    txt_filepath = os.path.join(labels_dir, f"{filename_base}.txt")
                    
                    with open(txt_filepath, 'w', encoding='utf-8') as f:
                        if yolo_strings: # 이 조건은 항상 참이어야 함
                            f.write("\n".join(yolo_strings))
                        else:
                            # 이 경우는 발생하지 않아야 하지만, 방어적으로 빈 파일 생성
                            f.write("") 
                            print(f"Warning: 라벨이 있는 프레임({frame_num})으로 간주되었으나 YOLO 문자열이 비어있습니다.")


                    # 이미지 파일 저장 (옵션)
                    if include_images and images_dir: # images_dir도 확인
                        if not self.cap or not self.cap.isOpened():
                            print(f"Warning: 이미지 저장을 위해 비디오 캡처가 유효하지 않습니다 (프레임 {frame_num}).")
                            # 이미지를 저장할 수 없는 경우 실패로 간주할지, 라벨만 저장하고 성공으로 간주할지 결정 필요
                            # 여기서는 일단 라벨은 저장된 것으로 계속 진행
                        else:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                            ret, frame = self.cap.read()
                            
                            if ret:
                                processed_frame = self._resize_with_padding(frame, self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT)
                                img_filepath = os.path.join(images_dir, f"{filename_base}.jpg")
                                cv2.imwrite(img_filepath, processed_frame)
                            else:
                                print(f"Warning: 프레임 {frame_num}의 이미지를 읽는 데 실패했습니다.")
                                # 이미지 저장 실패 시 처리 (예: failed_count에 포함 안 함, 라벨만 성공)

                    exported_count += 1

                except Exception as e:
                    print(f"프레임 {frame_num} 내보내기 중 오류 발생: {e}")
                    failed_count += 1
            
            # 루프 정상 종료 후 진행률 창 닫기 (사용자가 먼저 닫지 않았다면)
            if progress_window.winfo_exists():
                progress_window.destroy()

            # 작업 완료 후, 현재 선택된 프레임으로 화면 복구 (필수)
            self.show_frame_and_labels(self.current_frame_number)

            # 최종 결과 표시
            if exported_count > 0:
                result_message = f"YOLO 일괄 내보내기 완료\n\n"
                result_message += f"성공적으로 내보낸 프레임 수: {exported_count}개\n"
                if failed_count > 0:
                    result_message += f"내보내기 실패 프레임 수: {failed_count}개\n"
                result_message += f"출력 폴더: {output_dir}"
                messagebox.showinfo("내보내기 완료", result_message)
            elif failed_count > 0 : # 성공은 없고 실패만 있는 경우
                 messagebox.showerror("내보내기 실패", f"모든 프레임 내보내기에 실패했습니다. (실패: {failed_count}개)")
            # else : exported_count == 0 and failed_count == 0 인 경우는 labeled_frame_numbers가 비어있을 때 이미 처리됨

        except Exception as e: # try 블록의 최상위 예외 처리
            if progress_window.winfo_exists():
                progress_window.destroy()
            messagebox.showerror("내보내기 중 심각한 오류", f"일괄 내보내기 중 예기치 않은 오류 발생:\n{e}")
            # 이 경우에도 현재 프레임 복구
            self.show_frame_and_labels(self.current_frame_number)

    def show_all_labels_statistics(self):
        """전체 라벨 통계 표시"""
        if not self.app.frame_label_managers:
            messagebox.showinfo("통계", "표시할 라벨 데이터가 없습니다.")
            return

        # 통계 수집
        total_frames = len(self.app.frame_label_managers)
        total_labels = 0
        class_counts = {}
        frame_label_counts = {}

        for frame_num, label_manager in self.app.frame_label_managers.items():
            frame_labels = label_manager.get_labels()
            frame_label_count = len(frame_labels)
            
            total_labels += frame_label_count
            frame_label_counts[frame_num] = frame_label_count

            for label in frame_labels:
                class_name = label['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # 통계 텍스트 생성
        stats_lines = [
            f"=== 전체 라벨링 통계 ===",
            f"",
            f"비디오 정보:",
            f"  총 프레임 수: {self.total_frames}",
            f"  라벨된 프레임: {total_frames}개 ({total_frames/self.total_frames*100:.1f}%)",
            f"  총 라벨 수: {total_labels}개",
            f"  평균 라벨/프레임: {total_labels/total_frames:.1f}개" if total_frames > 0 else "  평균 라벨/프레임: 0개",
            f"",
            f"클래스별 분포:"
        ]

        if class_counts:
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / total_labels) * 100 if total_labels > 0 else 0
                stats_lines.append(f"  {class_name}: {count}개 ({percentage:.1f}%)")
        else:
            stats_lines.append("  (라벨 없음)")

        stats_lines.extend([
            f"",
            f"프레임별 라벨 수 분포:"
        ])

        # 프레임별 라벨 수 히스토그램
        label_count_histogram = {}
        for count in frame_label_counts.values():
            label_count_histogram[count] = label_count_histogram.get(count, 0) + 1

        for label_count in sorted(label_count_histogram.keys()):
            frame_count = label_count_histogram[label_count]
            stats_lines.append(f"  {label_count}개 라벨: {frame_count}개 프레임")

        # 통계 창 표시
        stats_window = tk.Toplevel(self.app.master)
        stats_window.title("전체 라벨링 통계")
        stats_window.geometry("500x600")
        stats_window.transient(self.app.master)

        # 텍스트 위젯
        text_frame = ttk.Frame(stats_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        stats_text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=stats_text.yview)
        stats_text.configure(yscrollcommand=scrollbar.set)

        stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 통계 텍스트 삽입
        stats_text.insert(tk.END, "\n".join(stats_lines))
        stats_text.config(state=tk.DISABLED)

        # 닫기 버튼
        close_button = ttk.Button(stats_window, text="닫기", command=stats_window.destroy)
        close_button.pack(pady=10)

    # === 프레임 네비게이션 ===
    def next_frame(self):
        """다음 프레임으로 이동"""
        if self.is_video_loaded and self.current_frame_number < self.total_frames - 1:
            self.clear_active_segmentation_data_internally()
            self.selected_label_id_from_list = None
            self.clear_listbox_selection()
            self.show_frame_and_labels(self.current_frame_number + 1)

    def prev_frame(self):
        """이전 프레임으로 이동"""
        if self.is_video_loaded and self.current_frame_number > 0:
            self.clear_active_segmentation_data_internally()
            self.selected_label_id_from_list = None
            self.clear_listbox_selection()
            self.show_frame_and_labels(self.current_frame_number - 1)

    def go_to_frame(self):
        """특정 프레임으로 이동"""
        if not self.is_video_loaded:
            return
        
        try:
            target_frame_str = self.entry_goto_frame.get()
            if not target_frame_str:
                return
            
            target_frame = int(target_frame_str)
            if 0 <= target_frame < self.total_frames:
                self.clear_active_segmentation_data_internally()
                self.selected_label_id_from_list = None
                self.clear_listbox_selection()
                self.show_frame_and_labels(target_frame)
            else:
                messagebox.showwarning("입력 오류", f"프레임 번호는 0에서 {self.total_frames - 1} 사이여야 합니다.")
        except ValueError:
            messagebox.showerror("입력 오류", "유효한 숫자를 입력하세요.")
        finally:
            self.clear_goto_entry()

    def go_to_frame_event(self, event=None):
        """엔터 키로 프레임 이동"""
        self.go_to_frame()

    # === UI 업데이트 메서드 ===
    def update_ui_state(self):
        """UI 상태 업데이트"""
        video_loaded = self.is_video_loaded
        
        # 네비게이션 버튼
        self.btn_prev_frame.config(state=tk.NORMAL if video_loaded and self.current_frame_number > 0 else tk.DISABLED)
        self.btn_next_frame.config(state=tk.NORMAL if video_loaded and self.current_frame_number < self.total_frames - 1 else tk.DISABLED)
        self.btn_goto_frame.config(state=tk.NORMAL if video_loaded else tk.DISABLED)
        self.entry_goto_frame.config(state=tk.NORMAL if video_loaded else tk.DISABLED)

        # 라벨링 버튼
        can_add_or_update = video_loaded and self.current_active_bbox_processed is not None
        self.btn_add_label.config(state=tk.NORMAL if can_add_or_update else tk.DISABLED)
        
        can_clear_active = video_loaded and (self.current_active_bbox_processed is not None or self.current_active_mask_processed is not None)
        self.btn_clear_active.config(state=tk.NORMAL if can_clear_active else tk.DISABLED)
        
        can_delete_selected = video_loaded and self.selected_label_id_from_list is not None
        self.btn_delete_label.config(state=tk.NORMAL if can_delete_selected else tk.DISABLED)
        
        # 클래스 콤보박스
        self.class_combobox.config(state="readonly" if video_loaded else tk.DISABLED, values=list(self.app.class_names.values()))
        if video_loaded and not self.class_var.get() and self.app.class_names:
            self.class_combobox.current(0)

        # 저장 관련
        self.entry_filename_prefix.config(state=tk.NORMAL if video_loaded else tk.DISABLED)
        self.btn_save_labels.config(state=tk.NORMAL if video_loaded else tk.DISABLED)
        
        # 전체 라벨 관리 버튼들
        has_any_labels = video_loaded and bool(self.app.frame_label_managers)
        self.btn_save_all_labels.config(state=tk.NORMAL if has_any_labels else tk.DISABLED)
        self.btn_load_all_labels.config(state=tk.NORMAL if video_loaded else tk.DISABLED)
        self.btn_export_yolo_batch.config(state=tk.NORMAL if has_any_labels else tk.DISABLED)
        self.btn_show_label_stats.config(state=tk.NORMAL if has_any_labels else tk.DISABLED)

    def update_frame_info(self, frame_num_text, file_name_text):
        """프레임 정보 업데이트"""
        self.lbl_frame_info.config(text=frame_num_text)
        self.lbl_file_name.config(text=file_name_text)

    def update_video_display(self, photo_image):
        """비디오 디스플레이 업데이트"""
        self.video_label.config(image=photo_image)
        self.photo_image = photo_image

    def update_labels_listbox_display(self, labels_data, selected_id):
        """라벨 목록 업데이트"""
        self.labels_listbox.delete(0, tk.END)
        for i, label in enumerate(labels_data):
            bbox_str = f"[{label['bbox_processed'][0]},{label['bbox_processed'][1]},{label['bbox_processed'][2]},{label['bbox_processed'][3]}]"
            display_text = f"{i+1}. {label['class_name']} - {bbox_str}"
            self.labels_listbox.insert(tk.END, display_text)
            if label.get('id') == selected_id:
                self.labels_listbox.selection_set(i)
                self.labels_listbox.activate(i)

    # === 유틸리티 메서드 ===
    def set_class_combobox(self, class_name):
        """클래스 콤보박스 설정"""
        self.class_var.set(class_name)

    def get_filename_prefix(self):
        """파일명 접두사 가져오기"""
        return self.filename_prefix_var.get()

    def set_add_update_button_text(self, text):
        """추가/업데이트 버튼 텍스트 설정"""
        self.btn_add_label.config(text=text)

    def clear_goto_entry(self):
        """프레임 이동 입력창 지우기"""
        self.entry_goto_frame.delete(0, tk.END)

    def focus_prefix_entry(self):
        """파일명 접두사 입력창에 포커스"""
        self.entry_filename_prefix.focus()

    def clear_listbox_selection(self):
        """라벨 목록 선택 해제"""
        self.labels_listbox.selection_clear(0, tk.END)
        self.selected_label_id_from_list = None
        self.set_add_update_button_text("새 라벨 추가")

    def show_about(self):
        """정보 다이얼로그 표시"""
        about_text = f"""자동 라벨링 앱 v2.5 (전체 라벨 관리 기능 추가)

처리 해상도: {self.PROCESSING_WIDTH}x{self.PROCESSING_HEIGHT}
OpenCV, Tkinter, Ultralytics SAM, PyYAML 사용
비디오 프레임의 객체 라벨링 및 YOLO 형식 저장 지원

=== 핫키 ===
← → : 이전/다음 프레임
↑ ↓ : 10프레임씩 이동  
Space : 다음 프레임
Delete : 선택된 라벨 삭제
Esc : 활성 세그멘테이션 지우기
Ctrl+S : 현재 프레임 저장

=== 자동 세그멘테이션 ===
Shift+← : 이전 프레임 bbox 참조해서 자동 세그멘테이션
Shift+→ : 다음 프레임 bbox 참조해서 자동 세그멘테이션

=== 전체 라벨 관리 (NEW!) ===
• 전체 라벨 JSON 저장/불러오기
• YOLO 일괄 내보내기 (이미지 포함 옵션)
• 전체 라벨링 통계 보기

=== 마우스 ===
좌클릭 : 세그멘테이션 실행
우클릭 : 활성 세그멘테이션 지우기
드래그 : 바운딩 박스 크기 조정"""
        
        messagebox.showinfo("자동 라벨링 앱 정보", about_text)