import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2 
from PIL import Image, ImageTk 
import os 
import json
from LabelManager import LabelManager

class ManagementTab:
    def __init__(self, parent_tab_frame, app_controller):
        self.parent_tab = parent_tab_frame
        self.app = app_controller 
        self.photo_image = None # For this tab's preview
        self.current_label_manager = LabelManager()  # 관리탭용 라벨 매니저

        # --- UI Elements for Management Tab ---
        main_container = ttk.Frame(self.parent_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 상단 버튼 패널
        top_button_frame = ttk.LabelFrame(main_container, text="라벨 관리 작업", padding="10")
        top_button_frame.pack(fill=tk.X, pady=(0, 5))

        # 파일 작업 버튼들
        file_ops_frame = ttk.Frame(top_button_frame)
        file_ops_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_new_labels = ttk.Button(file_ops_frame, text="새 라벨 세트", command=self.new_label_set_action)
        self.btn_new_labels.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_load_labels = ttk.Button(file_ops_frame, text="라벨 불러오기", command=self.load_labels_action)
        self.btn_load_labels.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_save_labels = ttk.Button(file_ops_frame, text="라벨 저장", command=self.save_labels_action)
        self.btn_save_labels.pack(side=tk.LEFT, padx=(0, 5))

        # YOLO 작업 버튼들
        yolo_ops_frame = ttk.Frame(top_button_frame)
        yolo_ops_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_import_yolo = ttk.Button(yolo_ops_frame, text="YOLO 파일 가져오기", command=self.import_yolo_action)
        self.btn_import_yolo.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_export_yolo = ttk.Button(yolo_ops_frame, text="YOLO로 내보내기", command=self.export_yolo_action)
        self.btn_export_yolo.pack(side=tk.LEFT, padx=(0, 5))

        # 일괄 처리 버튼들
        batch_ops_frame = ttk.Frame(top_button_frame)
        batch_ops_frame.pack(fill=tk.X)
        
        self.btn_batch_export = ttk.Button(batch_ops_frame, text="전체 프레임 일괄 저장", command=self.batch_export_all_frames_action)
        self.btn_batch_export.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_show_stats = ttk.Button(batch_ops_frame, text="라벨 통계", command=self.show_statistics_action)
        self.btn_show_stats.pack(side=tk.LEFT, padx=(0, 5))

        # 하단 컨테이너
        bottom_container = ttk.Frame(main_container)
        bottom_container.pack(fill=tk.BOTH, expand=True)

        # 좌측 패널 (저장된 이미지 목록)
        left_panel = ttk.Frame(bottom_container, padding="5")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        
        list_frame = ttk.LabelFrame(left_panel, text="저장된 이미지 목록", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.btn_refresh_list = ttk.Button(list_frame, text="목록 새로고침", command=self.refresh_list_action)
        self.btn_refresh_list.pack(fill=tk.X, pady=(0, 5))

        self.labeled_images_listbox = tk.Listbox(list_frame)
        self.labeled_images_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.labeled_images_listbox.bind("<<ListboxSelect>>", self.on_image_select_action)

        # 우측 패널 (미리보기 및 라벨 편집)
        right_panel = ttk.Frame(bottom_container, width=450, padding="5") 
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # 미리보기 섹션
        preview_frame = ttk.LabelFrame(right_panel, text="이미지 미리보기", padding="10")
        preview_frame.pack(fill=tk.X, pady=(0, 5))

        # 미리보기 컨테이너 프레임으로 크기 고정
        preview_container = ttk.Frame(preview_frame, height=200, width=400)
        preview_container.pack(fill=tk.X, pady=(0, 5))
        preview_container.pack_propagate(False)  # 크기 고정

        self.preview_label = ttk.Label(preview_container, background="lightgrey", anchor=tk.CENTER, text="이미지 선택 시 미리보기")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # 현재 라벨 편집 섹션
        edit_frame = ttk.LabelFrame(right_panel, text="현재 라벨 편집", padding="10")
        edit_frame.pack(fill=tk.BOTH, expand=True)

        # 라벨 목록
        self.current_labels_listbox = tk.Listbox(edit_frame, height=8)
        self.current_labels_listbox.pack(fill=tk.X, pady=(0, 5))
        self.current_labels_listbox.bind("<<ListboxSelect>>", self.on_current_label_select)

        # 라벨 편집 버튼들
        edit_buttons_frame = ttk.Frame(edit_frame)
        edit_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_delete_label = ttk.Button(edit_buttons_frame, text="선택 라벨 삭제", command=self.delete_current_label_action)
        self.btn_delete_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_clear_all = ttk.Button(edit_buttons_frame, text="모든 라벨 삭제", command=self.clear_all_labels_action)
        self.btn_clear_all.pack(side=tk.LEFT)

        # 라벨 정보 텍스트
        self.label_info_text = tk.Text(edit_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.label_info_text.pack(fill=tk.X)
        
        # 초기 UI 상태 설정
        self.labeled_image_files = []
        self.selected_labeled_image_file = None
        self.refresh_list_action() # Initial list load
        self.update_current_labels_display()

    def refresh_list_action(self):
        """UI action to refresh the list of labeled images."""
        try:
            # 이전 이미지 참조 정리
            if hasattr(self, 'photo_image') and self.photo_image:
                self.photo_image = None
            
            # 미리보기 초기화
            self.preview_label.config(image="", text="이미지 선택 시 미리보기")
            
            # 이미지 목록 새로고침
            self.labeled_image_files = self.app.data_manager.list_labeled_images()
            self.update_listbox_display(self.labeled_image_files)
            print(f"ManagementTab: 이미지 목록 새로고침 완료 - {len(self.labeled_image_files)}개")
            
        except Exception as e:
            print(f"ManagementTab Error: 이미지 목록 새로고침 실패 - {e}")
            # 에러 발생 시에도 기본 상태로 복구
            self.labeled_image_files = []
            try:
                self.update_listbox_display([])
                self.preview_label.config(image="", text="목록 로드 오류")
            except:
                pass

    def on_image_select_action(self, event):
        """UI action when an image is selected from the list."""
        if not self.labeled_images_listbox.curselection():
            self.selected_labeled_image_file = None
            self.display_preview_and_labels(None, [])
            return

        selected_index = self.labeled_images_listbox.curselection()[0]
        if 0 <= selected_index < len(self.labeled_image_files):
            self.selected_labeled_image_file = self.labeled_image_files[selected_index]
            img_path = self.app.data_manager.get_image_path(self.selected_labeled_image_file)
            yolo_labels = self.app.data_manager.load_yolo_labels_from_file(self.selected_labeled_image_file)
            
            # YOLO 라벨을 현재 라벨 매니저로 로드
            self.load_yolo_labels_to_current_manager(yolo_labels)
            
            self.display_preview_and_labels(img_path, yolo_labels)
            self.update_current_labels_display()

    def load_yolo_labels_to_current_manager(self, yolo_labels):
        """YOLO 라벨 데이터를 현재 라벨 매니저로 로드합니다."""
        self.current_label_manager.clear_labels()
        
        processing_width = getattr(self.app.labeling_tab_ui, 'PROCESSING_WIDTH', 640)
        processing_height = getattr(self.app.labeling_tab_ui, 'PROCESSING_HEIGHT', 640)
        
        for yolo_label in yolo_labels:
            class_id = yolo_label['class_id']
            class_name = yolo_label['class_name']
            cx_rel, cy_rel, w_rel, h_rel = yolo_label['bbox_yolo']
            
            # YOLO 형식을 처리 해상도 절대 좌표로 변환
            x_proc = int((cx_rel - w_rel / 2) * processing_width)
            y_proc = int((cy_rel - h_rel / 2) * processing_height)
            w_proc = int(w_rel * processing_width)
            h_proc = int(h_rel * processing_height)
            
            # 경계 검사
            x_proc = max(0, min(x_proc, processing_width - 1))
            y_proc = max(0, min(y_proc, processing_height - 1))
            w_proc = max(1, min(w_proc, processing_width - x_proc))
            h_proc = max(1, min(h_proc, processing_height - y_proc))
            
            self.current_label_manager.add_label(class_id, class_name, [x_proc, y_proc, w_proc, h_proc])

    def new_label_set_action(self):
        """새 라벨 세트를 시작합니다."""
        if len(self.current_label_manager.get_labels()) > 0:
            if not messagebox.askyesno("확인", "현재 작업 중인 라벨이 있습니다. 새로 시작하시겠습니까?"):
                return
        
        self.current_label_manager.clear_labels()
        self.update_current_labels_display()
        self.label_info_text.config(state=tk.NORMAL)
        self.label_info_text.delete(1.0, tk.END)
        self.label_info_text.insert(tk.END, "새 라벨 세트가 시작되었습니다.")
        self.label_info_text.config(state=tk.DISABLED)
        messagebox.showinfo("완료", "새 라벨 세트가 준비되었습니다.")

    def load_labels_action(self):
        """라벨 파일을 불러옵니다."""
        filepath = filedialog.askopenfilename(
            title="라벨 파일 선택",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        success, message = self.current_label_manager.load_from_file(filepath)
        if success:
            self.update_current_labels_display()
            messagebox.showinfo("완료", message)
        else:
            messagebox.showerror("오류", message)

    def save_labels_action(self):
        """현재 라벨을 파일로 저장합니다."""
        if len(self.current_label_manager.get_labels()) == 0:
            messagebox.showwarning("경고", "저장할 라벨이 없습니다.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="라벨 저장",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        success, message = self.current_label_manager.save_to_file(filepath)
        if success:
            messagebox.showinfo("완료", message)
        else:
            messagebox.showerror("오류", message)

    def import_yolo_action(self):
        """YOLO 파일을 가져옵니다."""
        filepath = filedialog.askopenfilename(
            title="YOLO 파일 선택",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        processing_width = getattr(self.app.labeling_tab_ui, 'PROCESSING_WIDTH', 640)
        processing_height = getattr(self.app.labeling_tab_ui, 'PROCESSING_HEIGHT', 640)
        
        success, message = self.current_label_manager.import_from_yolo_file(
            filepath, self.app.class_names, processing_width, processing_height
        )
        
        if success:
            self.update_current_labels_display()
            messagebox.showinfo("완료", message)
        else:
            messagebox.showerror("오류", message)

    def export_yolo_action(self):
        """현재 라벨을 YOLO 형식으로 내보냅니다."""
        if len(self.current_label_manager.get_labels()) == 0:
            messagebox.showwarning("경고", "내보낼 라벨이 없습니다.")
            return
        
        output_dir = filedialog.askdirectory(title="YOLO 파일 저장 폴더 선택")
        if not output_dir:
            return
        
        # 파일명 입력 받기
        filename = simpledialog.askstring("파일명 입력", "파일명을 입력하세요 (확장자 제외):", initialvalue="labels")
        if not filename:
            return
        
        processing_width = getattr(self.app.labeling_tab_ui, 'PROCESSING_WIDTH', 640)
        processing_height = getattr(self.app.labeling_tab_ui, 'PROCESSING_HEIGHT', 640)
        
        success, txt_path, message = self.current_label_manager.export_to_yolo_files(
            output_dir, filename, processing_width, processing_height
        )
        
        if success:
            messagebox.showinfo("완료", message)
        else:
            messagebox.showerror("오류", message)

    def batch_export_all_frames_action(self):
        """모든 프레임의 라벨을 일괄 저장합니다."""
        if not hasattr(self.app, 'labeling_tab_ui') or not self.app.labeling_tab_ui.is_video_loaded:
            messagebox.showwarning("경고", "먼저 비디오를 로드해주세요.")
            return
        
        if not self.app.labeling_tab_ui.frame_label_managers:
            messagebox.showwarning("경고", "저장할 라벨이 있는 프레임이 없습니다.")
            return
        
        # 저장 옵션 선택
        result = messagebox.askyesnocancel(
            "일괄 저장 옵션", 
            "모든 프레임을 일괄 저장하시겠습니까?\n\n"
            "예: 라벨이 있는 프레임만 저장\n"
            "아니오: 모든 프레임 저장 (빈 라벨 포함)\n"
            "취소: 작업 취소"
        )
        
        if result is None:  # 취소
            return
        
        save_only_labeled = result  # True면 라벨이 있는 프레임만, False면 모든 프레임
        
        # 접두사 입력
        prefix = simpledialog.askstring(
            "파일명 접두사", 
            "저장할 파일의 접두사를 입력하세요:", 
            initialvalue="batch_frame"
        )
        if not prefix:
            return
        
        # 진행 상황 표시를 위한 다이얼로그
        progress_window = tk.Toplevel(self.app.master)
        progress_window.title("일괄 저장 진행 중...")
        progress_window.geometry("400x120")
        progress_window.transient(self.app.master)
        progress_window.grab_set()
        
        progress_label = ttk.Label(progress_window, text="저장 준비 중...")
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, mode='determinate')
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        cancel_button = ttk.Button(progress_window, text="취소", command=progress_window.destroy)
        cancel_button.pack(pady=5)
        
        # 처리할 프레임 목록 결정
        if save_only_labeled:
            frames_to_process = list(self.app.labeling_tab_ui.frame_label_managers.keys())
            frames_to_process = [f for f in frames_to_process if len(self.app.labeling_tab_ui.frame_label_managers[f].get_labels()) > 0]
        else:
            frames_to_process = list(range(self.app.labeling_tab_ui.total_frames))
        
        if not frames_to_process:
            progress_window.destroy()
            messagebox.showwarning("경고", "저장할 프레임이 없습니다.")
            return
        
        progress_bar.configure(maximum=len(frames_to_process))
        saved_count = 0
        failed_count = 0
        
        try:
            for i, frame_number in enumerate(frames_to_process):
                if not progress_window.winfo_exists():  # 취소된 경우
                    break
                
                progress_label.config(text=f"프레임 {frame_number} 저장 중... ({i+1}/{len(frames_to_process)})")
                progress_bar.configure(value=i)
                progress_window.update()
                
                # 해당 프레임으로 이동
                self.app.labeling_tab_ui.show_frame_and_labels(frame_number)
                
                # 프레임 이미지 가져오기
                self.app.labeling_tab_ui.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, original_frame = self.app.labeling_tab_ui.cap.read()
                
                if not ret:
                    print(f"BatchExport Warning: 프레임 {frame_number} 읽기 실패")
                    failed_count += 1
                    continue
                
                # 라벨 매니저 가져오기
                if frame_number in self.app.labeling_tab_ui.frame_label_managers:
                    label_manager = self.app.labeling_tab_ui.frame_label_managers[frame_number]
                else:
                    from LabelManager import LabelManager
                    label_manager = LabelManager()  # 빈 라벨 매니저
                
                # YOLO 형식 변환
                yolo_strings = label_manager.to_yolo_format_strings(
                    self.app.labeling_tab_ui.PROCESSING_WIDTH, 
                    self.app.labeling_tab_ui.PROCESSING_HEIGHT
                )
                
                # 저장
                success, img_path, txt_path, message = self.app.data_manager.save_image_and_yolo_labels(
                    original_frame, yolo_strings, prefix, f"{frame_number:05d}"
                )
                
                if success:
                    saved_count += 1
                else:
                    failed_count += 1
                    print(f"BatchExport Error: 프레임 {frame_number} 저장 실패 - {message}")
            
            progress_window.destroy()
            
            # 결과 표시
            if progress_window.winfo_exists():  # 취소되지 않은 경우만
                result_message = f"일괄 저장 완료\n\n성공: {saved_count}개\n실패: {failed_count}개"
                if saved_count > 0:
                    messagebox.showinfo("완료", result_message)
                else:
                    messagebox.showerror("실패", result_message)
                
                # 목록 새로고침
                self.refresh_list_action()
        
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("오류", f"일괄 저장 중 오류 발생:\n{e}")

    def show_statistics_action(self):
        """현재 라벨의 통계를 보여줍니다."""
        stats = self.current_label_manager.get_statistics()
        
        # 통계 창 생성
        stats_window = tk.Toplevel(self.app.master)
        stats_window.title("라벨 통계")
        stats_window.geometry("400x300")
        stats_window.transient(self.app.master)
        
        stats_text = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        stats_text.pack(fill=tk.BOTH, expand=True)
        
        stats_text.insert(tk.END, stats)
        stats_text.config(state=tk.DISABLED)
        
        close_button = ttk.Button(stats_window, text="닫기", command=stats_window.destroy)
        close_button.pack(pady=10)

    def on_current_label_select(self, event):
        """현재 라벨 목록에서 선택했을 때의 처리"""
        # 현재는 단순히 정보만 표시
        pass

    def delete_current_label_action(self):
        """선택된 라벨을 삭제합니다."""
        if not self.current_labels_listbox.curselection():
            messagebox.showwarning("경고", "삭제할 라벨을 선택해주세요.")
            return
        
        selected_index = self.current_labels_listbox.curselection()[0]
        labels = self.current_label_manager.get_labels()
        
        if 0 <= selected_index < len(labels):
            label_to_delete = labels[selected_index]
            success = self.current_label_manager.delete_label(label_to_delete['id'])
            
            if success:
                self.update_current_labels_display()
                messagebox.showinfo("완료", "라벨이 삭제되었습니다.")
            else:
                messagebox.showerror("오류", "라벨 삭제에 실패했습니다.")

    def clear_all_labels_action(self):
        """모든 라벨을 삭제합니다."""
        if len(self.current_label_manager.get_labels()) == 0:
            messagebox.showwarning("경고", "삭제할 라벨이 없습니다.")
            return
        
        if messagebox.askyesno("확인", "모든 라벨을 삭제하시겠습니까?"):
            self.current_label_manager.clear_labels()
            self.update_current_labels_display()
            messagebox.showinfo("완료", "모든 라벨이 삭제되었습니다.")

    def update_current_labels_display(self):
        """현재 라벨 목록을 업데이트합니다."""
        self.current_labels_listbox.delete(0, tk.END)
        
        labels = self.current_label_manager.get_labels()
        if not labels:
            self.current_labels_listbox.insert(tk.END, "라벨이 없습니다")
            return
        
        for i, label in enumerate(labels):
            bbox = label['bbox_processed']
            display_text = f"{i+1}. {label['class_name']} - [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
            self.current_labels_listbox.insert(tk.END, display_text)

    # --- UI Update Methods (called by app_controller or self) ---
    def update_ui_state(self):
        """UI 상태를 업데이트합니다."""
        has_labels = len(self.current_label_manager.get_labels()) > 0
        
        # 버튼 상태 업데이트
        self.btn_save_labels.config(state=tk.NORMAL if has_labels else tk.DISABLED)
        self.btn_export_yolo.config(state=tk.NORMAL if has_labels else tk.DISABLED)
        self.btn_clear_all.config(state=tk.NORMAL if has_labels else tk.DISABLED)
        self.btn_show_stats.config(state=tk.NORMAL if has_labels else tk.DISABLED)
        
        # 일괄 저장 버튼은 비디오가 로드되고 라벨이 있는 프레임이 있을 때만 활성화
        batch_enabled = (hasattr(self.app, 'labeling_tab_ui') and 
                        self.app.labeling_tab_ui.is_video_loaded and 
                        bool(getattr(self.app.labeling_tab_ui, 'frame_label_managers', {})))
        self.btn_batch_export.config(state=tk.NORMAL if batch_enabled else tk.DISABLED)

    def update_listbox_display(self, image_files):
        """이미지 목록 표시를 업데이트합니다."""
        self.labeled_images_listbox.delete(0, tk.END)
        for filename in image_files:
            self.labeled_images_listbox.insert(tk.END, filename)
        if not image_files:
            self.labeled_images_listbox.insert(tk.END, "저장된 이미지가 없습니다.")
        
        if not self.labeled_images_listbox.curselection(): # If nothing is selected (or list is empty)
            self.display_preview_and_labels(None, [])

    def display_preview_and_labels(self, image_path, yolo_labels):
        """이미지 미리보기와 라벨 정보를 표시합니다."""
        # 이전 이미지 참조 정리
        if hasattr(self, 'photo_image') and self.photo_image:
            try:
                self.photo_image = None
            except:
                pass
        
        if image_path and os.path.exists(image_path):
            try:
                img_bgr = cv2.imread(image_path)
                if img_bgr is None: 
                    raise ValueError("이미지를 읽을 수 없습니다.")
                
                display_img_for_preview = img_bgr.copy()
                label_info_str_parts = [f"이미지: {os.path.basename(image_path)}\n--- 라벨 정보 ---"]

                if yolo_labels:
                    img_h_orig, img_w_orig = display_img_for_preview.shape[:2]
                    for lbl in yolo_labels:
                        class_name = lbl['class_name']
                        cx_rel, cy_rel, w_rel, h_rel = lbl['bbox_yolo']
                        abs_w=w_rel*img_w_orig; abs_h=h_rel*img_h_orig; abs_cx=cx_rel*img_w_orig; abs_cy=cy_rel*img_h_orig
                        x1=int(abs_cx-abs_w/2); y1=int(abs_cy-abs_h/2); x2=int(abs_cx+abs_w/2); y2=int(abs_cy+abs_h/2)
                        
                        # 바운딩 박스 색상 설정 (기본값)
                        bbox_color = getattr(self.app.labeling_tab_ui, 'SAVED_BBOX_COLOR', (255, 0, 255))
                        cv2.rectangle(display_img_for_preview,(x1,y1),(x2,y2), bbox_color, 2) 
                        cv2.putText(display_img_for_preview,class_name,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,bbox_color,1)
                        label_info_str_parts.append(f"{class_name}: [중심: {cx_rel:.3f},{cy_rel:.3f}, 크기: {w_rel:.3f},{h_rel:.3f}]")
                else:
                    label_info_str_parts.append("라벨 없음")

                # 미리보기 라벨 크기 가져오기 (안전하게)
                try:
                    self.preview_label.update_idletasks()  # UI 업데이트 강제 실행
                    preview_label_w = self.preview_label.winfo_width()
                    preview_label_h = self.preview_label.winfo_height()
                    if preview_label_w <= 1 or preview_label_h <= 1: 
                        preview_label_w, preview_label_h = 350, 180  # 기본 크기
                except:
                    preview_label_w, preview_label_h = 350, 180

                img_h, img_w = display_img_for_preview.shape[:2]
                scale = min(preview_label_w / img_w, preview_label_h / img_h) if img_w > 0 and img_h > 0 else 1.0
                
                disp_w = int(img_w * scale) if scale > 0 else img_w
                disp_h = int(img_h * scale) if scale > 0 else img_h
                
                if disp_w > 0 and disp_h > 0:
                    resized_img = cv2.resize(display_img_for_preview, (disp_w, disp_h))
                else: 
                    resized_img = display_img_for_preview 

                img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # PhotoImage 생성 및 참조 저장
                self.photo_image = ImageTk.PhotoImage(image=pil_img) 
                
                # 라벨에 이미지 설정 (에러 처리 추가)
                try:
                    self.preview_label.config(image=self.photo_image, text="")
                except tk.TclError as e:
                    print(f"이미지 설정 중 오류: {e}")
                    self.preview_label.config(image="", text="이미지 표시 오류")
                    self.photo_image = None

                # 라벨 정보 텍스트 업데이트
                self.label_info_text.config(state=tk.NORMAL)
                self.label_info_text.delete(1.0, tk.END)
                self.label_info_text.insert(tk.END, "\n".join(label_info_str_parts))
                self.label_info_text.config(state=tk.DISABLED)
                
            except Exception as e:
                print(f"미리보기 이미지 로드/표시 오류: {e}")
                self.preview_label.config(image="", text="미리보기 로드 오류")
                self.photo_image = None
        else:
            # 이미지가 없을 때
            self.preview_label.config(image="", text="이미지 파일을 찾을 수 없음" if image_path else "이미지 선택 시 미리보기")
            self.photo_image = None
            self.label_info_text.config(state=tk.NORMAL)
            self.label_info_text.delete(1.0, tk.END)
            if not image_path: 
                self.label_info_text.insert(tk.END, "표시할 라벨 정보 없음")
            self.label_info_text.config(state=tk.DISABLED)