
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import datetime
import json
from ultralytics import YOLO
import tensorflow as tf
import numpy as np

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class TumorDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MRI Tumor Detection")
        self.geometry("1400x800")
        self.lang = "en"
        self.history = []

        if os.path.exists("history.json"):
            with open("history.json", "r") as f:
                self.history = json.load(f)

        self.model = None
        self.model_type = None

        try:
            self.model = YOLO(r"C:\Users\ADMIN\Downloads\best (3).pt")
            self.model_type = 'yolo'
            print("YOLO model (.pt) loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default YOLO model:\n{e}")

        self.init_ui()

    def init_ui(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)

        self.sidebar = ctk.CTkFrame(self.main_frame, width=300, corner_radius=10)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.header = ctk.CTkFrame(self.main_frame, height=60, corner_radius=10)
        self.header.pack(fill="x", padx=10, pady=(10, 5))

        self.tabview = ctk.CTkScrollableFrame(self.main_frame, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.title_label = ctk.CTkLabel(
            self.header,
            text="MRI Tumor Detection System",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(side="left", padx=20)

        ctk.CTkLabel(
            self.sidebar,
            text="Control Panel",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 10))

        self.upload_button = ctk.CTkButton(
            self.sidebar,
            text="Upload MRI",
            command=self.upload_image,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.upload_button.pack(pady=10, padx=20, fill="x")

        self.load_model_button = ctk.CTkButton(
            self.sidebar,
            text="Load Model",
            command=self.load_model,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.load_model_button.pack(pady=10, padx=20, fill="x")

        self.dark_mode_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Dark Mode",
            command=self.toggle_dark_mode,
            font=ctk.CTkFont(size=14)
        )
        self.dark_mode_switch.pack(pady=10, padx=20)

        self.lang_button = ctk.CTkButton(
            self.sidebar,
            text="Tiếng Việt",
            command=self.switch_language,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.lang_button.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(
            self.sidebar,
            text="Patient Info",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 10))

        self.name_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Name", height=40)
        self.name_entry.pack(pady=5, padx=20, fill="x")

        self.age_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Age", height=40)
        self.age_entry.pack(pady=5, padx=20, fill="x")

        self.diagnosis_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Diagnosis", height=40)
        self.diagnosis_entry.pack(pady=5, padx=20, fill="x")

        self.save_button = ctk.CTkButton(
            self.sidebar,
            text="Save Record",
            command=self.save_record,
            height=40
        )
        self.save_button.pack(pady=10, padx=20, fill="x")

        self.main_content = ctk.CTkFrame(self.tabview, corner_radius=10)
        self.main_content.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_frame = ctk.CTkFrame(self.main_content, corner_radius=10)
        self.image_frame.pack(pady=20, padx=20, fill="both")

        self.image_label = ctk.CTkLabel(self.image_frame, text="No Image Selected", height=400)
        self.image_label.pack(expand=True, fill="both")

        self.result_label = ctk.CTkLabel(self.main_content, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.result_label.pack(pady=20)

        self.history_frame = ctk.CTkFrame(self.tabview, corner_radius=10)
        self.history_label = ctk.CTkLabel(self.history_frame, text="Diagnosis History", font=ctk.CTkFont(size=18, weight="bold"))
        self.history_label.pack(pady=20)

        self.history_text = ctk.CTkTextbox(self.history_frame, height=400, width=800, font=ctk.CTkFont(size=14))
        self.history_text.pack(pady=10, padx=20)
        self.update_history_display()

        self.tab_control = ctk.CTkSegmentedButton(self.header, values=["Main", "History"], command=self.switch_tab)
        self.tab_control.pack(side="right", padx=20)
        self.tab_control.set("Main")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pt *.keras")]
        )
        if file_path:
            try:
                if file_path.endswith('.pt'):
                    self.model = YOLO(file_path)
                    self.model_type = 'yolo'
                    messagebox.showinfo("Success", "Mô hình YOLO (.pt) được tải thành công." if self.lang == "vi" else "YOLO model (.pt) loaded successfully.")
                elif file_path.endswith('.keras'):
                    self.model = tf.keras.models.load_model(file_path, custom_objects={'weighted_dice_loss': weighted_dice_loss})
                    self.model_type = 'keras'
                    messagebox.showinfo("Success", "Mô hình Keras (.keras) được tải thành công." if self.lang == "vi" else "Keras model (.keras) loaded successfully.")
                else:
                    messagebox.showerror("Error", "Định dạng file không được hỗ trợ. Vui lòng chọn file .pt hoặc .keras." if self.lang == "vi" else "Unsupported file type. Please select a .pt or .keras file.")
            except Exception as e:
                messagebox.showerror("Error", f"Không thể tải mô hình:\n{e}" if self.lang == "vi" else f"Failed to load model:\n{e}")
                self.model = None
                self.model_type = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            try:
                img = Image.open(file_path)
                img = img.resize((256, 256), Image.Resampling.LANCZOS)  # Resize to match model input
                
                if self.model is None:
                    messagebox.showerror("Error", "Chưa có mô hình nào được tải. Vui lòng tải mô hình .pt hoặc .keras trước." if self.lang == "vi" else "No model loaded. Please load a .pt or .keras model first.")
                    return

                if self.model_type == 'yolo':
                    results = self.model(file_path)
                    pred = results[0].boxes
                    if pred:
                        class_id = int(pred.cls[0]) if pred.cls is not None else -1
                        confidence = float(pred.conf[0]) if pred.conf is not None else 0.0
                        label = results[0].names.get(class_id, "Unknown")
                        result_text = f"Dự đoán: {label} (Độ tin cậy: {confidence:.2f})" if self.lang == "vi" else f"Prediction: {label} (Confidence: {confidence:.2f})"

                        box = pred.xyxy[0].cpu().numpy()
                        x_min, y_min, x_max, y_max = box
                        orig_img = Image.open(file_path)
                        orig_width, orig_height = orig_img.size
                        scale_x = 256 / orig_width
                        scale_y = 256 / orig_height
                        x_min = int(x_min * scale_x)
                        y_min = int(y_min * scale_y)
                        x_max = int(x_max * scale_x)
                        y_max = int(y_max * scale_y)

                        draw = ImageDraw.Draw(img)
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="cyan", width=2)
                        label_text = f"{label}: {confidence:.2f}"
                        try:
                            font = ImageFont.truetype("arial.ttf", 15)
                        except:
                            font = ImageFont.load_default()
                        text_bbox = draw.textbbox((x_min, y_min - 20), label_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([x_min, y_min - text_height - 5, x_min + text_width, y_min], fill="cyan")
                        draw.text((x_min, y_min - text_height - 2), label_text, fill="black", font=font)
                    else:
                        result_text = "Không phát hiện khối u" if self.lang == "vi" else "No tumor detected"
                else:  # Keras model
                    img_array = np.array(img) / 255.0
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    img_array = np.expand_dims(img_array, axis=0)
                    pred = self.model.predict(img_array)
                    pred_mask = tf.squeeze(pred).numpy()
                    confidence = np.mean(pred_mask)  # Simplified confidence metric
                    label = "Có Khối u" if confidence > 0.5 else "Không có Khối u" if self.lang == "vi" else "No Tumor" if confidence > 0.5 else " Tumor"
                    result_text = f"Dự đoán: {label} (Độ tin cậy: {confidence:.2f})" if self.lang == "vi" else f"Prediction: {label} (Confidence: {confidence:.2f})"

                    # Overlay predicted mask
                    draw = ImageDraw.Draw(img)
                    mask_array = (pred_mask > 0.5).astype(np.uint8) * 255
                    mask_image = Image.fromarray(mask_array).resize((256, 256), Image.Resampling.NEAREST)
                    mask_array = np.array(mask_image)
                    overlay = np.zeros_like(np.array(img))
                    overlay[:, :, 2] = mask_array  # Blue channel for mask
                    overlay_image = Image.fromarray((np.array(img) * 0.7 + overlay * 0.3).astype(np.uint8))
                    img = overlay_image

                img_tk = ImageTk.PhotoImage(img)
                self.image_label.configure(image=img_tk, text="")
                self.image_label.image = img_tk
                self.result_label.configure(text=result_text)
                self.diagnosis_entry.delete(0, tk.END)
                self.diagnosis_entry.insert(0, result_text)

            except Exception as e:
                messagebox.showerror("Error", f"Không thể xử lý ảnh:\n{e}" if self.lang == "vi" else f"Failed to process image:\n{e}")

    def switch_tab(self, value):
        if value in ["Main", "Chính"]:
            self.history_frame.pack_forget()
            self.main_content.pack(fill="both", expand=True, padx=10, pady=10)
        else:
            self.main_content.pack_forget()
            self.history_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def toggle_dark_mode(self):
        ctk.set_appearance_mode("Dark" if self.dark_mode_switch.get() else "Light")

    def switch_language(self):
        if self.lang == "en":
            self.lang = "vi"
            self.upload_button.configure(text="Tải ảnh MRI")
            self.load_model_button.configure(text="Tải Mô hình")
            self.dark_mode_switch.configure(text="Chế độ Tối")
            self.lang_button.configure(text="English")
            self.title_label.configure(text="Hệ thống Phát hiện Khối u MRI")
            self.history_label.configure(text="Lịch sử Chẩn đoán")
            self.name_entry.configure(placeholder_text="Tên")
            self.age_entry.configure(placeholder_text="Tuổi")
            self.diagnosis_entry.configure(placeholder_text="Chẩn đoán")
            self.save_button.configure(text="Lưu Hồ sơ")
            self.tab_control.configure(values=["Chính", "Lịch sử"])
            self.tab_control.set("Chính")
        else:
            self.lang = "en"
            self.upload_button.configure(text="Upload MRI")
            self.load_model_button.configure(text="Load Model")
            self.dark_mode_switch.configure(text="Dark Mode")
            self.lang_button.configure(text="Tiếng Việt")
            self.title_label.configure(text="MRI Tumor Detection System")
            self.history_label.configure(text="Diagnosis History")
            self.name_entry.configure(placeholder_text="Name")
            self.age_entry.configure(placeholder_text="Age")
            self.diagnosis_entry.configure(placeholder_text="Diagnosis")
            self.save_button.configure(text="Save Record")
            self.tab_control.configure(values=["Main", "History"])
            self.tab_control.set("Main")

    def save_record(self):
        name = self.name_entry.get()
        age = self.age_entry.get()
        diagnosis = self.diagnosis_entry.get()
        if not name or not age or not diagnosis:
            messagebox.showwarning("Warning", "Vui lòng điền đầy đủ thông tin." if self.lang == "vi" else "Please fill in all fields.")
            return
        record = {
            "name": name,
            "age": age,
            "diagnosis": diagnosis,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(record)
        with open("history.json", "w") as f:
            json.dump(self.history, f, indent=4)
        self.update_history_display()
        messagebox.showinfo("Success", "Hồ sơ đã được lưu thành công." if self.lang == "vi" else "Record saved successfully.")

    def update_history_display(self):
        self.history_text.delete("1.0", tk.END)
        for record in self.history:
            text = f"Tên: {record['name']}\nTuổi: {record['age']}\nChẩn đoán: {record['diagnosis']}\nThời gian: {record['timestamp']}\n{'-'*50}\n" if self.lang == "vi" else f"Name: {record['name']}\nAge: {record['age']}\nDiagnosis: {record['diagnosis']}\nTime: {record['timestamp']}\n{'-'*50}\n"
            self.history_text.insert(tk.END, text)

def weighted_dice_loss(y_true, y_pred):
    """Weighted Dice loss for imbalanced segmentation"""
    pos_weight = 20.0
    smooth = 1e-5
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    weight_mask = y_true_f * (pos_weight - 1.0) + 1.0
    weighted_intersection = tf.reduce_sum(weight_mask * y_true_f * y_pred_f)
    weighted_union = tf.reduce_sum(weight_mask * y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * weighted_intersection + smooth) / (weighted_union + smooth)
    return 1 - dice

if __name__ == "__main__":
    app = TumorDetectionApp()
    app.mainloop()
