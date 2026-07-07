#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# Interfaz grafica del clasificador de colores
# ----------------------------------------------
#
# Dos pestanas:
#   - "Imagen": boton para elegir un archivo de imagen desde tu compu y
#     detectar su color dominante. Opcionalmente puedes arrastrar el mouse
#     sobre ella para analizar solo una region.
#   - "Camara": enciende la webcam dentro de la misma ventana y muestra la
#     deteccion en vivo. Incluye una casilla para analizar solo la zona
#     central (recomendado: evita que el fondo/mano contaminen la lectura)
#     o el frame completo si lo prefieres.
#
# Usa las mismas funciones de color_recognition_api que los otros scripts,
# asi que cualquier mejora a la extraccion de caracteristicas o al
# clasificador aplica automaticamente aqui tambien.
# ----------------------------------------------

import os
import queue
import threading
from collections import Counter, deque

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier

# Aseguramos que las rutas relativas (training.data, test.data) apunten
# siempre a la carpeta src/, sin importar desde donde se ejecute el script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

TRAINING_DATA_PATH = './training.data'

# Mismos parametros que color_classification_webcam.py
BLUR_THRESHOLD = 60
MIN_CONTRAST = 8
SMOOTHING_WINDOW = 7
PROCESS_EVERY_N_FRAMES = 2

# --- Paleta de la interfaz ---
BG_DARK = '#1b1e26'
BG_PANEL = '#242832'
BG_VIDEO = '#11131a'
ACCENT = '#5b8def'
ACCENT_DARK = '#4470c4'
TEXT_LIGHT = '#eef1f6'
TEXT_MUTED = '#9aa2b1'
DANGER = '#e6584c'
OK_GREEN = '#3ecf8e'
FONT_FAMILY = 'Segoe UI'

# Color aproximado para mostrar una muestra visual de cada etiqueta predicha
COLOR_SWATCHES = {
    'red': '#e53935', 'orange': '#fb8c00', 'yellow': '#fdd835', 'green': '#43a047',
    'blue': '#1e88e5', 'violet': '#8e24aa', 'white': '#f5f5f5', 'black': '#202124',
    'grey': '#9e9e9e', 'gray': '#9e9e9e', 'pink': '#f06292', 'cafe': '#6d4c41',
    'brown': '#6d4c41',
}


def ensure_training_data():
    """Genera training.data si no existe o esta vacio (una sola vez)."""
    if os.path.isfile(TRAINING_DATA_PATH) and os.path.getsize(TRAINING_DATA_PATH) > 0:
        return
    if os.path.isfile(TRAINING_DATA_PATH):
        os.remove(TRAINING_DATA_PATH)
    color_histogram_feature_extraction.training()


def is_blurry(gray_image, threshold=BLUR_THRESHOLD, min_contrast=MIN_CONTRAST):
    # Una superficie lisa (poco contraste) no permite medir enfoque de forma
    # confiable, se asume que esta bien enfocada en ese caso.
    if gray_image.std() < min_contrast:
        return False
    return cv2.Laplacian(gray_image, cv2.CV_64F).var() < threshold


def bgr_to_photoimage(bgr_image, max_size):
    """Convierte una imagen de OpenCV (BGR) a un ImageTk.PhotoImage que cabe
    dentro de un cuadro de max_size x max_size, manteniendo la proporcion."""
    h, w = bgr_image.shape[:2]
    scale = max_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(bgr_image, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img), scale


def swatch_color_for(label):
    return COLOR_SWATCHES.get(label.lower(), '#555b6b')


def configure_style():
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    style.configure('TNotebook', background=BG_DARK, borderwidth=0, tabmargins=(8, 8, 8, 0))
    style.configure('TNotebook.Tab', background=BG_PANEL, foreground=TEXT_MUTED,
                     padding=(18, 10), font=(FONT_FAMILY, 11, 'bold'), borderwidth=0)
    style.map('TNotebook.Tab',
              background=[('selected', ACCENT)],
              foreground=[('selected', '#ffffff')])

    style.configure('TFrame', background=BG_DARK)
    style.configure('Panel.TFrame', background=BG_PANEL)

    style.configure('TLabel', background=BG_DARK, foreground=TEXT_LIGHT, font=(FONT_FAMILY, 10))
    style.configure('Muted.TLabel', background=BG_DARK, foreground=TEXT_MUTED, font=(FONT_FAMILY, 9))
    style.configure('Title.TLabel', background=BG_DARK, foreground=TEXT_LIGHT, font=(FONT_FAMILY, 13, 'bold'))
    style.configure('Panel.TLabel', background=BG_PANEL, foreground=TEXT_MUTED, font=(FONT_FAMILY, 9))

    style.configure('Accent.TButton', background=ACCENT, foreground='#ffffff',
                     font=(FONT_FAMILY, 10, 'bold'), padding=(14, 8), borderwidth=0)
    style.map('Accent.TButton', background=[('active', ACCENT_DARK), ('disabled', '#3a3f4d')])

    style.configure('Ghost.TButton', background=BG_PANEL, foreground=TEXT_LIGHT,
                     font=(FONT_FAMILY, 10), padding=(14, 8), borderwidth=1)
    style.map('Ghost.TButton', background=[('active', '#2c313d'), ('disabled', '#23262e')])

    style.configure('TCheckbutton', background=BG_DARK, foreground=TEXT_LIGHT, font=(FONT_FAMILY, 10))
    style.map('TCheckbutton', background=[('active', BG_DARK)])

    return style


# ---------------------------------------------------------------------------
# Widget reutilizable: muestra de color + texto de prediccion
# ---------------------------------------------------------------------------

class PredictionBadge(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style='Panel.TFrame', padding=14)
        self.swatch = tk.Canvas(self, width=46, height=46, highlightthickness=0, bg=BG_PANEL)
        self.swatch_rect = self.swatch.create_oval(3, 3, 43, 43, fill='#555b6b', outline='')
        self.swatch.pack(side='left', padx=(0, 14))

        text_frame = ttk.Frame(self, style='Panel.TFrame')
        text_frame.pack(side='left', fill='both', expand=True)
        ttk.Label(text_frame, text='PREDICCION', style='Panel.TLabel').pack(anchor='w')
        self.text_var = tk.StringVar(value='-')
        self.text_label = tk.Label(text_frame, textvariable=self.text_var, font=(FONT_FAMILY, 20, 'bold'),
                                    bg=BG_PANEL, fg=TEXT_LIGHT, anchor='w')
        self.text_label.pack(anchor='w', fill='x')

    def set_result(self, label, color_hex=None, text_color=TEXT_LIGHT):
        self.text_var.set(label)
        self.text_label.config(fg=text_color)
        self.swatch.itemconfig(self.swatch_rect, fill=color_hex or '#555b6b')

    def set_neutral(self, text):
        self.set_result(text, color_hex='#555b6b', text_color=TEXT_MUTED)


# ---------------------------------------------------------------------------
# Pestana 1: deteccion sobre una imagen subida por el usuario
# ---------------------------------------------------------------------------

class ImageTab(ttk.Frame):
    CANVAS_SIZE = 520

    def __init__(self, master):
        super().__init__(master, padding=16)

        self.image_bgr = None       # imagen original cargada (sin redimensionar)
        self.display_scale = 1.0    # factor imagen_mostrada -> imagen_original
        self.roi = None             # (x1,y1,x2,y2) en coordenadas de la imagen ORIGINAL
        self.drag_start = None
        self.tk_img = None
        self.rect_id = None

        ttk.Label(self, text='Detectar color en una imagen', style='Title.TLabel').pack(anchor='w', pady=(0, 4))
        ttk.Label(
            self,
            text='Sube una foto y opcionalmente arrastra el mouse sobre ella para elegir solo una parte.',
            style='Muted.TLabel',
        ).pack(anchor='w', pady=(0, 12))

        controls = ttk.Frame(self)
        controls.pack(fill='x', pady=(0, 10))
        ttk.Button(controls, text='Subir imagen...', style='Accent.TButton', command=self.select_image).pack(side='left', padx=(0, 8))
        ttk.Button(controls, text='Detectar color', style='Accent.TButton', command=self.detect).pack(side='left', padx=8)
        ttk.Button(controls, text='Quitar seleccion', style='Ghost.TButton', command=self.clear_roi).pack(side='left', padx=8)

        canvas_wrap = tk.Frame(self, bg='#000000')
        canvas_wrap.pack()
        self.canvas = tk.Canvas(canvas_wrap, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
                                 bg=BG_VIDEO, highlightthickness=1, highlightbackground='#3a3f4d')
        self.canvas.pack(padx=1, pady=1)
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        self.badge = PredictionBadge(self)
        self.badge.pack(fill='x', pady=(16, 0))

        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete('all')
        self.canvas.create_text(
            self.CANVAS_SIZE // 2, self.CANVAS_SIZE // 2,
            text='Sube una imagen para empezar',
            fill=TEXT_MUTED, font=(FONT_FAMILY, 12),
        )

    def select_image(self):
        path = filedialog.askopenfilename(
            title='Selecciona una imagen',
            filetypes=[('Imagenes', '*.jpg *.jpeg *.png *.bmp'), ('Todos los archivos', '*.*')],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Error', f'No se pudo leer la imagen:\n{path}')
            return
        self.image_bgr = img
        self.roi = None
        self.badge.set_neutral('-')
        self.render()

    def render(self):
        if self.image_bgr is None:
            self._draw_placeholder()
            return
        self.tk_img, self.display_scale = bgr_to_photoimage(self.image_bgr, self.CANVAS_SIZE)
        self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
        self.rect_id = None
        if self.roi:
            x1, y1, x2, y2 = [v * self.display_scale for v in self.roi]
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=OK_GREEN, width=2)

    def on_press(self, event):
        if self.image_bgr is None:
            return
        self.drag_start = (event.x, event.y)

    def on_drag(self, event):
        if self.image_bgr is None or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline=OK_GREEN, width=2)

    def on_release(self, event):
        if self.image_bgr is None or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None

        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        if x_max - x_min < 5 or y_max - y_min < 5:
            self.roi = None  # seleccion demasiado chica, se ignora
            self.render()
            return

        s = self.display_scale
        h, w = self.image_bgr.shape[:2]
        self.roi = (
            max(0, x_min / s), max(0, y_min / s),
            min(w, x_max / s), min(h, y_max / s),
        )

    def clear_roi(self):
        self.roi = None
        self.render()

    def detect(self):
        if self.image_bgr is None:
            messagebox.showinfo('Sin imagen', 'Primero sube una imagen.')
            return

        if self.roi:
            x1, y1, x2, y2 = [int(v) for v in self.roi]
            region = self.image_bgr[y1:y2, x1:x2]
            if region.size == 0:
                region = self.image_bgr
        else:
            region = self.image_bgr

        try:
            ensure_training_data()
            color_histogram_feature_extraction.color_histogram_of_test_image(region, use_center_roi=False)
            prediction = knn_classifier.main(TRAINING_DATA_PATH, 'test.data')
            self.badge.set_result(prediction, swatch_color_for(prediction))
        except Exception as exc:  # noqa: BLE001 - se muestra cualquier error al usuario
            messagebox.showerror('Error al clasificar', str(exc))


# ---------------------------------------------------------------------------
# Pestana 2: deteccion en vivo desde la webcam
# ---------------------------------------------------------------------------

class CameraTab(ttk.Frame):
    DISPLAY_MAX_WIDTH = 640
    ROI_RATIO = 0.45  # fraccion de la dimension menor que cubre la zona central

    def __init__(self, master):
        super().__init__(master, padding=16)

        self.cap = None
        self.running = False
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=2)

        self.recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
        self.stable_prediction = 'n.a.'
        self.frame_count = 0
        self.use_center_only = tk.BooleanVar(value=True)

        ttk.Label(self, text='Deteccion de color en vivo', style='Title.TLabel').pack(anchor='w', pady=(0, 4))
        ttk.Label(
            self,
            text='Si hay fondo, mano u otros objetos en camara, activa "solo el centro" para leer\n'
                 'unicamente lo que este dentro del recuadro y evitar lecturas mezcladas.',
            style='Muted.TLabel', justify='left',
        ).pack(anchor='w', pady=(0, 12))

        controls = ttk.Frame(self)
        controls.pack(fill='x', pady=(0, 10))
        self.start_btn = ttk.Button(controls, text='Iniciar camara', style='Accent.TButton', command=self.start_camera)
        self.start_btn.pack(side='left', padx=(0, 8))
        self.stop_btn = ttk.Button(controls, text='Detener camara', style='Ghost.TButton', command=self.stop_camera, state='disabled')
        self.stop_btn.pack(side='left', padx=8)
        ttk.Checkbutton(
            controls, text='Analizar solo el centro (recomendado)',
            variable=self.use_center_only,
        ).pack(side='left', padx=(16, 0))

        video_wrap = tk.Frame(self, bg='#000000')
        video_wrap.pack()
        # Importante: si a un Label de Tkinter se le pone width/height en
        # numero (pensando en pixeles) ANTES de tener una imagen asignada,
        # Tkinter los interpreta como unidades de TEXTO (caracteres/lineas),
        # no pixeles, y el widget queda diminuto aunque luego se le ponga
        # una imagen grande. La forma correcta de reservar el espacio es
        # asignarle desde el inicio una imagen placeholder del tamano real
        # que va a tener el video.
        placeholder = Image.new('RGB', (self.DISPLAY_MAX_WIDTH, 480), color=(17, 19, 26))
        self.placeholder_img = ImageTk.PhotoImage(placeholder)
        self.video_label = tk.Label(video_wrap, image=self.placeholder_img, bg=BG_VIDEO, bd=0)
        self.video_label.pack(padx=1, pady=1)

        self.badge = PredictionBadge(self)
        self.badge.pack(fill='x', pady=(16, 0))

    def start_camera(self):
        if self.running:
            return

        try:
            ensure_training_data()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror('Error', str(exc))
            return

        cap = None
        for i in range(10):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    cap = test_cap
                    break
                test_cap.release()

        if cap is None:
            messagebox.showerror('Camara no encontrada', 'No se pudo abrir ninguna camara. Revisa la conexion (o DroidCam).')
            return

        # Intento de activar autoenfoque. No todas las camaras/drivers (por
        # ejemplo DroidCam por red) soportan controlar esto desde OpenCV, asi
        # que si falla simplemente se ignora sin afectar el resto.
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass

        self.cap = cap
        self.running = True
        self.stop_event.clear()
        self.recent_predictions.clear()
        self.stable_prediction = 'n.a.'
        self.frame_count = 0

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.after(15, self._update_gui)

    def _capture_loop(self):
        # Corre en un hilo aparte para que leer la camara no trabe la
        # interfaz. Solo mete frames a una cola; Tkinter los consume desde
        # el hilo principal (Tkinter no es seguro de actualizar desde otro
        # hilo).
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _update_gui(self):
        if not self.running:
            return

        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            self.after(15, self._update_gui)
            return

        self.frame_count += 1
        center_mode = self.use_center_only.get()

        # Zona a analizar: el centro (si esta activado) o el frame completo
        if center_mode:
            x1, y1, x2, y2 = color_histogram_feature_extraction.get_center_roi(frame, roi_ratio=self.ROI_RATIO)
            analysis_region = frame[y1:y2, x1:x2]
        else:
            analysis_region = frame

        status_text = self.stable_prediction
        badge_color = swatch_color_for(self.stable_prediction) if self.stable_prediction != 'n.a.' else None
        text_color = TEXT_LIGHT

        if analysis_region.size > 0 and self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2GRAY)
            if is_blurry(gray):
                status_text = 'Desenfocado - ajusta el enfoque'
                text_color = DANGER
                badge_color = '#4a2f2f'
            else:
                try:
                    color_histogram_feature_extraction.color_histogram_of_test_image(analysis_region, use_center_roi=False)
                    prediction = knn_classifier.main(TRAINING_DATA_PATH, 'test.data')
                    self.recent_predictions.append(prediction)
                    self.stable_prediction = Counter(self.recent_predictions).most_common(1)[0][0]
                    status_text = self.stable_prediction
                    badge_color = swatch_color_for(status_text)
                except Exception:
                    pass

        self.badge.set_result(status_text, badge_color, text_color)

        # Dibujar guia visual del recuadro central cuando ese modo esta activo
        display_frame = frame
        if center_mode:
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (94, 207, 62), 2)

        tk_img, _ = bgr_to_photoimage(display_frame, self.DISPLAY_MAX_WIDTH)
        self.video_label.imgtk = tk_img  # referencia viva para que no la borre el garbage collector
        self.video_label.config(image=tk_img)

        self.after(15, self._update_gui)

    def stop_camera(self):
        self.running = False
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        self.cap = None

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.video_label.config(image=self.placeholder_img)
        self.badge.set_neutral('-')


class ColorClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Detector de colores')
        self.geometry('700x860')
        self.minsize(650, 740)
        self.configure(bg=BG_DARK)

        configure_style()

        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.image_tab = ImageTab(notebook)
        self.camera_tab = CameraTab(notebook)
        notebook.add(self.image_tab, text='  Imagen  ')
        notebook.add(self.camera_tab, text='  Camara  ')

        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def on_close(self):
        self.camera_tab.stop_camera()
        self.destroy()


if __name__ == '__main__':
    app = ColorClassifierApp()
    app.mainloop()
