#!/usr/bin/python
# -*- coding: utf-8 -*-
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
from color_recognition_api import focus_utils

# Aseguramos que las rutas relativas (training.data, test.data) apunten
# siempre a la carpeta src/, sin importar desde donde se ejecute el script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

TRAINING_DATA_PATH = './training.data'
ASSETS_DIR = os.path.join(SCRIPT_DIR, 'assets')

# Mismos parametros que color_classification_webcam.py
BLUR_THRESHOLD = 60
MIN_CONTRAST = 8
SMOOTHING_WINDOW = 7
PROCESS_EVERY_N_FRAMES = 2
MIN_CONFIDENCE = 40  # por debajo de este % de confianza, la lectura no se toma en cuenta

INSTRUCTIONS_TEXT = (
    "COMO USAR ESTA INTERFAZ\n\n"
    "Pestana 'Imagen'\n"
    "  1. Pulsa 'Subir imagen...' y elige una foto de tu computadora.\n"
    "  2. (Opcional) Arrastra el mouse sobre la imagen para marcar solo\n"
    "     la parte que quieres analizar, asi el fondo no afecta el resultado.\n"
    "  3. Pulsa 'Detectar color' para ver la prediccion y su confianza.\n"
    "  4. 'Quitar seleccion' vuelve a usar la imagen completa.\n\n"
    "Pestana 'Camara'\n"
    "  1. Pulsa 'Iniciar camara'.\n"
    "  2. Coloca el objeto a identificar dentro del recuadro verde central,\n"
    "     ocupando la mayor parte posible del recuadro.\n"
    "  3. Observa la barra de 'Enfoque': si esta baja (naranja/roja), aleja\n"
    "     o acerca el objeto, mejora la luz, o pulsa 'Reenfocar'.\n"
    "  4. La prediccion y su confianza (%) aparecen debajo del video.\n"
    "     Una confianza baja significa que la lectura es poco confiable\n"
    "     (prueba con mejor luz o centrando mejor el objeto).\n"
    "  5. Desmarca 'Analizar solo el centro' si prefieres usar el frame\n"
    "     completo (no recomendado si hay varios colores en camara).\n"
    "  6. 'Detener camara' apaga la webcam.\n\n"
    "Consejos para mejores resultados\n"
    "  - Usa luz uniforme (evita contraluz y sombras fuertes sobre el objeto).\n"
    "  - Evita superficies muy brillantes/reflectantes.\n"
    "  - Mantén la camara quieta unos segundos para que la prediccion se\n"
    "    estabilice (se promedian varias lecturas recientes).\n"
)

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


def load_logo_image(filename, target_height):
    """Carga un logo desde src/assets/, lo escala a target_height de alto
    (manteniendo proporcion) y devuelve un ImageTk.PhotoImage listo para
    usar en un Label. Si el archivo no existe, devuelve None sin romper
    la interfaz (por si algun logo llegara a faltar)."""
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.isfile(path):
        return None
    img = Image.open(path).convert('RGBA')
    w, h = img.size
    scale = target_height / h
    new_size = (max(1, int(w * scale)), target_height)
    img = img.resize(new_size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


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
            prediction, confidence = knn_classifier.main(TRAINING_DATA_PATH, 'test.data')
            self.badge.set_result(f'{prediction} ({confidence:.0f}%)', swatch_color_for(prediction))
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
        self.last_confidence = 0.0
        self.frame_count = 0
        self.use_center_only = tk.BooleanVar(value=True)
        self.request_refocus = threading.Event()
        self._refocus_just_completed = False
        self.refocus_status = tk.StringVar(value='')

        ttk.Label(self, text='Deteccion de color en vivo', style='Title.TLabel').pack(anchor='w', pady=(0, 4))
        ttk.Label(
            self,
            text='Si hay fondo, mano u otros objetos en camara, activa "solo el centro" para leer\n'
                 'unicamente lo que este dentro del recuadro y evitar lecturas mezcladas. Coloca el\n'
                 'objeto llenando el recuadro y observa la barra de enfoque debajo del video.',
            style='Muted.TLabel', justify='left',
        ).pack(anchor='w', pady=(0, 12))

        controls = ttk.Frame(self)
        controls.pack(fill='x', pady=(0, 10))
        self.start_btn = ttk.Button(controls, text='Iniciar camara', style='Accent.TButton', command=self.start_camera)
        self.start_btn.pack(side='left', padx=(0, 8))
        self.stop_btn = ttk.Button(controls, text='Detener camara', style='Ghost.TButton', command=self.stop_camera, state='disabled')
        self.stop_btn.pack(side='left', padx=8)
        self.refocus_btn = ttk.Button(controls, text='Reenfocar', style='Ghost.TButton', command=self.request_refocus_now, state='disabled')
        self.refocus_btn.pack(side='left', padx=8)
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

        focus_row = ttk.Frame(self)
        focus_row.pack(fill='x', pady=(10, 0))
        ttk.Label(focus_row, text='Enfoque:', style='Muted.TLabel').pack(side='left', padx=(0, 8))
        self.focus_bar = ttk.Progressbar(focus_row, length=200, maximum=100, mode='determinate')
        self.focus_bar.pack(side='left')
        self.focus_pct_label = ttk.Label(focus_row, text='0%', style='Muted.TLabel')
        self.focus_pct_label.pack(side='left', padx=(8, 0))
        ttk.Label(focus_row, textvariable=self.refocus_status, style='Muted.TLabel').pack(side='left', padx=(16, 0))

        self.badge = PredictionBadge(self)
        self.badge.pack(fill='x', pady=(16, 0))

    def request_refocus_now(self):
        self.request_refocus.set()
        self.refocus_status.set('Recalibrando enfoque...')

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

        # Intento de activar autoenfoque continuo. No todas las camaras/
        # drivers (por ejemplo DroidCam por red) lo soportan desde OpenCV;
        # si falla, se ignora. El boton "Reenfocar" siempre queda disponible
        # como respaldo manual.
        focus_utils.try_enable_autofocus(cap)

        self.cap = cap
        self.running = True
        self.stop_event.clear()
        self.request_refocus.clear()
        self.recent_predictions.clear()
        self.stable_prediction = 'n.a.'
        self.last_confidence = 0.0
        self.frame_count = 0
        self.refocus_status.set('')

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.refocus_btn.config(state='normal')

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.after(15, self._update_gui)

    def _roi_gray_of(self, frame):
        x1, y1, x2, y2 = color_histogram_feature_extraction.get_center_roi(frame, roi_ratio=self.ROI_RATIO)
        return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    def _capture_loop(self):
        # Corre en un hilo aparte para que leer la camara no trabe la
        # interfaz. Solo mete frames a una cola; Tkinter los consume desde
        # el hilo principal (Tkinter no es seguro de actualizar desde otro
        # hilo).
        while not self.stop_event.is_set():
            if self.request_refocus.is_set():
                focus_utils.autofocus_sweep(self.cap, self._roi_gray_of)
                self.request_refocus.clear()
                self.recent_predictions.clear()
                # No se toca el StringVar de Tkinter desde este hilo; se
                # deja una bandera simple que _update_gui (hilo principal)
                # revisa para actualizar el texto de forma segura.
                self._refocus_just_completed = True
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

        if self._refocus_just_completed:
            self._refocus_just_completed = False
            self.refocus_status.set('Enfoque recalibrado')
            self.after(2500, lambda: self.refocus_status.set(''))

        # Zona a analizar: el centro (si esta activado) o el frame completo
        if center_mode:
            x1, y1, x2, y2 = color_histogram_feature_extraction.get_center_roi(frame, roi_ratio=self.ROI_RATIO)
            analysis_region = frame[y1:y2, x1:x2]
        else:
            analysis_region = frame

        status_text = self.stable_prediction
        badge_color = swatch_color_for(self.stable_prediction) if self.stable_prediction != 'n.a.' else None
        text_color = TEXT_LIGHT
        sharpness_pct = 0.0

        if analysis_region.size > 0:
            gray = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2GRAY)
            sharpness_pct = focus_utils.sharpness_percent(gray)

            if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
                if focus_utils.is_blurry(gray, threshold=BLUR_THRESHOLD, min_contrast=MIN_CONTRAST):
                    # Imagen desenfocada: no se actualiza el badge, se mantiene
                    # la ultima prediccion estable visible (la barra de nitidez
                    # ya le indica al usuario que debe ajustar el enfoque).
                    pass
                else:
                    try:
                        color_histogram_feature_extraction.color_histogram_of_test_image(analysis_region, use_center_roi=False)
                        prediction, confidence = knn_classifier.main(TRAINING_DATA_PATH, 'test.data')
                        if confidence >= MIN_CONFIDENCE:
                            self.recent_predictions.append(prediction)
                            self.last_confidence = confidence
                        if self.recent_predictions:
                            self.stable_prediction = Counter(self.recent_predictions).most_common(1)[0][0]
                        status_text = self.stable_prediction
                        badge_color = swatch_color_for(status_text)
                    except Exception:
                        pass

        self.focus_bar['value'] = sharpness_pct
        self.focus_pct_label.config(text=f'{int(sharpness_pct)}%')

        display_text = status_text
        if status_text != 'n.a.':
            display_text = f'{status_text} ({int(self.last_confidence)}%)'
        self.badge.set_result(display_text, badge_color, text_color)

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
        self.request_refocus.clear()
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        self.cap = None

        self.refocus_btn.config(state='disabled')
        self.refocus_status.set('')
        self.focus_bar['value'] = 0
        self.focus_pct_label.config(text='0%')
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.video_label.config(image=self.placeholder_img)
        self.badge.set_neutral('-')


class ColorClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Detector de colores')
        self.geometry('700x900')
        self.minsize(650, 780)
        self.configure(bg=BG_DARK)

        configure_style()

        # --- Fila superior: logo TecNM (izquierda) y logo ITSR (derecha) ---
        logos_row = ttk.Frame(self)
        logos_row.pack(fill='x', padx=10, pady=(10, 0))

        # Guardamos referencia en self para que el garbage collector no las borre
        self.logo_tecnm_img = load_logo_image('logo_tecnm.png', target_height=100)
        self.logo_itsr_img = load_logo_image('logo_itsr.png', target_height=100)

        if self.logo_tecnm_img:
            ttk.Label(logos_row, image=self.logo_tecnm_img, style='TLabel').pack(side='left')
        if self.logo_itsr_img:
            ttk.Label(logos_row, image=self.logo_itsr_img, style='TLabel').pack(side='right')

        # --- Fila de titulo y boton de ayuda, debajo de los logos ---
        header = ttk.Frame(self)
        header.pack(fill='x', padx=10, pady=(8, 0))
        ttk.Label(header, text='Detector de colores', style='Title.TLabel').pack(side='left')
        ttk.Button(header, text='Ayuda / Instrucciones', style='Ghost.TButton',
                   command=self.show_help).pack(side='right')

        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.image_tab = ImageTab(notebook)
        self.camera_tab = CameraTab(notebook)
        notebook.add(self.image_tab, text='  Imagen  ')
        notebook.add(self.camera_tab, text='  Camara  ')

        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def show_help(self):
        messagebox.showinfo('Como usar la interfaz', INSTRUCTIONS_TEXT)

    def on_close(self):
        self.camera_tab.stop_camera()
        self.destroy()


if __name__ == '__main__':
    app = ColorClassifierApp()
    app.mainloop()
