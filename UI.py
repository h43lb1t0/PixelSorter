import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from MaskHandler import YoloSegmentation
from PixelSorter import PixelSorter, Image, SortBy
from Enums import WhatToSort, SortDirection
from PIL import Image as PILImage, ImageTk, ImageDraw, ImageChops, ImageOps
import numpy as np
from tkinterdnd2 import DND_FILES, TkinterDnD

# --- DARK THEME CONSTANTS ---
BG_DARK = "#2e2e2e"
FG_LIGHT = "#ffffff"
BG_SECONDARY = "#3c3c3c"
BG_TERTIARY = "#4a4a4a"
ACCENT_COLOR = "#0078d7"  # Accent blue

class DarkDropdown(ttk.Frame):
    """
    A simple dark-themed dropdown (Menubutton + Menu) to replace Combobox popups
    so the menu entries follow the dark theme and remain readable.
    Provides a minimal API similar to Combobox:
      DarkDropdown(parent, textvariable=var, values=[...], width=...)
    """
    def __init__(self, parent, textvariable: tk.Variable, values, width=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.var = textvariable
        self.values = list(values)
        self.width = width

        # Visual Menubutton
        self.button = tk.Menubutton(self, text=self.var.get(), relief="raised",
                                   bg=BG_SECONDARY, fg=FG_LIGHT, activebackground=BG_TERTIARY,
                                   activeforeground=FG_LIGHT, indicatoron=True, borderwidth=1,
                                   anchor="w", padx=6)
        if self.width:
            self.button.config(width=self.width)
        self.button.pack(fill="x", expand=True)

        # Menu (the popup)
        self.menu = tk.Menu(self.button, tearoff=False, bg=BG_DARK, fg=FG_LIGHT,
                            activebackground=BG_TERTIARY, activeforeground=FG_LIGHT,
                            borderwidth=0)
        # Add items
        for val in self.values:
            # Use lambda with default arg to capture value
            self.menu.add_command(label=str(val), command=lambda v=val: self._set_and_update(v))
        self.button.config(menu=self.menu)

        # Update the button label when variable changes
        self.var.trace_add("write", self._on_var_change)

    def _set_and_update(self, value):
        self.var.set(value)
        # update button text as well
        self.button.config(text=str(value))

    def _on_var_change(self, *args):
        # Keep the button showing the current selection
        self.button.config(text=str(self.var.get()))

    # method kept for simple replacement compatibility if code expects widget
    def grid(self, *args, **kwargs):
        super().grid(*args, **kwargs)

    def pack(self, *args, **kwargs):
        super().pack(*args, **kwargs)

    def place(self, *args, **kwargs):
        super().place(*args, **kwargs)

class MaskEditor(tk.Toplevel):
    """
    An advanced mask editor with soft brush, image underlay, transparency,
    and mask inversion capabilities.
    """
    def __init__(self, parent, pil_mask_image, original_color_image, callback):
        super().__init__(parent, bg=BG_DARK)
        self.transient(parent)
        self.grab_set()
        self.title("Mask Editor")
        
        self.callback = callback
        self.original_mask = pil_mask_image.convert("L")
        self.original_color_image = original_color_image

        self._setup_scaling()
        
        self.background_image = self.original_color_image.resize(
            self.display_size, PILImage.Resampling.LANCZOS).convert("RGB")
        self.editable_image = self.original_mask.resize(
            self.display_size, PILImage.Resampling.LANCZOS)
        self.draw = ImageDraw.Draw(self.editable_image)
        
        self.brush_size = tk.IntVar(value=30)
        self.transparency = tk.DoubleVar(value=0.6)
        self.soft_brush_var = tk.BooleanVar(value=False)

        self.brush_cache = {}

        self._create_widgets()

        self.canvas.bind("<B1-Motion>", lambda e: self._paint(e, "white"))
        self.canvas.bind("<ButtonPress-1>", lambda e: self._paint(e, "white"))
        self.canvas.bind("<B3-Motion>", lambda e: self._paint(e, "black"))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._paint(e, "black"))
        self.canvas.bind("<MouseWheel>", self._adjust_brush_size_on_scroll)
        self.canvas.bind("<Button-4>", self._adjust_brush_size_on_scroll)
        self.canvas.bind("<Button-5>", self._adjust_brush_size_on_scroll)

    def _create_soft_brush(self, diameter):
        if diameter in self.brush_cache:
            return self.brush_cache[diameter]
        radius = diameter / 2
        x = np.arange(0, diameter) - radius
        y = np.arange(0, diameter) - radius
        xx, yy = np.meshgrid(x, y)
        distance = np.sqrt(xx**2 + yy**2)
        brush_map = 1.0 - (distance / radius)
        brush_map = np.clip(brush_map, 0, 1)
        brush_array = (brush_map * 255).astype(np.uint8)
        brush_image = PILImage.fromarray(brush_array, 'L')
        self.brush_cache[diameter] = brush_image
        return brush_image
        
    def _setup_scaling(self):
        max_w = int(self.winfo_screenwidth() * 0.9)
        max_h = int(self.winfo_screenheight() * 0.9 - 100)
        original_w, original_h = self.original_mask.size
        self.scale_factor = 1.0
        if original_w > max_w or original_h > max_h:
            self.scale_factor = min(max_w / original_w, max_h / original_h)
        self.display_size = (int(original_w * self.scale_factor), int(original_h * self.scale_factor))

    def _create_widgets(self):
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill="x")

        controls_frame = ttk.Frame(top_frame)
        controls_frame.pack(fill="x", expand=True, side="left")
        
        ttk.Label(controls_frame, text="Brush Size:").pack(side="left", padx=(0, 5))
        ttk.Scale(controls_frame, from_=1, to=200, variable=self.brush_size, orient="horizontal").pack(side="left", fill="x", expand=True)
        ttk.Label(controls_frame, text="Mask Opacity:").pack(side="left", padx=(10, 5))
        ttk.Scale(
            controls_frame, from_=0, to=1, variable=self.transparency, orient="horizontal",
            command=lambda e: self._update_canvas_image()
        ).pack(side="left", fill="x", expand=True)
        
        actions_frame = ttk.Frame(top_frame)
        actions_frame.pack(side="left", padx=(10,0))
        ttk.Checkbutton(actions_frame, text="Soft Brush", variable=self.soft_brush_var).pack(anchor="w")
        ttk.Button(actions_frame, text="Invert Mask", command=self._invert_mask).pack(anchor="w", pady=(5,0))

        canvas_w, canvas_h = self.display_size
        self.canvas = tk.Canvas(self, width=canvas_w, height=canvas_h, bg=BG_SECONDARY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw")

        buttons_frame = ttk.Frame(self, padding=10)
        buttons_frame.pack(fill="x")
        ttk.Label(buttons_frame, text="L-Click: Add | R-Click: Remove | Scroll: Brush Size").pack(side="left")
        ttk.Button(buttons_frame, text="Cancel", command=self.destroy).pack(side="right", padx=5)
        ttk.Button(buttons_frame, text="Save and Use Mask", command=self._save_and_close).pack(side="right")
        
        self._update_canvas_image()
        
    def _paint(self, event, color):
        size = self.brush_size.get()
        x, y = event.x, event.y
        if self.soft_brush_var.get():
            brush_img = self._create_soft_brush(size * 2)
            brush_layer = PILImage.new('L', self.editable_image.size, 0)
            paste_pos = (x - size, y - size)
            brush_layer.paste(brush_img, paste_pos)
            if color == 'white':
                self.editable_image = ImageChops.lighter(self.editable_image, brush_layer)
            else:
                inverted_brush_layer = PILImage.eval(brush_layer, lambda p: 255 - p)
                self.editable_image = ImageChops.darker(self.editable_image, inverted_brush_layer)
        else:
            x1, y1 = (x - size), (y - size)
            x2, y2 = (x + size), (y + size)
            self.draw = ImageDraw.Draw(self.editable_image)
            self.draw.ellipse([x1, y1, x2, y2], fill=color)
        self._update_canvas_image()

    def _adjust_brush_size_on_scroll(self, event):
        step = 0
        if hasattr(event, 'delta') and event.delta != 0: step = 1 if event.delta > 0 else -1
        elif event.num == 4: step = 5
        elif event.num == 5: step = -5
        if step != 0:
            current_size = self.brush_size.get()
            new_size = current_size + step
            clamped_size = max(1, min(200, new_size))
            self.brush_size.set(clamped_size)

    def _invert_mask(self):
        self.editable_image = ImageOps.invert(self.editable_image.convert('L'))
        self.draw = ImageDraw.Draw(self.editable_image)
        self._update_canvas_image()

    def _update_canvas_image(self):
        mask_rgb = self.editable_image.convert("RGB")
        alpha = self.transparency.get()
        blended_image = PILImage.blend(self.background_image, mask_rgb, alpha)
        self.photo_image = ImageTk.PhotoImage(blended_image)
        self.canvas.itemconfig(self.canvas_image_id, image=self.photo_image)

    def _save_and_close(self):
        if self.callback:
            if self.scale_factor < 1.0:
                final_mask = self.editable_image.resize(self.original_mask.size, PILImage.Resampling.NEAREST)
            else:
                final_mask = self.editable_image
            self.callback(final_mask)
        self.destroy()


class PixelSorterApp(TkinterDnD.Tk):
    PREVIEW_SIZE = (280, 280)
    def __init__(self):
        super().__init__()
        self.title("Pixel Sorter UI")
        # Set the window to fullscreen on high-res displays, but allow resizing
        self.geometry(f"{int(self.winfo_screenwidth()*0.9)}x{int(self.winfo_screenheight()*0.9)}")
        self.minsize(600, 1050)
        
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except Exception:
            self.style.theme_use(self.style.theme_use())
        
        # --- DARK THEME SETUP ---
        self.style.configure('.', background=BG_DARK, foreground=FG_LIGHT)
        self.style.configure('TFrame', background=BG_DARK)
        self.style.configure('TLabel', background=BG_DARK, foreground=FG_LIGHT)
        self.style.configure('TCheckbutton', background=BG_DARK, foreground=FG_LIGHT)
        self.style.configure('TRadiobutton', background=BG_DARK, foreground=FG_LIGHT)
        self.style.configure('TButton', background=BG_SECONDARY, foreground=FG_LIGHT)
        self.style.map('TButton', background=[('active', BG_TERTIARY)])
        
        # Configure Labelframe (title bar)
        self.style.configure('TLabelframe', background=BG_DARK, foreground=FG_LIGHT)
        self.style.configure('TLabelframe.Label', background=BG_DARK, foreground=FG_LIGHT)
        
        # Configure Entry (input fields)
        self.style.configure('TEntry', fieldbackground=BG_SECONDARY, foreground=FG_LIGHT, bordercolor=BG_TERTIARY)
        self.style.map('TEntry', fieldbackground=[('readonly', BG_TERTIARY)])
        
        # Configure Combobox fallback style (still used for any internal ttk combo visuals)
        self.style.configure('TCombobox', fieldbackground=BG_SECONDARY, selectbackground=ACCENT_COLOR, 
                             selectforeground=FG_LIGHT, background=BG_SECONDARY, foreground=FG_LIGHT)
        
        # Configure Scale (sliders)
        self.style.configure('Horizontal.TScale', background=BG_DARK, troughcolor=BG_TERTIARY)
        
        # Configure Status Bar
        self.style.configure('TStatus.TFrame', relief="flat", background=BG_SECONDARY)
        self.style.configure('TStatus.TLabel', background=BG_SECONDARY, foreground=FG_LIGHT)
        
        self.config(bg=BG_DARK) # Set the root window's background
        # --- END DARK THEME SETUP ---
        
        self.input_image_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()
        self.yolo_model_path = tk.StringVar()
        self.custom_mask_path = tk.StringVar()
        self.edited_mask_data = None
        self.sort_by_var = tk.StringVar(value=SortBy.list_static_methods()[0])
        self.sort_direction_var = tk.StringVar(value=SortDirection.COLUMN_BOTTOM_TO_TOP.name)
        self.use_perlin_var = tk.BooleanVar(value=False)
        self.mask_type_var = tk.StringVar(value="YOLO")
        self.what_to_sort_var = tk.StringVar(value=WhatToSort.BACKGROUND.name)
        self.yolo_conf_var = tk.DoubleVar(value=0.35)
        self.yolo_blur_include_var = tk.DoubleVar(value=0.7)
        self.yolo_blur_extend_var = tk.DoubleVar(value=0.7)
        self.yolo_conf_label_var = tk.StringVar(value=f"{self.yolo_conf_var.get():.2f}")
        self.yolo_blur_include_label_var = tk.StringVar(value=f"{self.yolo_blur_include_var.get():.2f}")
        self.yolo_blur_extend_label_var = tk.StringVar(value=f"{self.yolo_blur_extend_var.get():.2f}")
        self.status_var = tk.StringVar(value="Ready")
        self.output_dir_path.set(os.path.join(os.getcwd(), "output"))
        self._create_widgets()
        self._create_placeholder_images()
        self._update_mask_ui()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Use a flat background for better dark theme look
        io_frame = ttk.Labelframe(main_frame, text="File Input/Output", padding="10") 
        io_frame.pack(fill="x", expand=False, pady=5)
        io_frame.columnconfigure(1, weight=1)
        
        ttk.Label(io_frame, text="Input Image:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(io_frame, textvariable=self.input_image_path, state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Button(io_frame, text="Browse...", command=self._select_input_image).grid(row=0, column=2, padx=5)
        
        ttk.Label(io_frame, text="Output Dir:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(io_frame, textvariable=self.output_dir_path, state="readonly").grid(row=1, column=1, sticky="ew")
        ttk.Button(io_frame, text="Browse...", command=self._select_output_dir).grid(row=1, column=2, padx=5)
        
        preview_frame = ttk.Labelframe(main_frame, text="Previews", padding="10")
        preview_frame.pack(fill="x", expand=False, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        
        self.input_preview_label = tk.Label(preview_frame, text="Drop Image Here\n(or click to browse)", compound="top",
                                            bg=BG_SECONDARY, fg=FG_LIGHT, relief="ridge", bd=2, padx=6, pady=6, anchor="center")
        self.input_preview_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        # Bind click for fallback browse
        self.input_preview_label.bind("<Button-1>", lambda e: self._select_input_image())

        # If tkinterdnd2 is available, register drop target and bind
        try:
            self.input_preview_label.drop_target_register(DND_FILES)
            self.input_preview_label.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            # If registration fails, ignore and keep click-to-browse behavior
            pass

        self.output_preview_label = ttk.Label(preview_frame, text="Output Preview", compound="top", background=BG_SECONDARY)
        self.output_preview_label.grid(row=0, column=1, padx=5, pady=5)
        
        sort_frame = ttk.Labelframe(main_frame, text="Sorting Options", padding="10")
        sort_frame.pack(fill="x", expand=False, pady=5)
        sort_frame.columnconfigure(1, weight=1)
        
        sort_by_options = SortBy.list_static_methods()
        sort_direction_options = [attr for attr in dir(SortDirection) if not callable(getattr(SortDirection, attr)) and not attr.startswith("__")]
        
        ttk.Label(sort_frame, text="Sort By:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        # Replace Combobox with DarkDropdown
        dd_sort_by = DarkDropdown(sort_frame, textvariable=self.sort_by_var, values=sort_by_options, width=28)
        dd_sort_by.grid(row=0, column=1, sticky="ew", columnspan=2)
        
        ttk.Label(sort_frame, text="Sort Direction:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        dd_sort_dir = DarkDropdown(sort_frame, textvariable=self.sort_direction_var, values=sort_direction_options, width=28)
        dd_sort_dir.grid(row=1, column=1, sticky="ew", columnspan=2)
        
        ttk.Checkbutton(sort_frame, text="Use Perlin Noise", variable=self.use_perlin_var).grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        
        masking_frame = ttk.Labelframe(main_frame, text="Masking", padding="10")
        masking_frame.pack(fill="x", expand=False, pady=5)
        masking_frame.columnconfigure(0, weight=1)
        
        mask_type_frame = ttk.Frame(masking_frame)
        mask_type_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Radiobutton(mask_type_frame, text="YOLO", variable=self.mask_type_var, value="YOLO", command=self._update_mask_ui).pack(side="left", padx=5)
        ttk.Radiobutton(mask_type_frame, text="Create New", variable=self.mask_type_var, value="Create", command=self._update_mask_ui).pack(side="left", padx=5)
        ttk.Radiobutton(mask_type_frame, text="Custom Image", variable=self.mask_type_var, value="Custom", command=self._update_mask_ui).pack(side="left", padx=5)
        ttk.Radiobutton(mask_type_frame, text="None", variable=self.mask_type_var, value="None", command=self._update_mask_ui).pack(side="left", padx=5)
        
        self.edited_mask_radio = ttk.Radiobutton(mask_type_frame, text="Edited Mask", variable=self.mask_type_var, value="Edited", command=self._update_mask_ui)
        self.edited_mask_radio.pack(side="left", padx=5)
        self.edited_mask_radio.config(state="disabled")
        
        self.yolo_options_frame = ttk.Labelframe(masking_frame, text="YOLO Options", padding="10")
        self.yolo_options_frame.grid(row=1, column=0, sticky="ew")
        self.yolo_options_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.yolo_options_frame, text="Model Path (optional):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(self.yolo_options_frame, textvariable=self.yolo_model_path, state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Button(self.yolo_options_frame, text="Browse...", command=self._select_yolo_model).grid(row=0, column=2, padx=5)
        
        what_to_sort_options = [attr for attr in dir(WhatToSort) if not callable(getattr(WhatToSort, attr)) and not attr.startswith("__")]
        ttk.Label(self.yolo_options_frame, text="What to Sort:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        dd_what_to_sort = DarkDropdown(self.yolo_options_frame, textvariable=self.what_to_sort_var, values=what_to_sort_options, width=28)
        dd_what_to_sort.grid(row=1, column=1, columnspan=2, sticky="ew")
        
        self._create_slider_with_label(self.yolo_options_frame, 2, "Confidence:", self.yolo_conf_var, self.yolo_conf_label_var)
        self._create_slider_with_label(self.yolo_options_frame, 3, "Blur Include:", self.yolo_blur_include_var, self.yolo_blur_include_label_var)
        self._create_slider_with_label(self.yolo_options_frame, 4, "Blur Extend:", self.yolo_blur_extend_var, self.yolo_blur_extend_label_var)
        
        self.generate_edit_button = ttk.Button(self.yolo_options_frame, text="Generate & Edit Mask", command=self._start_mask_generation)
        self.generate_edit_button.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.create_mask_frame = ttk.Labelframe(masking_frame, text="Create Mask Options", padding="10")
        self.create_mask_frame.grid(row=1, column=0, sticky="ew")
        ttk.Button(self.create_mask_frame, text="Create & Edit Blank Mask", command=self._start_new_mask_creation).pack(pady=10)
        
        self.custom_mask_frame = ttk.Labelframe(masking_frame, text="Custom Mask Options", padding="10")
        self.custom_mask_frame.grid(row=1, column=0, sticky="ew")
        self.custom_mask_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.custom_mask_frame, text="Mask Image:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(self.custom_mask_frame, textvariable=self.custom_mask_path, state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Button(self.custom_mask_frame, text="Browse...", command=self._select_custom_mask).grid(row=0, column=2, padx=5)
        
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill="x", expand=False, pady=5)
        self.run_button = ttk.Button(action_frame, text="Run Pixel Sort", command=self._start_sorting_thread)
        self.run_button.pack(pady=10)
        
        status_bar = ttk.Frame(self, style='TStatus.TFrame', relief="flat", padding=(5, 2))
        status_bar.pack(side="bottom", fill="x")
        ttk.Label(status_bar, textvariable=self.status_var, style='TStatus.TLabel').pack(fill="x")

    def _on_drop(self, event):
        """
        Handler for <<Drop>> from tkinterdnd2. event.data often contains a string like:
          '{C:\\path\\to\\file1.jpg} {C:\\path\\to\\file2.png}'
        We'll take the first path, strip braces, and load it.
        """
        data = event.data
        if not data:
            return
        # On Windows and some other platforms paths may be grouped and wrapped in {}
        # Split on whitespace while respecting braced paths
        if data.startswith("{") and "}" in data:
            # naive parse: extract segments between braces
            paths = []
            cur = ""
            in_brace = False
            for ch in data:
                if ch == "{":
                    in_brace = True
                    cur = ""
                elif ch == "}":
                    in_brace = False
                    paths.append(cur)
                    cur = ""
                else:
                    if in_brace:
                        cur += ch
            if not paths:
                paths = data.split()
        else:
            paths = data.split()
        for p in paths:
            p = p.strip()
            if p:
                # On Linux the path may be URI-like (file:///...). handle that
                if p.startswith("file://"):
                    # Remove file:// and any leading /// (windows UNC may have file:///C:/..)
                    p = p[7:]
                    if p.startswith("/"):
                        # On Windows this will leave a leading / before drive letter: /C:/...
                        # Strip leading slash for windows style drive
                        if len(p) > 2 and p[2] == ":":
                            p = p[1:]
                # If the path exists, load it
                if os.path.exists(p):
                    self.input_image_path.set(p)
                    self._update_input_preview_from_path(p)
                    # reset edited mask etc to match Browse behavior
                    self.edited_mask_data = None
                    self.edited_mask_radio.config(state="disabled")
                    if self.mask_type_var.get() == "Edited":
                        self.mask_type_var.set("YOLO")
                    self._update_mask_ui()
                    break

    def _update_input_preview_from_path(self, path):
        photo_img = self._load_and_resize_image(path)
        if photo_img:
            self.input_photo_image = photo_img
            # For the drag'n'drop preview we used a tk.Label not ttk.Label, so use .config(image=...)
            self.input_preview_label.config(image=self.input_photo_image, text="")

    def _update_mask_ui(self, event=None):
        mask_type = self.mask_type_var.get()
        self.yolo_options_frame.grid_remove()
        self.custom_mask_frame.grid_remove()
        self.create_mask_frame.grid_remove()
        if mask_type == "YOLO":
            self.yolo_options_frame.grid()
        elif mask_type == "Custom":
            self.custom_mask_frame.grid()
        elif mask_type == "Create":
            self.create_mask_frame.grid()

    def _start_new_mask_creation(self):
        input_path = self.input_image_path.get()
        if not input_path:
            messagebox.showerror("Error", "Please select an input image first to define mask dimensions.")
            return
        try:
            self.status_var.set("Creating blank mask...")
            original_pil_image = PILImage.open(input_path)
            blank_mask = PILImage.new('L', original_pil_image.size, color='black')
            self._open_mask_editor(blank_mask, original_pil_image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image to get dimensions: {e}")
            self.status_var.set("Ready")

    def _start_mask_generation(self):
        if not self.input_image_path.get():
            messagebox.showerror("Error", "Please select an input image first.")
            return
        self.run_button.config(state="disabled")
        self.generate_edit_button.config(state="disabled")
        self.status_var.set("Generating YOLO mask...")
        self.update_idletasks()
        thread = threading.Thread(target=self._run_yolo_for_editing)
        thread.daemon = True
        thread.start()
        
    def _run_yolo_for_editing(self):
        try:
            input_path = self.input_image_path.get()
            image_for_library = Image.load_image(input_path)
            original_pil_image = PILImage.open(input_path)
            model_path = self.yolo_model_path.get() or "yolo12l-person-seg-extended.pt"
            what_to_sort_enum = getattr(WhatToSort, self.what_to_sort_var.get())
            segmenter = YoloSegmentation(image_for_library, model_path=model_path)
            mask_array = segmenter.get_mask(what_to_sort=what_to_sort_enum, conf=self.yolo_conf_var.get(),
                                             blur_include=self.yolo_blur_include_var.get(), blur_extend=self.yolo_blur_extend_var.get())
            if mask_array.dtype != np.uint8:
                mask_array = (mask_array * 255).astype(np.uint8)
            pil_mask_image = PILImage.fromarray(mask_array)
            self.after(0, self._open_mask_editor, pil_mask_image, original_pil_image)
        except Exception as e:
            self.status_var.set(f"Error during mask generation: {e}")
            messagebox.showerror("Mask Generation Error", f"An error occurred:\n{e}")
            self.run_button.config(state="normal")
            self.generate_edit_button.config(state="normal")

    def _open_mask_editor(self, pil_mask_image, original_pil_image):
        self.status_var.set("Mask generated. Opening editor...")
        MaskEditor(self, pil_mask_image, original_pil_image, callback=self._on_mask_edited)
        self.run_button.config(state="normal")
        self.generate_edit_button.config(state="normal")

    def _on_mask_edited(self, edited_pil_image):
        self.status_var.set("Edited mask saved. Ready to sort.")
        self.edited_mask_data = np.array(edited_pil_image.convert('L'))
        self.edited_mask_radio.config(state="normal")
        self.mask_type_var.set("Edited")
        self._update_mask_ui()
        
    def _run_sorting_logic(self):
        try:
            input_path = self.input_image_path.get()
            output_dir = self.output_dir_path.get()
            self.status_var.set("Loading image...")
            image = Image.load_image(input_path)
            mask_type = self.mask_type_var.get()
            mask = None
            if mask_type == "Edited":
                self.status_var.set("Using edited mask...")
                if self.edited_mask_data is None: raise ValueError("No edited mask data found.")
                mask = self.edited_mask_data
            elif mask_type == "YOLO":
                self.status_var.set("Running YOLO segmentation...")
                model_path = self.yolo_model_path.get() or "yolo12l-person-seg-extended.pt"
                what_to_sort_enum = getattr(WhatToSort, self.what_to_sort_var.get())
                segmenter = YoloSegmentation(image, model_path=model_path)
                mask = segmenter.get_mask(what_to_sort=what_to_sort_enum, conf=self.yolo_conf_var.get(),
                                             blur_include=self.yolo_blur_include_var.get(), blur_extend=self.yolo_blur_extend_var.get())
            elif mask_type == "Custom":
                self.status_var.set("Loading custom mask...")
                mask_path = self.custom_mask_path.get()
                if not mask_path: raise ValueError("Custom mask file path is not selected.")
                mask = Image.load_image(mask_path)
            self.status_var.set("Sorting pixels...")
            sort_by_func = getattr(SortBy, self.sort_by_var.get())()
            sort_direction_enum = getattr(SortDirection, self.sort_direction_var.get())
            sorted_image = PixelSorter(image).sort_pixels(sort_by=sort_by_func, direction=sort_direction_enum,
                                                         mask=mask, use_perlin=self.use_perlin_var.get())
            self.status_var.set("Saving output image...")
            input_image_name = os.path.splitext(os.path.basename(input_path))[0]
            output_filename = f"{input_image_name}_sorted.png"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            Image.save_image(sorted_image, output_path)
            self.after(0, self._update_output_preview, output_path)
            self.status_var.set(f"Success! Image saved to {output_path}")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Processing Error", f"An error occurred:\n{e}")
        finally:
            self.run_button.config(state="normal")
            self.generate_edit_button.config(state="normal")

    def _create_slider_with_label(self, parent, row, text, var, label_var):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        slider = ttk.Scale(parent, from_=0, to=1, variable=var, orient="horizontal", command=lambda val: label_var.set(f"{float(val):.2f}"), style='Horizontal.TScale')
        slider.grid(row=row, column=1, sticky="ew", padx=(0, 5))
        ttk.Label(parent, textvariable=label_var, width=5).grid(row=row, column=2, sticky="w")
        
    def _create_placeholder_images(self):
        placeholder = PILImage.new('RGB', self.PREVIEW_SIZE, (46, 46, 46)) 
        self.input_photo_image = ImageTk.PhotoImage(placeholder)
        self.output_photo_image = ImageTk.PhotoImage(placeholder)
        self.input_preview_label.config(image=self.input_photo_image)
        self.output_preview_label.config(image=self.output_photo_image)
        if isinstance(self.input_preview_label, tk.Label):
            self.input_preview_label.config(text="Drop Image Here\n(or click to browse)")

    def _load_and_resize_image(self, path):
        try:
            img = PILImage.open(path)
            img.thumbnail(self.PREVIEW_SIZE, PILImage.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            self.status_var.set(f"Error loading preview: {e}")
            return None
            
    def _select_input_image(self):
        path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self.input_image_path.set(path)
            photo_img = self._load_and_resize_image(path)
            if photo_img:
                self.input_photo_image = photo_img
                if isinstance(self.input_preview_label, tk.Label):
                    self.input_preview_label.config(image=self.input_photo_image, text="")
                else:
                    self.input_preview_label.config(image=self.input_photo_image)
            self.output_preview_label.config(image=self.output_photo_image)
            self.edited_mask_data = None
            self.edited_mask_radio.config(state="disabled")
            if self.mask_type_var.get() == "Edited": self.mask_type_var.set("YOLO")
            self._update_mask_ui()
            
    def _select_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path: self.output_dir_path.set(path)
        
    def _select_yolo_model(self):
        path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")])
        if path: self.yolo_model_path.set(path)
        
    def _select_custom_mask(self):
        path = filedialog.askopenfilename(title="Select Custom Mask Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All files", "*.*")])
        if path: self.custom_mask_path.set(path)
        
    def _start_sorting_thread(self):
        if not self.input_image_path.get() or not self.output_dir_path.get():
            messagebox.showerror("Error", "Please select an input image and an output directory.")
            return
        self.run_button.config(state="disabled")
        self.generate_edit_button.config(state="disabled")
        self.status_var.set("Processing... please wait.")
        self.update_idletasks()
        thread = threading.Thread(target=self._run_sorting_logic)
        thread.daemon = True
        thread.start()
        
    def _update_output_preview(self, image_path):
        photo_img = self._load_and_resize_image(image_path)
        if photo_img:
            self.output_photo_image = photo_img
            self.output_preview_label.config(image=self.output_photo_image)

if __name__ == "__main__":
    app = PixelSorterApp()
    app.mainloop()
