import os
import sys
import json
import hashlib
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pefile
import lief
import capstone.x86 as cs_x86
import capstone as cs
from capstone import Cs
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import numpy as np
import unicorn as uc
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import *
import time
import tensorflow as tf
import struct
import tempfile
import configparser

print("Current PATH:", os.environ['PATH'])
print("Files in unicorn dir:", os.listdir(os.path.join(sys._MEIPASS, 'unicorn')))

# Добавляем путь к библиотекам в PATH для PyInstaller
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
    lib_dirs = [
        os.path.join(base_dir, 'unicorn', 'lib'),
        os.path.join(base_dir, 'unicorn', 'lib64'),
        os.path.join(base_dir, 'capstone'),
        os.path.join(base_dir, 'keystone')
    ]
    for lib_dir in lib_dirs:
        if os.path.exists(lib_dir):
            os.environ['PATH'] = lib_dir + os.pathsep + os.environ['PATH']

# Явно установим среду TensorFlow для совместимости
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Проверка наличия keystone-engine для ассемблирования
try:
    from keystone import Ks, KS_ARCH_X86, KS_MODE_64

    HAS_KEYSTONE = True
except ImportError:
    HAS_KEYSTONE = False
    logging.warning("keystone-engine not installed. Assembly features will be disabled.")

# Константы для файлов
MNEMONICS_DIR = "MNEMONICS"
NEO_RABBIT_DIR = "NEO_RABBIT"
AI_MODEL_DIR = "AI_MODEL"
CONFIG_DIR = "CONFIG"
os.makedirs(MNEMONICS_DIR, exist_ok=True)
os.makedirs(NEO_RABBIT_DIR, exist_ok=True)
os.makedirs(AI_MODEL_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

JSON_FILES = [
    "AVX2_AVX.json", "F16C_AVX.json", "BMI2_AVX.json",
    "FMA_AVX.json", "BMI1_AVX.json"
]
MODEL_FILE = os.path.join(AI_MODEL_DIR, "transformer_model.keras")
TOKENIZER_FILE = os.path.join(AI_MODEL_DIR, "tokenizer.json")
CONFIG_FILE = os.path.join(AI_MODEL_DIR, "model_config.json")
FEEDBACK_FILE = os.path.join(AI_MODEL_DIR, "feedback_data.json")

# Конфигурационный файл для параметров модели
CONFIG_INI = os.path.join(CONFIG_DIR, "model_config.ini")
DEFAULT_CONFIG = {
    'AI_Model': {
        'vocab_size': '2000',
        'max_input_len': '30',
        'max_output_len': '150',
        'embed_dim': '128',
        'num_heads': '4',
        'ff_dim': '256',
        'epochs': '50',
        'batch_size': '32',
        'learning_rate': '0.001'
    }
}

# Создаем конфигурационный файл, если его нет
if not os.path.exists(CONFIG_INI):
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    with open(CONFIG_INI, 'w') as configfile:
        config.write(configfile)


class TransformerBlock(tf.keras.layers.Layer):
    """Кастомный блок трансформера для TensorFlow с исправленной сериализацией"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config


def calculate_file_hash(filepath):
    """Вычисляет SHA-256 хеш файла"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return None


class EnhancedInstructionAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Instruction Analyzer")
        self.root.geometry("1400x800")

        # Загрузка конфигурации модели
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_INI)
        self.model_config = self.config['AI_Model']

        # Параметры модели из конфига
        self.vocab_size = self.model_config.getint('vocab_size', 2000)
        self.max_input_len = self.model_config.getint('max_input_len', 30)
        self.max_output_len = self.model_config.getint('max_output_len', 150)
        self.embed_dim = self.model_config.getint('embed_dim', 128)
        self.num_heads = self.model_config.getint('num_heads', 4)
        self.ff_dim = self.model_config.getint('ff_dim', 256)
        self.epochs = self.model_config.getint('epochs', 50)
        self.batch_size = self.model_config.getint('batch_size', 32)
        self.learning_rate = self.model_config.getfloat('learning_rate', 0.001)

        # Конфигурация логирования
        logging.basicConfig(level=logging.DEBUG, filename="instruction_analyzer.log",
                            format="%(asctime)s - %(levelname)s - %(message)s")

        # Доступные наборы инструкций
        self.instruction_sets = ["SSE", "AVX", "AVX2", "F16C", "FMA", "BMI1", "BMI2"]

        # Загрузка мнемоник из файлов
        self.mnemonic_patterns = self.load_mnemonics_from_files()

        # Загрузка JSON-замен
        self.replacement_mappings = []
        self.json_files = JSON_FILES
        self.load_json_mappings()

        # Настройка нейросети
        self.model = None
        self.tokenizer = None
        self.model_file = MODEL_FILE
        self.tokenizer_file = TOKENIZER_FILE
        self.feedback_file = FEEDBACK_FILE
        self.config_file = CONFIG_FILE
        self.prediction_cache = {}
        self.context_cache = {}
        self.instruction_replacements = {}

        # Переменные для отслеживания обучения
        self.is_training = False
        self.training_dialog = None
        self.training_progressbar = None
        self.training_status_label = None

        # Очередь для межпоточного взаимодействия
        self.queue = queue.Queue()

        # Событие для остановки анализа
        self.stop_event = threading.Event()

        # Основной фрейм
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Фрейм для фильтра
        self.filter_frame = ttk.Frame(self.main_frame)
        self.filter_frame.pack(fill=tk.X, pady=(0, 5))

        self.filter_label = ttk.Label(self.filter_frame, text="Instruction Set Filter:")
        self.filter_label.pack(side=tk.LEFT, padx=(0, 5))

        self.filter_combobox = ttk.Combobox(self.filter_frame, values=self.instruction_sets, state="readonly",
                                             width=15)
        self.filter_combobox.set("SSE")
        self.filter_combobox.pack(side=tk.LEFT)

        # Фрейм для кнопок
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=(0, 5))

        self.open_button = ttk.Button(self.button_frame, text="Open File", command=self.open_file)
        self.open_button.pack(side=tk.LEFT, padx=3)

        self.analyze_button = ttk.Button(self.button_frame, text="Analyze File", command=self.start_analysis)
        self.analyze_button.pack(side=tk.LEFT, padx=3)
        self.analyze_button.config(state="disabled")

        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop_analysis)
        self.stop_button.pack(side=tk.LEFT, padx=3)
        self.stop_button.config(state="disabled")

        self.ai_replace_button = ttk.Button(self.button_frame, text="AI Replace", command=self.start_ai_replace)
        self.ai_replace_button.pack(side=tk.LEFT, padx=3)
        self.ai_replace_button.config(state="disabled")

        self.benchmark_button = ttk.Button(self.button_frame, text="Benchmark", command=self.run_benchmark)
        self.benchmark_button.pack(side=tk.LEFT, padx=3)
        self.benchmark_button.config(state="disabled")

        self.export_button = ttk.Button(self.button_frame, text="Export Binary", command=self.export_binary)
        self.export_button.pack(side=tk.LEFT, padx=3)
        self.export_button.config(state="disabled")

        # Кнопка для изменения параметров модели
        self.settings_button = ttk.Button(self.button_frame, text="Model Settings", command=self.open_settings)
        self.settings_button.pack(side=tk.LEFT, padx=3)

        # Прогресс-бар
        self.progress = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 5))

        # Notebook для разделения на вкладки
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка с оригинальными инструкциями
        self.original_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.original_tab, text="Original Instructions")

        # Treeview для оригинальных инструкций
        self.original_tree = ttk.Treeview(self.original_tab,
                                           columns=("Mnemonic", "Operand", "Size", "Bytes", "Address"),
                                           show="headings")

        original_columns = {
            "Mnemonic": {"text": "Mnemonic", "width": 100, "anchor": "w"},
            "Operand": {"text": "Operand", "width": 200, "anchor": "w"},
            "Size": {"text": "Size", "width": 50, "anchor": "center"},
            "Bytes": {"text": "Bytes", "width": 150, "anchor": "w"},
            "Address": {"text": "Address", "width": 100, "anchor": "w"}
        }

        for col, params in original_columns.items():
            self.original_tree.heading(col, text=params["text"])
            self.original_tree.column(col, width=params["width"], anchor=params["anchor"], stretch=True)

        # Полосы прокрутки
        self.original_v_scrollbar = ttk.Scrollbar(self.original_tab, orient=tk.VERTICAL, command=self.original_tree.yview)
        self.original_tree.configure(yscrollcommand=self.original_v_scrollbar.set)
        self.original_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.original_h_scrollbar = ttk.Scrollbar(self.original_tab, orient=tk.HORIZONTAL,
                                                   command=self.original_tree.xview)
        self.original_tree.configure(xscrollcommand=self.original_h_scrollbar.set)
        self.original_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.original_tree.pack(fill=tk.BOTH, expand=True)

        # Вкладка с замененными инструкциями
        self.replaced_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.replaced_tab, text="AI Replacement")

        # Treeview для замененных инструкций
        self.replaced_tree = ttk.Treeview(self.replaced_tab, columns=(
            "Mnemonic", "Operand", "Size", "Bytes", "Replacement", "Validated", "Speedup", "Address"),
                                           show="headings")

        replaced_columns = {
            "Mnemonic": {"text": "Mnemonic", "width": 100, "anchor": "w"},
            "Operand": {"text": "Operand", "width": 150, "anchor": "w"},
            "Size": {"text": "Size", "width": 50, "anchor": "center"},
            "Bytes": {"text": "Bytes", "width": 120, "anchor": "w"},
            "Replacement": {"text": "Replacement", "width": 200, "anchor": "w"},
            "Validated": {"text": "Validated", "width": 70, "anchor": "center"},
            "Speedup": {"text": "Speedup", "width": 70, "anchor": "center"},
            "Address": {"text": "Address", "width": 100, "anchor": "w"}
        }

        for col, params in replaced_columns.items():
            self.replaced_tree.heading(col, text=params["text"])
            self.replaced_tree.column(col, width=params["width"], anchor=params["anchor"], stretch=True)

        # Полосы прокрутки
        self.replaced_v_scrollbar = ttk.Scrollbar(self.replaced_tab, orient=tk.VERTICAL, command=self.replaced_tree.yview)
        self.replaced_tree.configure(yscrollcommand=self.replaced_v_scrollbar.set)
        self.replaced_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.replaced_h_scrollbar = ttk.Scrollbar(self.replaced_tab, orient=tk.HORIZONTAL,
                                                   command=self.replaced_tree.xview)
        self.replaced_tree.configure(xscrollcommand=self.replaced_h_scrollbar.set)
        self.replaced_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.replaced_tree.pack(fill=tk.BOTH, expand=True)

        # Статус-бар
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=(5, 0))
        self.status_labels = {}
        self.found_instructions = {k: 0 for k in self.instruction_sets}
        for inst_set in self.instruction_sets:
            label = ttk.Label(self.status_frame, text=f"{inst_set}: 0")
            label.pack(side=tk.LEFT, padx=10)
            self.status_labels[inst_set] = label

        # Контекстное меню
        self.context_menu = tk.Menu(self.root, tearoff=0, font=("Segoe UI", 10))
        self.context_menu.add_command(label="Save to File", command=self.save_to_file)
        self.replaced_tree.bind("<Button-3>", self.show_context_menu)

        # Состояние анализа
        self.is_analyzing = False
        self.selected_file = None
        self.original_bytes = None
        self.modified_bytes = None
        self.performance_data = {}
        self.group_mapping = {}
        self.instruction_addresses = {}

        # Загрузка модели в фоновом режиме
        threading.Thread(target=self.setup_neural_network, daemon=True).start()

        # Периодическая проверка очереди
        self.root.after(100, self.check_queue)

    def open_settings(self):
        """Открывает окно настроек модели"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Model Settings")
        settings_window.geometry("255x420")
        settings_window.resizable(False, False)
        settings_window.transient(self.root)
        settings_window.grab_set()

        # Центрируем окно
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        dialog_width = 255
        dialog_height = 420
        x = root_x + (root_width - dialog_width) // 2
        y = root_y + (root_height - dialog_height) // 2
        settings_window.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

        # Основной фрейм
        main_frame = ttk.Frame(settings_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Параметры для настройки
        params = [
            ("vocab_size", "Vocabulary Size:", 500, 10000),
            ("max_input_len", "Max Input Length:", 10, 100),
            ("max_output_len", "Max Output Length:", 20, 300),
            ("embed_dim", "Embedding Dimension:", 32, 512),
            ("num_heads", "Number of Attention Heads:", 1, 16),
            ("ff_dim", "Feed Forward Dimension:", 64, 1024),
            ("epochs", "Training Epochs:", 10, 200),
            ("batch_size", "Batch Size:", 8, 128),
            ("learning_rate", "Learning Rate:", 0.0001, 0.01, 0.0001)
        ]

        self.settings_vars = {}
        row = 0

        for param, label, min_val, max_val, *step in params:
            frame = ttk.Frame(main_frame)
            frame.pack(fill=tk.X, pady=5)

            lbl = ttk.Label(frame, text=label, width=20, anchor="e")
            lbl.pack(side=tk.LEFT, padx=5)

            current_val = self.model_config.getint(param) if param != "learning_rate" else self.model_config.getfloat(
                param)
            var = tk.StringVar(value=str(current_val))
            self.settings_vars[param] = var

            if len(step) > 0:
                spinbox = ttk.Spinbox(
                    frame,
                    from_=min_val,
                    to=max_val,
                    increment=step[0],
                    textvariable=var,
                    width=10
                )
            else:
                spinbox = ttk.Spinbox(
                    frame,
                    from_=min_val,
                    to=max_val,
                    textvariable=var,
                    width=10
                )
            spinbox.pack(side=tk.LEFT, padx=5)

        # Кнопки
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        save_btn = ttk.Button(
            btn_frame,
            text="Save Settings",
            command=lambda: self.save_settings(settings_window)
        )
        save_btn.pack(side=tk.LEFT, padx=10)

        cancel_btn = ttk.Button(
            btn_frame,
            text="Cancel",
            command=settings_window.destroy
        )
        cancel_btn.pack(side=tk.RIGHT, padx=10)

    def save_settings(self, window):
        """Сохраняет настройки и перезагружает модель"""
        try:
            for param, var in self.settings_vars.items():
                value = var.get()
                if param == "learning_rate":
                    self.config.set('AI_Model', param, str(float(value)))
                else:
                    self.config.set('AI_Model', param, str(int(value)))

            with open(CONFIG_INI, 'w') as configfile:
                self.config.write(configfile)

            messagebox.showinfo("Success", "Settings saved. Model will be reloaded.")
            window.destroy()

            # Перезагрузка модели
            self.model = None
            self.tokenizer = None
            threading.Thread(target=self.setup_neural_network, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_mnemonics_from_files(self):
        """Загрузка паттернов мнемоник из файлов в папке MNEMONICS"""
        mnemonics = {}
        for inst_set in self.instruction_sets:
            file_path = os.path.join(MNEMONICS_DIR, f"{inst_set}.txt")
            try:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        mnemonics[inst_set] = [line.strip().lower() for line in f.readlines() if line.strip()]
                    logging.info(f"Loaded mnemonics from {file_path}")
                else:
                    mnemonics[inst_set] = []
                    logging.warning(f"MNEMONICS file not found: {file_path}")
            except Exception as e:
                mnemonics[inst_set] = []
                logging.error(f"Failed to load {file_path}: {str(e)}")
        return mnemonics

    def load_json_mappings(self):
        """Загрузка JSON-замен с проверкой существования файлов"""
        self.replacement_mappings = []
        for json_file in self.json_files:
            file_path = os.path.join(NEO_RABBIT_DIR, json_file)
            try:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        self.replacement_mappings.extend(json.load(f))
                    logging.info(f"Loaded replacement mappings from {file_path}")
                else:
                    logging.warning(f"JSON file not found: {file_path}")
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {str(e)}")
                messagebox.showerror("Error", f"Failed to load {file_path}: {str(e)}")

    def calculate_json_hashes(self):
        """Вычисление хешей для JSON-файлов"""
        return {os.path.join(NEO_RABBIT_DIR, file): calculate_file_hash(os.path.join(NEO_RABBIT_DIR, file))
                for file in self.json_files}

    def save_model_config(self, hashes):
        """Сохранение конфигурации модели"""
        config = {
            "model_version": "1.0",
            "json_hashes": hashes,
            "model_file": self.model_file,
            "tokenizer_file": self.tokenizer_file,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_params": {
                "vocab_size": self.vocab_size,
                "max_input_len": self.max_input_len,
                "max_output_len": self.max_output_len,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim
            }
        }
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            logging.info("Model configuration saved")
        except Exception as e:
            logging.error(f"Failed to save model config: {str(e)}")

    def load_model_config(self):
        """Загрузка конфигурации модели"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load model config: {str(e)}")
        return None

    def json_files_changed(self):
        """Проверка изменений в JSON-файлах"""
        saved_config = self.load_model_config()
        if not saved_config:
            return True  # Конфиг отсутствует - файлы считаются измененными

        saved_hashes = saved_config.get("json_hashes", {})
        current_hashes = self.calculate_json_hashes()

        # Проверка различий в хешах
        for file in self.json_files:
            current_hash = current_hashes.get(file)
            saved_hash = saved_hashes.get(file)

            # Файл отсутствует или хеш изменился
            if current_hash != saved_hash:
                return True

        return False

    def prompt_for_retraining(self):
        """Запрос пользователя о переобучении модели"""
        response = messagebox.askyesno(
            "Model Update Required",
            "JSON files have been modified. Would you like to retrain the model?"
        )
        return response

    def setup_neural_network(self):
        """Инициализация и обучение нейросети с проверкой изменений"""
        # Загрузка обратной связи
        self.load_feedback_data()

        # Проверка необходимости обучения
        model_exists = os.path.exists(self.model_file)
        config_exists = os.path.exists(self.config_file)

        # Если модель существует, проверяем изменения в JSON
        if model_exists and config_exists:
            if self.json_files_changed():
                # Запрос в основном потоке
                self.queue.put(("retrain_prompt", None))
                return
            else:
                # Загрузка существующей модели
                self.load_model()
                return

        # Если модель не существует или требуется переобучение
        self.train_model()

    def load_model(self):
        """Загрузка предварительно обученной модели"""
        try:
            # Проверяем, что файл модели существует
            if not os.path.exists(self.model_file):
                raise FileNotFoundError(f"Model file not found: {self.model_file}")

            # Загружаем модель
            self.model = tf.keras.models.load_model(
                self.model_file,
                custom_objects={'TransformerBlock': TransformerBlock}
            )

            # Проверяем, что токенизатор существует
            if not os.path.exists(self.tokenizer_file):
                raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_file}")

            # Загружаем токенизатор
            with open(self.tokenizer_file, "r") as f:
                tokenizer_data = json.load(f)
                self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
                self.tokenizer.word_index = tokenizer_data

            logging.info("Loaded pre-trained transformer model and tokenizer")
            self.queue.put(("model_loaded", "Model loaded successfully"))

        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.queue.put(("model_error", f"Failed to load model: {str(e)}. Training new model."))
            self.train_model()

    def show_training_dialog(self):
        """Показывает диалог с прогрессом обучения"""
        self.training_dialog = tk.Toplevel(self.root)
        self.training_dialog.title("Training Neural Network")
        self.training_dialog.geometry("400x150")
        self.training_dialog.resizable(False, False)
        self.training_dialog.transient(self.root)
        self.training_dialog.grab_set()

        # Центрируем диалог
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        dialog_width = 400
        dialog_height = 150
        x = root_x + (root_width - dialog_width) // 2
        y = root_y + (root_height - dialog_height) // 2
        self.training_dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

        ttk.Label(self.training_dialog, text="Training transformer model...", font=("Segoe UI", 11)).pack(pady=10)

        self.training_progressbar = ttk.Progressbar(
            self.training_dialog,
            orient=tk.HORIZONTAL,
            length=350,
            mode="determinate"
        )
        self.training_progressbar.pack(pady=10)

        self.training_status = ttk.Label(
            self.training_dialog,
            text="Starting training...",
            font=("Segoe UI", 9)
        )
        self.training_status.pack(pady=5)

    def train_model(self):
        """Обучение новой модели с отображением прогресса"""
        if not self.replacement_mappings:
            logging.warning("No JSON mappings available for training neural network")
            self.queue.put(("training_error", "No JSON mappings available. Neural network will not be trained."))
            return

        # Показываем диалог обучения
        self.is_training = True
        self.queue.put(("show_training_dialog", None))

        try:
            # Подготовка данных
            inputs = [entry["original"] for entry in self.replacement_mappings]
            outputs = ["; ".join(entry["replacement"]) for entry in self.replacement_mappings]

            # Токенизация входов и выходов
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=self.vocab_size,
                oov_token="<OOV>"
            )
            all_texts = inputs + outputs
            self.tokenizer.fit_on_texts(all_texts)

            # Сохранение токенизатора
            with open(self.tokenizer_file, "w") as f:
                json.dump(self.tokenizer.word_index, f)

            input_sequences = self.tokenizer.texts_to_sequences(inputs)
            output_sequences = self.tokenizer.texts_to_sequences(outputs)

            input_padded = tf.keras.preprocessing.sequence.pad_sequences(
                input_sequences, maxlen=self.max_input_len, padding="post"
            )
            output_padded = tf.keras.preprocessing.sequence.pad_sequences(
                output_sequences, maxlen=self.max_output_len, padding="post"
            )

            # Построение модели трансформера
            inputs = tf.keras.Input(shape=(self.max_input_len,))
            x = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim
            )(inputs)
            transformer_block = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim
            )
            x = transformer_block(x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            outputs = tf.keras.layers.Dense(
                self.max_output_len * self.vocab_size,
                activation='softmax'
            )(x)
            outputs = tf.keras.layers.Reshape((self.max_output_len, self.vocab_size))(outputs)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            # Колбэк для обновления прогресса
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, analyzer):
                    self.analyzer = analyzer

                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.analyzer.epochs * 100
                    self.analyzer.queue.put(("training_progress", progress))
                    status = f"Epoch {epoch + 1}/{self.analyzer.epochs} - Loss: {logs['loss']:.4f}"
                    if 'val_loss' in logs:
                        status += f", Val Loss: {logs['val_loss']:.4f}"
                    self.analyzer.queue.put(("training_status", status))

            # Обучение модели
            self.model.fit(
                input_padded,
                output_padded,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[ProgressCallback(self)]
            )
            self.model.save(self.model_file)

            # Сохранение конфигурации с хешами
            current_hashes = self.calculate_json_hashes()
            self.save_model_config(current_hashes)

            logging.info("Transformer neural network trained and saved")
            self.queue.put(("training_complete", "Model training completed successfully"))

        except Exception as e:
            logging.error(f"Failed to train neural network: {str(e)}")
            self.queue.put(("training_error", f"Failed to train neural network: {str(e)}"))
            self.model = None
            self.tokenizer = None

        finally:
            self.is_training = False
            self.queue.put(("close_training_dialog", None))

    def load_feedback_data(self):
        """Загрузка данных обратной связи для дообучения."""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, "r") as f:
                    feedback_data = json.load(f)
                    self.replacement_mappings.extend(feedback_data)
                    logging.info(f"Loaded {len(feedback_data)} feedback entries")
        except Exception as e:
            logging.error(f"Failed to load feedback data: {str(e)}")

    def save_feedback_data(self):
        """Сохранение данных обратной связи."""
        try:
            with open(self.feedback_file, "w") as f:
                json.dump(self.replacement_mappings, f, indent=4)
            logging.info("Feedback data saved")
        except Exception as e:
            logging.error(f"Failed to save feedback data: {str(e)}")

    def retrain_model(self):
        """Переобучение модели на основе новых данных."""
        # Удаление старой модели
        if os.path.exists(self.model_file):
            os.remove(self.model_file)
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        if os.path.exists(self.tokenizer_file):
            os.remove(self.tokenizer_file)

        # Обучение новой модели
        messagebox.showinfo("Info", "Model retraining started in background")
        threading.Thread(target=self.train_model, daemon=True).start()

    def assemble_instruction(self, code_str):
        """Ассемблирование текстовой инструкции в машинный код."""
        if not HAS_KEYSTONE:
            logging.error("Keystone engine not installed. Cannot assemble instructions.")
            return None

        try:
            ks = Ks(KS_ARCH_X86, KS_MODE_64)
            assembled_bytes = b""

            # Разделяем инструкции по точке с запятой
            instructions = code_str.split(';')
            for instr in instructions:
                instr = instr.strip()
                if not instr:
                    continue

                # Ассемблирование каждой инструкции
                encoding, count = ks.asm(instr)
                if count > 0:
                    assembled_bytes += bytes(encoding)

            return assembled_bytes
        except Exception as e:
            logging.error(f"Assembly error: {str(e)}")
            return None

    def predict_replacement(self, instruction, context=""):
        """Предсказание замены с помощью нейросети с кэшированием."""
        # Если модель не загружена, возвращаем None
        if self.model is None or self.tokenizer is None:
            logging.warning("Model or tokenizer not available for prediction")
            return None, "Model not loaded"

        # Создаем ключ кэша с учетом контекста
        cache_key = hashlib.md5((instruction + context).encode()).hexdigest()

        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        try:
            # Объединяем инструкцию с контекстом
            full_input = f"{context} | {instruction}" if context else instruction

            sequence = self.tokenizer.texts_to_sequences([full_input])
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                sequence, maxlen=self.max_input_len, padding="post"
            )

            # Предсказание
            prediction = self.model.predict(padded, verbose=0)
            predicted_sequence = tf.argmax(prediction, axis=-1).numpy()[0]

            # Конвертация в текст
            replacement = []
            for idx in predicted_sequence:
                if idx == 0:  # Паддинг
                    break
                word = self.tokenizer.index_word.get(idx, "<OOV>")
                replacement.append(word)

            # Разделение на инструкции
            replacement_str = " ".join(replacement)
            replacements = replacement_str.split(";")
            replacements = [r.strip() for r in replacements if r.strip()]

            # Кэшируем результат
            self.prediction_cache[cache_key] = (replacements, "Transformer prediction (TensorFlow)")
            return replacements, "Transformer prediction (TensorFlow)"
        except Exception as e:
            logging.error(f"Prediction error for {instruction}: {str(e)}")
            return None, f"Prediction failed: {str(e)}"

    def normalize_instruction(self, instruction):
        """Извлечение мнемоники и нормализация операндов."""
        parts = instruction.strip().split(maxsplit=1)
        mnemonic = parts[0].lower()
        operands = parts[1].lower() if len(parts) > 1 else ""
        operand_pattern = re.sub(r'\b(ymm\d+|xmm\d+|r\d+[a-z]?|[er]?[a-z]{2,3}|imm\d+|m\d+|qword|dword|\[.*?\])\b', '|',
                                 operands)
        return mnemonic, operand_pattern, operands

    def match_instruction(self, mnemonic, op_str, inst_set, context=""):
        """Сопоставление инструкции с JSON-заменами или нейросетью."""
        src_mnemonic = mnemonic.lower()
        _, src_op_pattern, src_operands = self.normalize_instruction(f"{mnemonic} {op_str}")

        for entry in self.replacement_mappings:
            entry_mnemonic, entry_op_pattern, _ = self.normalize_instruction(entry["original"])
            if src_mnemonic == entry_mnemonic and src_op_pattern == entry_op_pattern:
                src_parts = re.split(r'[\s,]+', src_operands)
                entry_parts = re.split(r'[\s,]+', entry["original"].split(maxsplit=1)[1])
                operand_map = {}
                for s, e in zip(src_parts, entry_parts):
                    if s and e and s != e:
                        operand_map[e] = s

                replacements = []
                for repl in entry["replacement"]:
                    new_repl = repl
                    for old_op, new_op in operand_map.items():
                        new_repl = re.sub(r'\b' + re.escape(old_op) + r'\b', new_op, new_repl)
                    replacements.append(new_repl)
                return replacements, f"JSON mapping for {inst_set}"

        # Для AVX2 используем нейросеть с контекстом
        if inst_set in ["AVX2", "F16C", "FMA", "BMI1", "BMI2"]:
            return self.predict_replacement(f"{mnemonic} {op_str}", context)
        return None, "No mapping found"

    def validate_replacement(self, original_bytes, replacement_bytes):
        """Проверка замены через эмуляцию."""
        if replacement_bytes is None:
            return False

        try:
            # Инициализация эмулятора
            mu = Uc(UC_ARCH_X86, UC_MODE_64)

            # Выделение памяти
            ADDRESS = 0x1000000
            SIZE = 1024 * 1024
            mu.mem_map(ADDRESS, SIZE)

            # Запись оригинального кода
            mu.mem_write(ADDRESS, original_bytes)

            # Эмуляция оригинального кода
            mu.emu_start(ADDRESS, ADDRESS + len(original_bytes))
            orig_result = {
                UC_X86_REG_RAX: mu.reg_read(UC_X86_REG_RAX),
                UC_X86_REG_RBX: mu.reg_read(UC_X86_REG_RBX),
                UC_X86_REG_RCX: mu.reg_read(UC_X86_REG_RCX),
                UC_X86_REG_RDX: mu.reg_read(UC_X86_REG_RDX)
            }

            # Сброс состояния
            mu = Uc(UC_ARCH_X86, UC_MODE_64)
            mu.mem_map(ADDRESS, SIZE)

            # Запись замененного кода
            mu.mem_write(ADDRESS, replacement_bytes)

            # Эмуляция замененного кода
            mu.emu_start(ADDRESS, ADDRESS + len(replacement_bytes))
            repl_result = {
                UC_X86_REG_RAX: mu.reg_read(UC_X86_REG_RAX),
                UC_X86_REG_RBX: mu.reg_read(UC_X86_REG_RBX),
                UC_X86_REG_RCX: mu.reg_read(UC_X86_REG_RCX),
                UC_X86_REG_RDX: mu.reg_read(UC_X86_REG_RDX)
            }

            # Сравнение результатов
            return orig_result == repl_result
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return False

    def benchmark_code(self, code_bytes, iterations=100):
        """Замер производительности кода."""
        try:
            # Инициализация эмулятора
            mu = Uc(UC_ARCH_X86, UC_MODE_64)

            # Выделение памяти
            ADDRESS = 0x1000000
            SIZE = 1024 * 1024
            mu.mem_map(ADDRESS, SIZE)
            mu.mem_write(ADDRESS, code_bytes)

            # Замер времени выполнения
            start_time = time.time()
            for _ in range(iterations):
                mu.emu_start(ADDRESS, ADDRESS + len(code_bytes))
                mu.reg_write(UC_X86_REG_RIP, ADDRESS)  # Сброс указателя инструкций
            end_time = time.time()

            return (end_time - start_time) / iterations
        except Exception as e:
            logging.error(f"Benchmark failed: {str(e)}")
            return float('inf')

    def start_ai_replace(self):
        """Запуск замены инструкций в отдельном потоке"""
        if not self.model or not self.tokenizer:
            messagebox.showerror("Error", "Neural network model is not loaded. Please wait or retrain.")
            return

        if not self.original_tree.get_children():
            messagebox.showwarning("Warning", "No instructions to replace. Analyze a file first.")
            return

        # Отключаем кнопку на время выполнения
        self.ai_replace_button.config(state="disabled")
        self.progress["value"] = 0

        # Запускаем в отдельном потоке
        threading.Thread(target=self.ai_replace_instructions, daemon=True).start()

    def ai_replace_instructions(self):
        """Замена инструкций с проверкой и бенчмарком"""
        logging.info("Starting AI replacement with validation")
        feedback_data = []
        self.instruction_replacements = {}
        valid_replacements_count = 0

        # Очищаем предыдущие результаты замены
        self.queue.put(("clear_replaced_tree", None))

        # Переключаемся на вкладку с результатами заменя
        self.queue.put(("select_replaced_tab", None))

        total_instructions = len(self.original_tree.get_children())
        processed = 0

        for item in self.original_tree.get_children():
            if self.stop_event.is_set():
                break

            values = self.original_tree.item(item, "values")
            mnemonic, op_str, size, bytes_str, address = values

            # Получаем контекст для текущей инструкции
            context = self.context_cache.get(item, "")

            inst_set = self.get_instruction_by_mnemonic(mnemonic)

            if inst_set in ["AVX2", "F16C", "FMA", "BMI1", "BMI2"]:
                replacements, comment = self.match_instruction(mnemonic, op_str, inst_set, context)
                if replacements:
                    replacement_str = "; ".join(replacements)

                    # Ассемблируем замену
                    replacement_bytes = self.assemble_instruction(replacement_str)

                    if replacement_bytes is None:
                        validated = False
                        speedup = "N/A"
                        logging.error(f"Failed to assemble: {replacement_str}")
                    else:
                        # Проверка замены через эмуляцию
                        original_bytes_val = bytes.fromhex(bytes_str.replace(" ", ""))
                        validated = self.validate_replacement(original_bytes_val, replacement_bytes)

                        # Замер производительности
                        orig_time = self.benchmark_code(original_bytes_val)
                        repl_time = self.benchmark_code(replacement_bytes)
                        speedup = f"{(1 - repl_time / orig_time) * 100:.1f}%" if orig_time > 0 and repl_time != float(
                            'inf') else "N/A"

                    # Сохраняем замену для последующего патчинга
                    if validated:
                        self.instruction_replacements[address] = {
                            "original": bytes_str,
                            "replacement": replacement_bytes.hex(),
                            "size": size,
                            "new_size": len(replacement_bytes)
                        }
                        valid_replacements_count += 1

                    # Отправляем результат в главный поток
                    self.queue.put(("insert_replaced_tree", (
                        mnemonic, op_str, size, bytes_str,
                        replacement_str,
                        "Yes" if validated else "No",
                        speedup,
                        address
                    )))

                    # Сохранение для обратной связи
                    if validated:
                        feedback_data.append({
                            "original": f"{mnemonic} {op_str}",
                            "replacement": replacements,
                            "context": context
                        })

                    logging.info(f"Replaced {inst_set} {mnemonic} {op_str} with {replacement_str} ({comment})")
                else:
                    self.queue.put(("insert_replaced_tree", (
                        mnemonic, op_str, size, bytes_str, f"No AVX replacement ({comment})", "No", "N/A", address
                    )))
                    logging.warning(f"No replacement for {inst_set} {mnemonic} {op_str}")
            else:
                self.queue.put(("insert_replaced_tree", (
                    mnemonic, op_str, size, bytes_str, "N/A", "N/A", "N/A", address
                )))
                logging.debug(f"Skipping {inst_set or 'unknown'} instruction: {mnemonic} {op_str}")

            processed += 1
            progress = processed / total_instructions * 100
            self.queue.put(("progress", progress))

        # Сохранение обратной связи
        if feedback_data:
            self.replacement_mappings.extend(feedback_data)
            self.save_feedback_data()
            logging.info(f"Saved {len(feedback_data)} new feedback entries")

        logging.info("AI replacement completed")

        # Активируем кнопки экспорта и бенчмарка
        self.queue.put(("ai_replace_done", valid_replacements_count))

    def run_benchmark(self):
        """Запуск комплексного тестирования производительности."""
        if not self.replaced_tree.get_children():
            messagebox.showwarning("Warning", "No instructions to benchmark. Perform replacement first.")
            return

        # Сбор всех инструкций
        original_code = []
        replaced_code = []

        for item in self.original_tree.get_children():
            values = self.original_tree.item(item, "values")
            original_code.append(f"{values[0]} {values[1]}")

        for item in self.replaced_tree.get_children():
            values = self.replaced_tree.item(item, "values")
            if values[4] != "N/A":
                replaced_code.append(values[4])

        # Конвертация в байты
        original_bytes = self.assemble_instruction("; ".join(original_code))
        replaced_bytes = self.assemble_instruction("; ".join(replaced_code))

        if original_bytes is None or replaced_bytes is None:
            messagebox.showerror("Error", "Failed to assemble instructions for benchmarking")
            return

        # Сохранение для экспорта
        self.original_bytes = original_bytes
        self.modified_bytes = replaced_bytes

        # Замер производительности
        orig_time = self.benchmark_code(original_bytes, 1000)
        repl_time = self.benchmark_code(replaced_bytes, 1000)

        # Расчет улучшения
        improvement = (1 - repl_time / orig_time) * 100 if orig_time > 0 else 0

        # Сохранение результатов
        self.performance_data = {
            "original_time": orig_time,
            "replaced_time": repl_time,
            "improvement": improvement
        }

        # Отображение результатов
        messagebox.showinfo("Benchmark Results",
                            f"Original: {orig_time * 1000:.4f} ms\n"
                            f"Replaced: {repl_time * 1000:.4f} ms\n"
                            f"Improvement: {improvement:.2f}%")

    def export_binary(self):
        """Экспорт модифицированного исполняемого файла с использованием lief"""
        # Определяем константы характеристик секций
        MEM_READ = 0x40000000
        MEM_WRITE = 0x80000000
        MEM_EXECUTE = 0x20000000
        CNT_CODE = 0x00000020
        CNT_INITIALIZED_DATA = 0x00000040

        if not self.selected_file or not self.instruction_replacements:
            messagebox.showwarning("Warning", "No modified binary to export. Perform replacement first.")
            return

        try:
            # Проверяем, что файл существует
            if not os.path.isfile(self.selected_file):
                raise Exception(f"File not found: {self.selected_file}")

            # Загружаем бинарник с помощью lief
            try:
                binary = lief.parse(self.selected_file)
            except Exception as e:
                raise Exception(f"LIEF parsing failed: {str(e)}")

            if binary is None:
                raise Exception("Failed to parse the binary file")

            # Получаем секцию .text
            text_section = None
            for section in binary.sections:
                if section.name == ".text":
                    text_section = section
                    break

            if text_section is None:
                # Если не нашли .text, попробуем найти исполняемую секцию
                for section in binary.sections:
                    if section.characteristics & MEM_EXECUTE:
                        text_section = section
                        logging.warning(f"Using executable section instead of .text: {section.name}")
                        break

                if text_section is None:
                    raise Exception("No executable code section found")

            # Создаем новую секцию для длинных замен
            new_section = lief.PE.Section(".newsec")

            # Устанавливаем характеристики секции с помощью числовых значений
            new_section.characteristics = MEM_READ | MEM_EXECUTE | CNT_CODE

            # Выравнивание секций
            section_align = binary.optional_header.section_alignment
            file_align = binary.optional_header.file_alignment

            # Рассчитываем адреса для новой секции
            last_section = binary.sections[-1]
            new_section.virtual_address = last_section.virtual_address + last_section.virtual_size
            new_section.virtual_address = (
                                                  new_section.virtual_address + section_align - 1) // section_align * section_align
            new_section.pointerto_raw_data = last_section.pointerto_raw_data + last_section.sizeof_raw_data
            new_section.pointerto_raw_data = (
                                                     new_section.pointerto_raw_data + file_align - 1) // file_align * file_align

            # Инициализируем данные новой секции
            new_section_data = bytearray()
            patched_count = 0
            skipped_count = 0

            # Получаем содержимое .text секции как байтовый массив
            try:
                text_content = bytearray(text_section.content)
            except Exception as e:
                raise Exception(f"Failed to get .text content: {str(e)}")

            text_virtual_address = text_section.virtual_address
            image_base = binary.optional_header.imagebase

            for address, replacement in self.instruction_replacements.items():
                try:
                    # Преобразуем адрес в RVA
                    rva = int(address, 16) - image_base

                    # Получаем смещение в .text секции
                    offset_in_section = rva - text_virtual_address

                    # Проверяем границы
                    if offset_in_section < 0 or offset_in_section + int(replacement["size"]) > len(text_content):
                        logging.warning(f"Address {address} is outside .text section boundaries")
                        skipped_count += 1
                        continue

                    # Получаем заменяемые байты
                    new_bytes = bytes.fromhex(replacement["replacement"])
                    orig_size = int(replacement["size"])
                    new_size = replacement["new_size"]

                    # Если новая инструкция короче или равна оригиналу
                    if new_size <= orig_size:
                        # Заменяем байты в .text секции
                        text_content[offset_in_section:offset_in_section + new_size] = new_bytes
                        # Заполняем остаток NOP
                        for i in range(new_size, orig_size):
                            text_content[offset_in_section + i] = 0x90
                        patched_count += 1

                    # Если новая инструкция длиннее
                    else:
                        # Рассчитываем адрес для перехода в новую секцию
                        jmp_target = new_section.virtual_address + len(new_section_data)
                        # Смещение = цель - (текущий адрес + 5)
                        jmp_offset = jmp_target - (rva + 5)
                        jmp_bytes = struct.pack("<i", jmp_offset)

                        # Заменяем первые 5 байт на JMP
                        text_content[offset_in_section] = 0xE9
                        text_content[offset_in_section + 1:offset_in_section + 5] = jmp_bytes

                        # Заполняем остаток NOP
                        for i in range(5, orig_size):
                            text_content[offset_in_section + i] = 0x90

                        # Добавляем замененные инструкции в новую секцию
                        new_section_data += new_bytes

                        # Добавляем обратный переход
                        return_target = rva + orig_size
                        return_jmp_address = new_section.virtual_address + len(new_section_data) + 5
                        return_offset = return_target - return_jmp_address
                        return_jmp = b"\xE9" + struct.pack("<i", return_offset)
                        new_section_data += return_jmp
                        patched_count += 1

                except Exception as e:
                    logging.error(f"Patching failed at {address}: {str(e)}")
                    skipped_count += 1

            # Обновляем содержимое .text секции
            text_section.content = list(text_content)

            # Завершаем создание новой секции
            if new_section_data:
                # Устанавливаем размеры секции
                new_section.virtual_size = len(new_section_data)
                new_section.sizeof_raw_data = (len(new_section_data) + file_align - 1) // file_align * file_align

                # Заполняем данные секции
                padding = b'\x00' * (new_section.sizeof_raw_data - len(new_section_data))
                new_section.content = list(new_section_data + padding)

                # Добавляем новую секцию в бинарник
                binary.add_section(new_section)

                # Обновляем SizeOfImage
                binary.optional_header.sizeof_image = new_section.virtual_address + new_section.virtual_size
                binary.optional_header.sizeof_image = (
                                                              binary.optional_header.sizeof_image + section_align - 1) // section_align * section_align

            # Сохранение файла
            new_filename = filedialog.asksaveasfilename(
                defaultextension=".exe",
                filetypes=[("Executable files", "*.exe"), ("All files", "*.*")],
                title="Save Modified Executable"
            )

            if not new_filename:
                return  # Пользователь отменил сохранение

            # Создаем директорию, если нужно
            os.makedirs(os.path.dirname(new_filename), exist_ok=True)

            # Пишем бинарник
            try:
                builder = lief.PE.Builder(binary)
                builder.build()
                builder.write(new_filename)
            except Exception as e:
                raise Exception(f"Failed to write binary: {str(e)}")

            # Обновляем контрольную сумму
            try:
                # Переоткрываем файл для обновления контрольной суммы
                updated_binary = lief.parse(new_filename)
                if updated_binary:
                    updated_binary.optional_header.compute_checksum()
                    builder = lief.PE.Builder(updated_binary)
                    builder.build()
                    builder.write(new_filename)
            except Exception as e:
                logging.warning(f"Failed to update checksum: {str(e)}")

            message = f"Modified executable saved with {patched_count} patches"
            if skipped_count > 0:
                message += f" (skipped {skipped_count})"
            message += f":\n{new_filename}"
            messagebox.showinfo("Success", message)
            logging.info(f"Exported modified binary: {new_filename}")

        except Exception as e:
            logging.error(f"Export failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to export binary: {str(e)}")

    def get_instruction_by_mnemonic(self, mnemonic):
        """Определение набора инструкций по мнемонике."""
        mnemonic = mnemonic.lower()
        for inst_set, mnemonics in self.mnemonic_patterns.items():
            if mnemonic in mnemonics:
                return inst_set
        return None

    def show_context_menu(self, event):
        """Показ контекстного меню при правом клике на замененных инструкциях"""
        if self.replaced_tree.get_children():
            self.context_menu.post(event.x_root, event.y_root)

    def save_to_file(self):
        """Сохранение данных из таблицы замен в файл"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "w") as f:
                # Заголовки
                headers = ["Mnemonic", "Operand", "Size", "Bytes", "Replacement", "Validated", "Speedup", "Address"]
                f.write("\t".join(headers) + "\n")

                # Данные
                for item in self.replaced_tree.get_children():
                    values = self.replaced_tree.item(item, "values")
                    f.write("\t".join(values) + "\n")

            messagebox.showinfo("Success", f"Data saved to:\n{file_path}")
            logging.info(f"Saved replacement data to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save file: {str(e)}")
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def open_file(self):
        """Открытие EXE-файла для анализа."""
        file_path = filedialog.askopenfilename(filetypes=[("Executable files", "*.exe")])
        if file_path:
            self.selected_file = file_path
            self.analyze_button.config(state="normal")
            self.original_tree.delete(*self.original_tree.get_children())
            self.found_instructions = {k: 0 for k in self.instruction_sets}
            self.update_status_bar()
            self.progress["value"] = 0
            self.context_cache = {}
            self.instruction_replacements = {}
            self.instruction_addresses = {}
            logging.info(f"Selected file: {file_path}")

    def start_analysis(self):
        """Начало анализа файла."""
        if self.is_analyzing:
            messagebox.showinfo("Info", "Analysis is already in progress.")
            return
        if not self.selected_file:
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        self.is_analyzing = True
        self.stop_event.clear()
        self.analyze_button.config(state="disabled")
        self.open_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.ai_replace_button.config(state="disabled")
        self.benchmark_button.config(state="disabled")
        self.export_button.config(state="disabled")
        self.original_tree.delete(*self.original_tree.get_children())
        self.found_instructions = {k: 0 for k in self.instruction_sets}
        self.update_status_bar()
        self.progress["value"] = 0
        self.context_cache = {}
        self.instruction_replacements = {}
        self.instruction_addresses = {}
        logging.info("Starting analysis")

        threading.Thread(target=self.analyze_file, args=(self.selected_file,), daemon=True).start()

    def stop_analysis(self):
        """Остановка анализа."""
        if self.is_analyzing:
            self.stop_event.set()
            self.queue.put(("stop", "Analysis stopped by user"))
            logging.info("Analysis stopped by user")

    def analyze_file(self, file_path):
        """Анализ EXE-файла с извлечением секции кода (без CFG)"""
        try:
            pe = pefile.PE(file_path)
            code_section = None
            for section in pe.sections:
                section_name = section.Name.decode().strip('\x00')
                if section_name == '.text' or section.Characteristics & 0x20000020:
                    code_section = section
                    break

            if not code_section:
                self.queue.put(("error", "No executable code section found in the file"))
                return

            code_data = code_section.get_data()
            base_address = code_section.VirtualAddress + pe.OPTIONAL_HEADER.ImageBase
            size = code_section.SizeOfRawData

            total_size = size
            processed_bytes = 0
            chunk_size = 4096
            row_count = 0

            md = Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
            md.detail = True

            while processed_bytes < total_size:
                if self.stop_event.is_set():
                    break

                chunk_end = min(processed_bytes + chunk_size, total_size)
                chunk = code_data[processed_bytes:chunk_end]

                results = self.analyze_chunk(chunk, base_address + processed_bytes, row_count)
                row_count += len(results)

                for result in results:
                    self.queue.put(("instruction", result))

                processed_bytes += len(chunk)
                progress = min(100, max(1, (processed_bytes / total_size) * 100))
                self.queue.put(("progress", progress))

            if not self.stop_event.is_set():
                self.queue.put(("complete", None))
                logging.info("Analysis completed")
        except Exception as e:
            self.queue.put(("error", str(e)))
            logging.error(f"Analysis error: {str(e)}")

    def analyze_chunk(self, chunk, base_address, row_count):
        """Анализ чанка данных (без CFG)"""
        results = []
        md = Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
        md.detail = True

        context_window = []
        context_size = 3

        for inst in md.disasm(chunk, base_address):
            if self.stop_event.is_set():
                break

            full_inst = f"{inst.mnemonic} {inst.op_str}"
            context_window.append(full_inst)
            if len(context_window) > context_size * 2 + 1:
                context_window.pop(0)

            instruction_set = self.get_instruction_set(inst)
            if instruction_set:
                self.found_instructions[instruction_set] += 1

                selected_set = self.filter_combobox.get()
                if instruction_set == selected_set:
                    context_id = f"{inst.address}_{inst.mnemonic}"
                    self.context_cache[context_id] = " | ".join(context_window)
                    self.instruction_addresses[context_id] = inst.address

                    results.append((
                        inst.mnemonic,
                        inst.op_str,
                        inst.size,
                        ' '.join(f"{b:02x}" for b in inst.bytes),
                        hex(inst.address),
                        "oddrow" if row_count % 2 else "evenrow",
                        context_id
                    ))

        return results

    def get_instruction_set(self, instruction):
        """Определение набора инструкций."""
        # Сначала проверяем группы (более надежно)
        for inst_set, groups in self.group_mapping.items():
            if any(group in groups for group in instruction.groups):
                return inst_set

        # Если не нашли по группам, проверяем по мнемоникам
        mnemonic = instruction.mnemonic.lower()
        for inst_set, mnemonics in self.mnemonic_patterns.items():
            if mnemonic in mnemonics:
                return inst_set
        return None

    def check_queue(self):
        """Обработка сообщений из очереди"""
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                logging.debug(f"Queue message: {msg_type}")

                if msg_type == "model_loaded":
                    messagebox.showinfo("Model Loaded", data)

                elif msg_type == "model_error":
                    messagebox.showwarning("Model Error", data)

                elif msg_type == "training_error":
                    messagebox.showerror("Training Error", data)

                elif msg_type == "show_training_dialog":
                    self.show_training_dialog()

                elif msg_type == "training_progress":
                    if self.training_progressbar:
                        self.training_progressbar['value'] = data

                elif msg_type == "training_status":
                    if self.training_status:
                        self.training_status.config(text=data)

                elif msg_type == "training_complete":
                    messagebox.showinfo("Training Complete", data)

                elif msg_type == "close_training_dialog":
                    if self.training_dialog:
                        self.training_dialog.destroy()

                if msg_type == "instruction":
                    mnemonic, op_str, size, bytes_str, address, tag, context_id = data
                    self.original_tree.insert("", "end", values=(
                        mnemonic, op_str, size, bytes_str, address
                    ), tags=(tag,), iid=context_id)
                elif msg_type == "progress":
                    self.progress["value"] = data
                    self.root.update()
                    logging.debug(f"Progress updated: {data:.2f}%")
                elif msg_type == "complete":
                    self.is_analyzing = False
                    self.analyze_button.config(state="normal" if self.selected_file else "disabled")
                    self.open_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.ai_replace_button.config(state="normal" if self.original_tree.get_children() else "disabled")
                    self.update_status_bar()
                    self.progress["value"] = 100
                    logging.info("Analysis complete, progress set to 100%")
                elif msg_type == "stop":
                    self.is_analyzing = False
                    self.analyze_button.config(state="normal" if self.selected_file else "disabled")
                    self.open_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.ai_replace_button.config(state="disabled")
                    self.update_status_bar()
                    messagebox.showinfo("Info", data)
                elif msg_type == "error":
                    self.is_analyzing = False
                    self.analyze_button.config(state="normal" if self.selected_file else "disabled")
                    self.open_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.ai_replace_button.config(state="disabled")
                    self.update_status_bar()
                    messagebox.showerror("Error", data)
                elif msg_type == "retrain_prompt":
                    if self.prompt_for_retraining():
                        self.retrain_model()
                    else:
                        try:
                            self.load_model()
                        except:
                            logging.warning("Using existing model without retraining")

                elif msg_type == "clear_replaced_tree":
                    self.replaced_tree.delete(*self.replaced_tree.get_children())
                elif msg_type == "select_replaced_tab":
                    self.notebook.select(self.replaced_tab)
                elif msg_type == "insert_replaced_tree":
                    self.replaced_tree.insert("", "end", values=data)
                elif msg_type == "ai_replace_done":
                    self.ai_replace_button.config(state="normal")
                    valid_count = data
                    if valid_count > 0:
                        self.export_button.config(state="normal")
                        self.benchmark_button.config(state="normal")
                        messagebox.showinfo("Info", f"AI replacement completed with {valid_count} valid replacements")
                    else:
                        messagebox.showinfo("Info", "AI replacement completed but no valid replacements found")
        except queue.Empty:
            pass

        # Продолжаем проверять очередь
        self.root.after(100, self.check_queue)

    def update_status_bar(self):
        """Обновление статус-бара."""
        for inst_set, count in self.found_instructions.items():
            self.status_labels[inst_set].config(text=f"{inst_set}: {count}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedInstructionAnalyzer(root)
    root.mainloop()
