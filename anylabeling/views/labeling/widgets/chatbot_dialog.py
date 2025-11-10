import base64
import datetime
import json
import os
import os.path as osp
import re
import shutil
import threading
import zipfile
from openai import OpenAI
from PIL import Image

from PyQt5.QtCore import QTimer, Qt, QSize, QPoint, QEvent
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
    QApplication,
    QScrollArea,
    QFrame,
    QSplitter,
    QTabWidget,
    QSlider,
    QSpinBox,
    QMessageBox,
    QProgressDialog,
    QFileDialog,
)
from PyQt5.QtGui import (
    QCursor,
    QIcon,
    QPixmap,
    QColor,
    QTextCursor,
    QTextCharFormat,
)

from anylabeling.app_info import __version__
from anylabeling.views.labeling.chatbot import *
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.general import open_url
from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path
from anylabeling.views.labeling.widgets.model_dropdown_widget import (
    ModelDropdown,
)


class ChatbotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(*DEFAULT_WINDOW_SIZE)

        dialog_style = ChatbotDialogStyle.get_dialog_style()
        menu_style = ChatbotDialogStyle.get_menu_style()
        combined_style = dialog_style + menu_style
        self.setStyleSheet(combined_style)

        # Initialize
        _model_settings = init_model_config()
        self.chat_history = []
        self.current_api_key = None
        self.current_api_address = None
        self.default_provider = _model_settings["provider"]
        self.providers = get_providers_data()

        # Create all tooltips first to ensure they exist before any event filtering
        self.temperature_tooltip = CustomTooltip(
            title="Recommended values:",
            value_pairs=[
                ("Coding / Math", "0"),
                ("Data Cleaning / Data Analysis", "1"),
                ("General Conversation", "1.3"),
                ("Translation", "1.3"),
                ("Creative Writing / Poetry", "1.5"),
            ],
        )
        self.clear_chat_tooltip = CustomTooltip(title=self.tr("Clear Chat"))
        self.open_image_folder_tooltip = CustomTooltip(
            title=self.tr("Open Image Folder")
        )
        self.open_image_file_tooltip = CustomTooltip(
            title=self.tr("Open Image File")
        )
        self.prev_image_tooltip = CustomTooltip(
            title=self.tr("Previous Image")
        )
        self.next_image_tooltip = CustomTooltip(title=self.tr("Next Image"))
        self.run_all_images_tooltip = CustomTooltip(
            title=self.tr("Run All Images")
        )
        self.import_export_tooltip = CustomTooltip(
            title=self.tr("Import/Export Dataset")
        )

        pixmap = QPixmap(new_icon_path("click", "svg"))
        scaled_pixmap = pixmap.scaled(
            *ICON_SIZE_SMALL, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.click_cursor = QCursor(scaled_pixmap)

        # Streaming handler setup
        self.stream_handler = StreamingHandler()
        self.stream_handler.text_update.connect(self.update_output)
        self.stream_handler.finished.connect(self.on_stream_finished)
        self.stream_handler.loading.connect(self.handle_loading_state)
        self.stream_handler.error_occurred.connect(self.handle_error)

        # Main horizontal layout with splitters
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create main splitter for three columns
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(1)
        self.main_splitter.setStyleSheet(
            ChatbotDialogStyle.get_main_splitter_style()
        )

        ################################
        # Left panel - Model Providers #
        ################################
        self.left_widget = QWidget()
        left_panel = QVBoxLayout(self.left_widget)
        left_panel.setContentsMargins(20, 24, 20, 24)
        left_panel.setSpacing(12)

        # Create button group for providers
        provider_group = QButtonGroup(self)

        # Set provider buttons
        for provider in self.providers.keys():
            btn = QPushButton(self.tr(provider.capitalize()))
            btn.setIcon(QIcon(new_icon(provider.lower())))
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            btn.setIconSize(QSize(*ICON_SIZE_SMALL))
            btn.setStyleSheet(ChatbotDialogStyle.get_provider_button_style())
            btn.clicked.connect(
                lambda checked, p=provider: (
                    self.switch_provider(p) if checked else None
                )
            )
            provider_group.addButton(btn)
            setattr(self, f"{provider}_btn", btn)
            left_panel.addWidget(btn)

        # Set default fields
        getattr(self, f"{self.default_provider}_btn").setChecked(True)

        # Add stretch to push everything to the top
        left_panel.addStretch()

        # Styling for the left panel
        self.left_widget.setStyleSheet(
            ChatbotDialogStyle.get_left_widget_style()
        )
        self.left_widget.setMinimumWidth(200)
        self.left_widget.setMaximumWidth(250)

        #################################
        # Middle panel - Chat interface #
        #################################
        self.middle_widget = QWidget()
        self.middle_widget.setStyleSheet(
            ChatbotDialogStyle.get_middle_widget_style()
        )
        middle_panel = QVBoxLayout(self.middle_widget)
        middle_panel.setContentsMargins(0, 0, 0, 0)
        middle_panel.setSpacing(0)

        # Chat area
        chat_container = QWidget()
        chat_container.setStyleSheet(
            ChatbotDialogStyle.get_chat_container_style()
        )
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 20, 0, 20)
        chat_layout.setSpacing(16)

        # Scroll area for chat messages
        self.chat_scroll_area = QScrollArea()
        self.chat_scroll_area.setWidgetResizable(True)
        self.chat_scroll_area.setFrameShape(QFrame.NoFrame)
        self.chat_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.chat_scroll_area.setStyleSheet(
            ChatbotDialogStyle.get_chat_scroll_area_style()
        )

        # Widget to contain all chat messages
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet(
            ChatbotDialogStyle.get_chat_container_style()
        )
        self.chat_messages_layout = QVBoxLayout(self.chat_container)
        self.chat_messages_layout.setContentsMargins(24, 12, 24, 12)
        self.chat_messages_layout.setSpacing(16)
        self.chat_messages_layout.addStretch()

        self.chat_scroll_area.setWidget(self.chat_container)
        chat_layout.addWidget(self.chat_scroll_area)

        # Input area with simplified design
        input_container = QWidget()
        input_container.setStyleSheet(
            ChatbotDialogStyle.get_input_container_style()
        )
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(24, 12, 24, 12)
        input_layout.setSpacing(0)

        # Create a container for the input field with embedded send button
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_frame.setStyleSheet(ChatbotDialogStyle.get_input_frame_style())

        # Use a relative layout for the input frame
        input_frame_layout = QVBoxLayout(input_frame)
        input_frame_layout.setContentsMargins(12, 8, 12, 8)
        input_frame_layout.setSpacing(0)

        # Create the message input
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText(
            self.tr(
                "Type something and Ctrl+↩︎ to send. Use @image to add an image."
            )
        )
        self.message_input.setStyleSheet(
            ChatbotDialogStyle.get_message_input_style()
        )
        self.message_input.setAcceptRichText(False)
        self.message_input.setMinimumHeight(MIN_MSG_INPUT_HEIGHT)
        self.message_input.setMaximumHeight(MAX_MSG_INPUT_HEIGHT)
        self.message_input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.message_input.setFrameShape(QFrame.NoFrame)
        self.message_input.setFrameShadow(QFrame.Plain)
        self.message_input.setLineWrapMode(QTextEdit.WidgetWidth)
        self.message_input.setContextMenuPolicy(Qt.CustomContextMenu)
        self.message_input.customContextMenuRequested.connect(
            self.show_message_input_context_menu
        )

        # Completely remove any internal frame or border
        document = self.message_input.document()
        document.setDocumentMargin(0)
        self.message_input.setDocument(document)

        # Set text format to remove any block margins
        cursor = self.message_input.textCursor()
        format = cursor.blockFormat()
        format.setBottomMargin(0)
        format.setTopMargin(0)
        cursor.setBlockFormat(format)
        self.message_input.setTextCursor(cursor)

        # Connect textChanged to dynamically resize input and highlight @image tag
        self.message_input.textChanged.connect(self.on_text_changed)
        self.message_input.installEventFilter(self)  # For Enter key handling

        # Initialize the input size
        self.resize_input()

        # Create a container for the input and send button
        input_with_button = QWidget()
        input_with_button_layout = QVBoxLayout(input_with_button)
        input_with_button_layout.setContentsMargins(0, 0, 0, 0)
        input_with_button_layout.setSpacing(0)

        # Add the message input to the layout (at the top)
        input_with_button_layout.addWidget(self.message_input, 1)

        # Create a button bar container for the bottom
        button_bar = QWidget()
        button_bar_layout = QHBoxLayout(button_bar)
        button_bar_layout.setContentsMargins(
            0, 4, 0, 0
        )  # Add a small top margin
        button_bar_layout.setSpacing(8)  # Spacing between buttons

        # Add clear context button (left side)
        self.clear_chat_btn = QPushButton()
        self.clear_chat_btn.setIcon(QIcon(new_icon("eraser", "svg")))
        self.clear_chat_btn.setIconSize(QSize(*ICON_SIZE_SMALL))
        self.clear_chat_btn.setStyleSheet(
            ChatbotDialogStyle.get_send_button_style()
        )
        self.clear_chat_btn.setFixedSize(*ICON_SIZE_SMALL)
        self.clear_chat_btn.clicked.connect(self.clear_conversation)
        self.clear_chat_btn.installEventFilter(self)
        self.clear_chat_btn.setObjectName("clear_chat_btn")

        # Add buttons to layout
        button_bar_layout.addWidget(self.clear_chat_btn, 0, Qt.AlignBottom)
        button_bar_layout.addStretch(1)  # Push buttons to left and right edges

        # Create the send button (right side)
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon(new_icon("send", "svg")))
        self.send_btn.setIconSize(QSize(*ICON_SIZE_SMALL))
        self.send_btn.setStyleSheet(ChatbotDialogStyle.get_send_button_style())
        self.send_btn.setFixedSize(*ICON_SIZE_SMALL)
        self.send_btn.clicked.connect(self.start_generation)
        self.send_btn.setEnabled(False)

        # Add send button to layout
        button_bar_layout.addWidget(self.send_btn, 0, Qt.AlignBottom)
        input_with_button_layout.addWidget(button_bar, 0, Qt.AlignBottom)
        input_frame_layout.addWidget(input_with_button)
        input_layout.addWidget(input_frame)

        # Add the chat container and input container to the middle panel
        middle_panel.addWidget(chat_container, CHAT_PANEL_PERCENTAGE)
        middle_panel.addWidget(input_container, INPUT_PANEL_PERCENTAGE)

        ############################################
        # Right panel - Image preview and settings #
        ############################################
        self.right_widget = QWidget()
        right_panel = QVBoxLayout(self.right_widget)
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(0)

        # Image preview panel
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(24, 24, 24, 16)

        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setStyleSheet(
            ChatbotDialogStyle.get_image_preview_style()
        )
        self.image_preview.setMinimumHeight(200)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setScaledContents(False)
        image_layout.addWidget(self.image_preview)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_image_btn = QPushButton()
        self.prev_image_btn.setIcon(QIcon(new_icon("arrow-left", "svg")))
        self.prev_image_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.prev_image_btn.setStyleSheet(
            ChatbotDialogStyle.get_button_style()
        )
        self.prev_image_btn.clicked.connect(
            lambda: self.navigate_image(direction="prev")
        )
        self.prev_image_btn.installEventFilter(self)
        self.prev_image_btn.setObjectName("prev_image_btn")
        self.prev_image_btn.setVisible(False)

        self.next_image_btn = QPushButton()
        self.next_image_btn.setIcon(QIcon(new_icon("arrow-right", "svg")))
        self.next_image_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.next_image_btn.setStyleSheet(
            ChatbotDialogStyle.get_button_style()
        )
        self.next_image_btn.clicked.connect(
            lambda: self.navigate_image(direction="next")
        )
        self.next_image_btn.installEventFilter(self)
        self.next_image_btn.setObjectName("next_image_btn")
        self.next_image_btn.setVisible(False)

        nav_layout.addWidget(self.prev_image_btn)
        nav_layout.addStretch()

        # Add image buttons for importing media
        import_media_btn_modes = ["image", "folder"]
        import_media_btn_names = [
            "open_image_file_btn",
            "open_image_folder_btn",
        ]
        for btn_mode, btn_name in zip(
            import_media_btn_modes, import_media_btn_names
        ):
            btn = QPushButton()
            btn.setIcon(QIcon(new_icon(btn_mode, "svg")))
            btn.setFixedSize(*ICON_SIZE_NORMAL)
            btn.setStyleSheet(ChatbotDialogStyle.get_button_style())
            btn.clicked.connect(
                lambda checked=False, mode=btn_mode: self.open_image_file_or_folder(
                    mode=mode
                )
            )
            btn.installEventFilter(self)
            btn.setObjectName(btn_name)
            nav_layout.addWidget(btn)
            setattr(self, btn_name, btn)  # Store reference as class attribute

        self.run_all_images_btn = QPushButton()
        self.run_all_images_btn.setIcon(QIcon(new_icon("run", "svg")))
        self.run_all_images_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.run_all_images_btn.setStyleSheet(
            ChatbotDialogStyle.get_button_style()
        )
        self.run_all_images_btn.clicked.connect(self.run_all_images)
        self.run_all_images_btn.installEventFilter(self)
        self.run_all_images_btn.setObjectName("run_all_images_btn")
        self.run_all_images_btn.setVisible(False)
        nav_layout.addWidget(self.run_all_images_btn)

        self.import_export_btn = QPushButton()
        self.import_export_btn.setIcon(QIcon(new_icon("import-export", "svg")))
        self.import_export_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.import_export_btn.setStyleSheet(
            ChatbotDialogStyle.get_button_style()
        )
        self.import_export_btn.clicked.connect(self.import_export_dataset)
        self.import_export_btn.installEventFilter(self)
        self.import_export_btn.setObjectName("import_export_btn")
        self.import_export_btn.setVisible(False)
        nav_layout.addWidget(self.import_export_btn)

        nav_layout.addStretch()
        nav_layout.addWidget(self.next_image_btn)
        image_layout.addLayout(nav_layout)

        # Settings panel with tabs
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(0)

        # Create tab widget
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setStyleSheet(
            ChatbotDialogStyle.get_tab_widget_style()
        )
        self.settings_tabs.setUsesScrollButtons(False)
        self.settings_tabs.setDocumentMode(True)
        self.settings_tabs.setElideMode(Qt.ElideNone)

        # First tab - API Settings
        api_settings_tab = QWidget()
        api_settings_layout = QVBoxLayout(api_settings_tab)
        api_settings_layout.setContentsMargins(24, 24, 24, 24)
        api_settings_layout.setSpacing(12)

        # API Address with help icon
        api_address_container = QHBoxLayout()
        api_address_label = QLabel(self.tr("API Address"))
        api_address_label.setStyleSheet(
            ChatbotDialogStyle.get_settings_label_style()
        )

        # Create a container for label and help button
        label_with_help = QWidget()
        label_help_layout = QHBoxLayout(label_with_help)
        label_help_layout.setContentsMargins(0, 0, 0, 0)

        api_docs_url = self.providers[self.default_provider]["api_docs_url"]
        api_help_btn = QPushButton()
        api_help_btn.setObjectName("api_help_btn")
        api_help_btn.setIcon(QIcon(new_icon("help-circle", "svg")))
        api_help_btn.setFixedSize(*ICON_SIZE_SMALL)
        api_help_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        api_help_btn.setCursor(self.click_cursor)
        api_help_btn.clicked.connect(lambda: open_url(api_docs_url))
        label_help_layout.addWidget(api_address_label)
        label_help_layout.addWidget(api_help_btn)
        label_help_layout.addStretch()
        if not api_docs_url:
            api_help_btn.setVisible(False)

        api_address_container.addWidget(label_with_help)
        api_address_container.addStretch()
        api_settings_layout.addLayout(api_address_container)

        self.api_address = QLineEdit(
            self.providers[self.default_provider]["api_address"]
        )
        self.api_address.setPlaceholderText(
            DEFAULT_PROVIDERS_DATA[self.default_provider].get(
                "api_address", ""
            )
        )
        self.api_address.setStyleSheet(
            ChatbotDialogStyle.get_settings_edit_style()
        )
        self.api_address.installEventFilter(self)
        self.api_address.textChanged.connect(self.on_api_address_changed)
        api_settings_layout.addWidget(self.api_address)
        self.current_api_address = self.api_address.text()

        # API Key with help icon
        api_key_container = QHBoxLayout()
        api_key_label = QLabel(self.tr("API Key"))
        api_key_label.setStyleSheet(
            ChatbotDialogStyle.get_settings_label_style()
        )

        # Create a container for label and help button
        key_label_with_help = QWidget()
        key_label_help_layout = QHBoxLayout(key_label_with_help)
        key_label_help_layout.setContentsMargins(0, 0, 0, 0)
        key_label_help_layout.addWidget(api_key_label)
        api_key_url = self.providers[self.default_provider]["api_key_url"]
        api_key_help_btn = QPushButton()
        api_key_help_btn.setObjectName("api_key_help_btn")
        api_key_help_btn.setIcon(QIcon(new_icon("help-circle", "svg")))
        api_key_help_btn.setFixedSize(*ICON_SIZE_SMALL)
        api_key_help_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        api_key_help_btn.setCursor(self.click_cursor)
        api_key_help_btn.clicked.connect(lambda: open_url(api_key_url))
        if not api_key_url:
            api_key_help_btn.setVisible(False)
        key_label_help_layout.addWidget(api_key_help_btn)
        key_label_help_layout.addStretch()

        api_key_container.addWidget(key_label_with_help)
        api_key_container.addStretch()
        api_settings_layout.addLayout(api_key_container)

        # API key input with toggle visibility
        api_key_container = QHBoxLayout()
        self.api_key = QLineEdit(
            self.providers[self.default_provider]["api_key"]
        )
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText(self.tr("Enter API key"))
        self.api_key.setStyleSheet(
            ChatbotDialogStyle.get_settings_edit_style()
        )
        self.api_key.installEventFilter(self)
        self.api_key.textChanged.connect(self.on_api_key_changed)
        self.current_api_key = self.api_key.text()

        self.toggle_visibility_btn = QPushButton()
        self.toggle_visibility_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.toggle_visibility_btn.setIcon(QIcon(new_icon("eye-off", "svg")))
        self.toggle_visibility_btn.setCheckable(True)
        self.toggle_visibility_btn.setStyleSheet(
            ChatbotDialogStyle.get_button_style()
        )
        self.toggle_visibility_btn.clicked.connect(
            self.toggle_api_key_visibility
        )

        api_key_container.addWidget(self.api_key)
        api_key_container.addWidget(self.toggle_visibility_btn)
        api_settings_layout.addLayout(api_key_container)

        # Model Name with help icon
        model_name_container = QHBoxLayout()
        model_name_label = QLabel(self.tr("Model Name"))
        model_name_label.setStyleSheet(
            ChatbotDialogStyle.get_settings_label_style()
        )

        # Create a container for label and buttons
        model_label_with_help = QWidget()
        model_label_help_layout = QHBoxLayout(model_label_with_help)
        model_label_help_layout.setContentsMargins(0, 0, 0, 0)
        model_label_help_layout.setSpacing(4)

        model_docs_url = self.providers[self.default_provider][
            "model_docs_url"
        ]
        model_help_btn = QPushButton()
        model_help_btn.setObjectName("model_help_btn")
        model_help_btn.setIcon(QIcon(new_icon("help-circle", "svg")))
        model_help_btn.setFixedSize(*ICON_SIZE_SMALL)
        model_help_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        model_help_btn.setCursor(self.click_cursor)
        model_help_btn.clicked.connect(lambda: open_url(model_docs_url))
        if not model_docs_url:
            model_help_btn.setVisible(False)

        model_label_help_layout.addWidget(model_name_label)
        model_label_help_layout.addWidget(model_help_btn)
        model_name_container.addWidget(model_label_with_help)
        model_name_container.addStretch()
        api_settings_layout.addLayout(model_name_container)

        # Create ComboBox for model selection
        self.model_button = QPushButton()
        self.model_button.setStyleSheet(
            ChatbotDialogStyle.get_model_button_style()
        )
        self.model_button.setMinimumHeight(40)
        self.model_button.setText(get_default_model_id(self.default_provider))
        api_settings_layout.addWidget(self.model_button)
        api_settings_layout.addStretch()

        # Second tab - Model Parameters
        model_params_tab = QWidget()
        model_params_layout = QVBoxLayout(model_params_tab)
        model_params_layout.setContentsMargins(24, 24, 24, 24)
        model_params_layout.setSpacing(16)

        # System prompt section
        system_prompt_label = QLabel(self.tr("System instruction"))
        system_prompt_label.setStyleSheet(
            ChatbotDialogStyle.get_settings_label_style()
        )
        model_params_layout.addWidget(system_prompt_label)

        # System prompt input
        system_prompt_container = QHBoxLayout()
        self.system_prompt_input = QLineEdit()
        self.system_prompt_input.setStyleSheet(
            ChatbotDialogStyle.get_settings_edit_style()
        )
        system_prompt_container.addWidget(self.system_prompt_input)
        model_params_layout.addLayout(system_prompt_container)

        # Temperature parameter with info icon
        temp_header = QHBoxLayout()
        temp_header.setSpacing(4)
        temp_label = QLabel(self.tr("Temperature"))
        temp_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())

        temp_info_btn = QPushButton()
        temp_info_btn.setIcon(QIcon(new_icon("help-circle", "svg")))
        temp_info_btn.setFixedSize(*ICON_SIZE_SMALL)
        temp_info_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        temp_info_btn.installEventFilter(self)
        temp_info_btn.setObjectName("temperature_btn")

        self.temp_value = QLabel(f"{_model_settings['temperature']/10:.1f}")
        self.temp_value.setStyleSheet(
            ChatbotDialogStyle.get_settings_label_style()
        )
        self.temp_value.setAlignment(Qt.AlignRight)

        temp_header.addWidget(temp_label)
        temp_header.addWidget(temp_info_btn)
        temp_header.addStretch()
        temp_header.addWidget(self.temp_value)
        model_params_layout.addLayout(temp_header)

        # Temperature slider
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(20)  # 0.0 to 2.0 with step of 0.1
        self.temp_slider.setValue(_model_settings["temperature"])
        self.temp_slider.setStyleSheet(ChatbotDialogStyle.get_slider_style())
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_value.setText(f"{v/10:.1f}")
        )
        model_params_layout.addWidget(self.temp_slider)

        # Temperature labels
        temp_labels_layout = QHBoxLayout()

        precise_label = QLabel(self.tr("Precise"))
        precise_label.setStyleSheet(
            ChatbotDialogStyle.get_temperature_label_style()
        )
        precise_label.setAlignment(Qt.AlignLeft)

        neutral_label = QLabel(self.tr("Neutral"))
        neutral_label.setStyleSheet(
            ChatbotDialogStyle.get_temperature_label_style()
        )
        neutral_label.setAlignment(Qt.AlignCenter)

        creative_label = QLabel(self.tr("Creative"))
        creative_label.setStyleSheet(
            ChatbotDialogStyle.get_temperature_label_style()
        )
        creative_label.setAlignment(Qt.AlignRight)

        temp_labels_layout.addWidget(precise_label)
        temp_labels_layout.addWidget(neutral_label)
        temp_labels_layout.addWidget(creative_label)

        model_params_layout.addLayout(temp_labels_layout)
        model_params_layout.addSpacing(16)

        # Maximum output length
        max_length_label = QLabel(self.tr("Max output tokens"))
        max_length_label.setStyleSheet(
            ChatbotDialogStyle.get_settings_label_style()
        )
        model_params_layout.addWidget(max_length_label)

        self.max_length_input = QSpinBox()
        self.max_length_input.setMinimum(0)
        self.max_length_input.setMaximum(9999999)
        self.max_length_input.setSingleStep(1)
        self.max_length_input.setButtonSymbols(QSpinBox.UpDownArrows)
        self.max_length_input.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        self.max_length_input.setFixedHeight(40)
        if _model_settings["max_length"]:
            self.max_length_input.setValue(_model_settings["max_length"])
        else:
            # Set a higher default value to prevent truncation
            self.max_length_input.setValue(4096)
        model_params_layout.addWidget(self.max_length_input)

        # Add stretch to push everything to the top
        model_params_layout.addStretch()

        # Add tabs to tab widget
        self.settings_tabs.addTab(api_settings_tab, self.tr("Backend"))
        self.settings_tabs.addTab(model_params_tab, self.tr("Generation"))
        self.settings_tabs.tabBar().setExpanding(True)
        self.settings_tabs.tabBar().setStyleSheet(
            ChatbotDialogStyle.get_settings_tabs_style()
        )

        # Add tab widget to settings layout
        settings_layout.addWidget(self.settings_tabs)

        # Create a splitter for the right panel to separate image and settings
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(1)
        right_splitter.setStyleSheet(
            ChatbotDialogStyle.get_right_splitter_style()
        )

        right_splitter.addWidget(image_panel)
        right_splitter.addWidget(settings_panel)

        # Set initial sizes for right splitter (40% image, 60% settings)
        right_splitter.setSizes([300, 400])
        right_panel.addWidget(right_splitter)

        # Styling for the right panel
        self.right_widget.setStyleSheet(
            ChatbotDialogStyle.get_right_widget_style()
        )
        self.right_widget.setFixedWidth(360)

        # Create wrapper for middle widget to handle centering when maximized
        self.middle_wrapper_widget = QWidget()
        self.middle_wrapper_layout = QHBoxLayout(self.middle_wrapper_widget)
        self.middle_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        self.middle_wrapper_layout.setSpacing(0)
        self.middle_wrapper_layout.addWidget(self.middle_widget)

        # Add panels to main splitter
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.middle_wrapper_widget)
        self.main_splitter.addWidget(self.right_widget)
        self.main_splitter.setSizes([200, 700, 300])

        # Set stretch factors to ensure middle panel gets priority when resizing (initially)
        self.main_splitter.setStretchFactor(
            0, 0
        )  # Left panel fixed size initially
        self.main_splitter.setStretchFactor(
            1, 1
        )  # Middle wrapper takes extra space initially
        self.main_splitter.setStretchFactor(
            2, 0
        )  # Right panel fixed size initially

        main_layout.addWidget(self.main_splitter)

        # Streaming state
        self.streaming = False
        self.stream_thread = None

        # Add loading indicators
        self.loading_timer = None
        self.loading_dots = 0
        self.loading_message = None

        # Current assistant message for streaming updates
        self.current_assistant_message = None

        # Fetch available models
        models_data = get_models_data(
            self.default_provider,
            self.providers[self.default_provider]["api_address"],
            self.providers[self.default_provider]["api_key"],
        )
        self.selected_model = _model_settings["model_id"]
        self.model_dropdown = ModelDropdown(models_data, self.default_provider)
        self.model_dropdown.hide()
        self.model_dropdown.modelSelected.connect(self.on_model_selected)
        self.model_dropdown.providerSelected.connect(self.on_provider_selected)
        self.model_button.clicked.connect(self.show_model_dropdown)

        # Set focus to the message input
        self.message_input.setFocus()

    def refresh_model_list(self):
        """Refresh the model list for the current provider"""
        for provider in self.providers:
            if getattr(self, f"{provider}_btn").isChecked():
                models_data = get_models_data(
                    provider,
                    self.providers[provider]["api_address"],
                    self.providers[provider]["api_key"],
                )
                self.model_dropdown.update_models_data(models_data, provider)
                break

    def show_model_dropdown(self):
        """Show the model dropdown"""
        self.refresh_model_list()

        button_rect = self.model_button.rect()
        button_pos = self.model_button.mapToGlobal(QPoint(0, 0))
        button_center_x = button_pos.x() + button_rect.width() / 2
        dropdown_x = button_center_x - (self.model_dropdown.width() / 2)
        right_panel_top = self.right_widget.mapToGlobal(QPoint(0, 0)).y()
        available_height = button_pos.y() - right_panel_top
        self.model_dropdown.resize(self.right_widget.width(), available_height)
        dropdown_y = button_pos.y() - self.model_dropdown.height()
        self.model_dropdown.move(int(dropdown_x), int(dropdown_y))
        self.model_dropdown.show()

    def on_model_selected(self, model_name):
        """Handle the model selected event"""
        self.selected_model = model_name
        self.model_button.setText(model_name + f" ({self.default_provider})")
        self.current_api_address = self.providers[self.default_provider][
            "api_address"
        ]
        self.current_api_key = self.providers[self.default_provider]["api_key"]

    def on_provider_selected(self, provider):
        """Handle the provider selected event"""
        getattr(self, f"{provider}_btn").setChecked(True)
        self.default_provider = provider
        self.switch_provider(provider)

    def switch_provider(self, provider):
        """Switch between different model providers"""
        if provider in self.providers:
            # set api address and key
            api_address = self.providers[provider]["api_address"]
            api_key = self.providers[provider]["api_key"]
            self.api_address.setText(api_address)
            self.api_address.setPlaceholderText(
                DEFAULT_PROVIDERS_DATA[provider].get("api_address", "")
            )
            self.api_key.setText(api_key)

            models_data = get_models_data(provider, api_address, api_key)
            self.model_dropdown.update_models_data(models_data, provider)

            # update help button urls
            button_url_mapping = [
                {"button_name": "api_help_btn", "url_key": "api_docs_url"},
                {"button_name": "model_help_btn", "url_key": "model_docs_url"},
                {"button_name": "api_key_help_btn", "url_key": "api_key_url"},
            ]
            for mapping in button_url_mapping:
                button_name, url_key = (
                    mapping["button_name"],
                    mapping["url_key"],
                )
                button = self.findChild(QPushButton, button_name)
                url = self.providers[provider][url_key]
                if button:
                    if url:
                        button.setVisible(True)
                        button.clicked.disconnect()
                        button.clicked.connect(
                            lambda checked=False, u=url: open_url(u)
                        )
                    else:
                        button.setVisible(False)

    def on_api_address_changed(self):
        """Handle the API address changed event"""
        for provider in self.providers:
            if getattr(self, f"{provider}_btn").isChecked():
                self.providers[provider][
                    "api_address"
                ] = self.api_address.text()
                save_json(self.providers, PROVIDERS_CONFIG_PATH)

                if provider == self.default_provider:
                    self.current_api_address = self.api_address.text()
                break

    def on_api_key_changed(self):
        """Handle the API key changed event"""
        for provider in self.providers:
            if getattr(self, f"{provider}_btn").isChecked():
                self.providers[provider]["api_key"] = self.api_key.text()
                save_json(self.providers, PROVIDERS_CONFIG_PATH)

                if provider == self.default_provider:
                    self.current_api_key = self.api_key.text()
                break

    def resize_input(self):
        """Dynamically resize input based on content"""
        # Calculate required height
        document = self.message_input.document()
        doc_height = document.size().height()
        margins = self.message_input.contentsMargins()

        # Calculate total height needed
        total_height = doc_height + margins.top() + margins.bottom() + 4

        # Set a maximum height limit
        max_height = MAX_MSG_INPUT_HEIGHT

        # Determine if we need scrollbars
        needs_scrollbar = total_height > max_height

        # Set the appropriate height
        if needs_scrollbar:
            # Use maximum height and enable scrollbar
            self.message_input.setMinimumHeight(int(max_height))
            self.message_input.setMaximumHeight(int(max_height))
            self.message_input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

            # Ensure cursor is visible by scrolling to it
            cursor = self.message_input.textCursor()
            self.message_input.ensureCursorVisible()
        else:
            # Use calculated height and disable scrollbar
            actual_height = max(total_height, MIN_MSG_INPUT_HEIGHT)
            self.message_input.setMinimumHeight(int(actual_height))
            self.message_input.setMaximumHeight(int(actual_height))
            self.message_input.setVerticalScrollBarPolicy(
                Qt.ScrollBarAlwaysOff
            )

        # Force update to ensure changes take effect immediately
        self.message_input.updateGeometry()
        QApplication.processEvents()

    def restore_send_button(self):
        """Restore the send button to its original state"""
        self.send_btn.setIcon(QIcon(new_icon("send", "svg")))
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.start_generation)

    def update_image_preview(self):
        """Update the image preview when switching images"""
        if self.parent().filename:
            pixmap = QPixmap(self.parent().filename)

            # If pixmap is valid
            if not pixmap.isNull():
                # Calculate scaled size while maintaining aspect ratio
                preview_size = self.image_preview.size()

                # Only scale if the preview has a valid size (width and height > 0)
                if preview_size.width() > 0 and preview_size.height() > 0:
                    scaled_pixmap = pixmap.scaled(
                        preview_size.width(),
                        preview_size.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )

                    self.image_preview.setPixmap(scaled_pixmap)
                    self.image_preview.setAlignment(Qt.AlignCenter)
                else:
                    # If the preview size isn't valid yet, schedule another update
                    QTimer.singleShot(
                        int(ANIMATION_DURATION[:-2]), self.update_image_preview
                    )
            else:
                self.image_preview.setText(self.tr("Image not available"))

    def load_initial_data(self):
        """Load initial data from parent's other_data if available"""
        if self.parent().filename:
            other_data = self.parent().other_data
            if "chat_history" in other_data and isinstance(
                other_data["chat_history"], list
            ):
                for message in other_data["chat_history"]:
                    if "role" in message and "content" in message:
                        self.add_message(message["role"], message["content"])

            self.update_image_preview()

            # Add a slight delay to ensure the widget is fully rendered before scaling the image
            QTimer.singleShot(
                int(ANIMATION_DURATION[:-2]), self.update_image_preview
            )

        self.update_import_buttons_visibility()

    def update_import_buttons_visibility(self):
        """Update visibility of import buttons based on whether files are loaded"""
        has_images = bool(self.parent().image_list)
        # Show navigation buttons only when images are loaded
        self.prev_image_btn.setVisible(has_images)
        self.next_image_btn.setVisible(has_images)

    def open_image_file_or_folder(self, mode="image"):
        """Open an image file or image folder"""
        if mode == "image":
            self.parent().open_file()
            if self.parent().filename:
                self.load_chat_for_current_image()
                self.update_import_buttons_visibility()
        else:
            self.parent().open_folder_dialog()
            if self.parent().image_list:
                self.load_chat_for_current_image()
                self.update_import_buttons_visibility()
                self.run_all_images_btn.setVisible(True)
                self.import_export_btn.setVisible(True)

    def add_message(self, role, content, delete_last_message=False):
        """Add a new message to the chat area"""
        # Remove the stretch item if it exists
        while self.chat_messages_layout.count() > 0:
            item = self.chat_messages_layout.itemAt(
                self.chat_messages_layout.count() - 1
            )
            if item and item.spacerItem():
                self.chat_messages_layout.removeItem(item)
                break

        # Check for special token in user messages
        image = None
        if role == "user" and ("@image" in content or "<image>" in content):
            pattern = r"@image\s*(?=\S|$)"
            content = re.sub(pattern, "<image>", content).strip()
            image = self.parent().filename

        # Create and add the message widget
        is_error = True if delete_last_message else False
        message_widget = ChatMessage(
            role,
            content,
            self.default_provider,
            self.chat_container,
            is_error=is_error,
        )
        self.chat_messages_layout.addWidget(message_widget)

        # Add the stretch back
        self.chat_messages_layout.addStretch()

        # Store message in chat history
        if delete_last_message:
            self.chat_history = self.chat_history[
                :-1
            ]  # roll back the last message
        else:
            self.chat_history.append(
                {"role": role, "content": content, "image": image}
            )

        # Scroll to bottom - use a longer delay to ensure layout is updated
        QTimer.singleShot(int(ANIMATION_DURATION[:-2]), self.scroll_to_bottom)

        # Update parent data
        if (
            role == "assistant"
            and self.parent().filename
            and not delete_last_message
        ):
            self.parent().other_data["chat_history"] = self.chat_history

    def scroll_to_bottom(self):
        """Scroll chat area to the bottom"""
        scrollbar = self.chat_scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Force update to ensure scrolling takes effect
        QApplication.processEvents()

    def start_generation(self):
        """Start generating a response"""
        if self.streaming:
            return

        user_message = self.message_input.toPlainText().strip()
        if not user_message:
            return

        # Add user message to chat history first
        self.add_message("user", user_message)

        # Clear input and reset to plain text mode
        self.message_input.clear()
        self.message_input.setPlainText(
            ""
        )  # Ensure we're back to plain text mode

        # Reset input box size to initial height
        self.message_input.setMinimumHeight(24)
        self.message_input.setMaximumHeight(24)

        # Start generation process
        self.streaming = True
        self.set_components_enabled(False)

        # Change send button to stop button
        self.send_btn.setIcon(QIcon(new_icon("stop", "svg")))
        self.send_btn.setEnabled(True)
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.stop_generation)

        # Create loading message
        self.add_loading_message()

        # Reset stream handler
        self.stream_handler.reset()

        # Start streaming in a separate thread
        self.stream_thread = threading.Thread(
            target=self.stream_generation,
            args=(self.current_api_address, self.current_api_key),
        )
        self.stream_thread.daemon = (
            True  # Make thread daemonic so it can be interrupted
        )
        self.stream_thread.start()

    def stop_generation(self):
        """Stop the ongoing generation process"""
        self.stream_handler.stop_requested = True

    def add_loading_message(self):
        """Add a loading message that will be replaced with the actual response"""
        while self.chat_messages_layout.count() > 0:
            item = self.chat_messages_layout.itemAt(
                self.chat_messages_layout.count() - 1
            )
            if item and item.spacerItem():
                self.chat_messages_layout.removeItem(item)
                break

        # Create a loading message widget
        self.loading_message = QFrame(self.chat_container)
        loading_layout = QVBoxLayout(self.loading_message)
        loading_layout.setContentsMargins(0, 0, 0, 0)

        # Create bubble with smooth corners
        bubble = QWidget(self.loading_message)
        bubble.setObjectName("messageBubble")
        bubble.setStyleSheet(ChatMessageStyle.get_bubble_style(is_user=False))

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 12, 12, 12)

        # Add header
        header_layout = QHBoxLayout()

        # Create a container frame for the icon with rounded background
        icon_container = QFrame()
        icon_container.setObjectName("roleLabelContainer")
        icon_container.setStyleSheet(
            ChatMessageStyle.get_role_label_background_style()
        )

        # Create fixed-size layout for the container
        icon_container_layout = QHBoxLayout(icon_container)
        icon_container_layout.setContentsMargins(2, 2, 2, 2)
        icon_container_layout.setSpacing(0)

        # Create the icon label
        role_label = QLabel()
        icon_pixmap = QPixmap(new_icon_path(self.default_provider.lower()))
        scaled_icon = icon_pixmap.scaled(
            *ICON_SIZE_SMALL, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        role_label.setPixmap(scaled_icon)
        role_label.setStyleSheet(ChatMessageStyle.get_role_label_style())

        # Add to layout
        icon_container_layout.addWidget(role_label)
        header_layout.addWidget(icon_container)
        header_layout.addStretch()
        bubble_layout.addLayout(header_layout)

        # Create layout for loading indicator and text
        loading_indicator_layout = QHBoxLayout()

        # Add pulsating dot animation
        self.pulsating_dot = PulsatingDot(
            bubble,
            size_range=(8, 16),
            color_range=((30, 30, 30), (150, 150, 150)),
            duration=500,
        )
        loading_indicator_layout.addWidget(self.pulsating_dot)
        loading_indicator_layout.addStretch()
        bubble_layout.addLayout(loading_indicator_layout)

        # Set maximum width for the bubble
        bubble.setMaximumWidth(2000)
        loading_layout.addWidget(bubble)
        loading_layout.setAlignment(Qt.AlignLeft)

        # Store bubble reference for later updates
        self.loading_message.bubble = bubble

        self.chat_messages_layout.addWidget(self.loading_message)
        self.chat_messages_layout.addStretch()
        QTimer.singleShot(0, self.scroll_to_bottom)

    def update_output(self, text):
        """Update the output text with streaming content"""
        if self.loading_message:
            # Stop the pulsating dot animation
            if hasattr(self, "pulsating_dot") and self.pulsating_dot:
                self.pulsating_dot.stop_animation()
                self.pulsating_dot.setParent(None)
                self.pulsating_dot.deleteLater()
                self.pulsating_dot = None

            # If this is the first update, create a content label
            if not hasattr(self.loading_message, "content_label"):
                self.loading_message.content_label = QLabel("")
                self.loading_message.content_label.setWordWrap(True)
                self.loading_message.content_label.setTextFormat(Qt.PlainText)
                self.loading_message.content_label.setStyleSheet(
                    ChatMessageStyle.get_fade_in_text_style()
                )
                self.loading_message.content_label.setMinimumWidth(100)
                self.loading_message.content_label.setMaximumWidth(1999)
                self.loading_message.bubble.layout().addWidget(
                    self.loading_message.content_label
                )

            current_text = self.loading_message.content_label.text()
            self.loading_message.content_label.setText(current_text + text)
            self.scroll_to_bottom()

    def on_stream_finished(self, success):
        """Handle completion of streaming"""
        # Stop the loading timer if it's still active
        if self.loading_timer and self.loading_timer.isActive():
            self.loading_timer.stop()
        
        logger.info(f"====!!!Stream finished with success====: {success}")

        if success and self.loading_message:
            # Get the final text
            final_text = ""
            if hasattr(self.loading_message, "content_label"):
                animation = QPropertyAnimation(
                    self.loading_message.content_label, b"styleSheet"
                )
                animation.setDuration(300)
                animation.setStartValue(
                    ChatMessageStyle.get_animation_style(0.5)
                )
                animation.setEndValue(
                    ChatMessageStyle.get_animation_style(1.0)
                )
                animation.start()
                final_text = self.loading_message.content_label.text()

            # Store reference to loading message for removal
            loading_message_to_remove = self.loading_message
            self.loading_message = None

            # Add the final message first
            self.add_message("assistant", final_text)

            # Then remove the loading message after adding the final message
            loading_message_to_remove.setParent(None)
            loading_message_to_remove.deleteLater()

            logger.info("=====Assistant message added to chat history====")  
            # Set dirty flag
            if self.parent().filename:
                self.parent().set_dirty()
                logger.info("=====Chat set_dirty saved====")
                # Save chat history with "chatbot_" prefix
                self.save_chat_history_with_prefix()
                logger.info("=====Chat history saved with prefix====")
                # Automatically generate XLABEL format file from assistant response
                self.generate_xlabel_from_assistant_response(final_text)
                logger.info("=====XLABEL generation attempted====")
                
    
        # Reset streaming state
        self.streaming = False
        self.set_components_enabled(True)

        # Auto focus on message input after generation
        self.message_input.setFocus()
        
    def extract_and_save_labeling_result(self, assistant_content, image_file):
        """
        Extract bounding box information from assistant content and save as XLabel format
        """
        try:
            # Import required modules
            from anylabeling.views.labeling.label_converter import LabelConverter
            import os.path as osp
            
            logger.info(f"Attempting to extract bounding box data from assistant content. Content length: {len(assistant_content)}")
            
            # Try to parse as Qwen3-VL-Thinking format first
            bbox_data = LabelConverter().extract_qwen3_vl_thinking_answer(assistant_content)
            logger.info(f"Qwen3-VL-Thinking format extraction result: {len(bbox_data)} boxes found")
            
            # If that fails, try to parse as standard JSON format
            if not bbox_data:
                bbox_data = LabelConverter().extract_bbox_answer(assistant_content)
                logger.info(f"Standard JSON format extraction result: {len(bbox_data)} boxes found")
            
            # Try to parse as direct JSON object format (for nested JSON structures)
            if not bbox_data:
                try:
                    import json
                    import re
                    # Look for JSON objects in the content
                    json_patterns = [
                        r'```(?:json)?\s*({.*?})\s*```',  # Code blocks with or without json specifier
                        r'<output>\s*({.*?})\s*</output>',  # Output tags
                        r'<answer>\s*({.*?})\s*</answer>',  # Answer tags
                        r'({.*?"bbox_2d".*?})'  # Direct JSON with bbox_2d field
                    ]
                    
                    for pattern in json_patterns:
                        matches = re.findall(pattern, assistant_content, re.DOTALL)
                        for match in matches:
                            try:
                                # Try to parse the matched content as JSON
                                data = json.loads(match)
                                if isinstance(data, list):
                                    bbox_data = data
                                    logger.info(f"Direct JSON list format extraction result: {len(bbox_data)} boxes found")
                                    break
                                elif isinstance(data, dict) and "bbox_2d" in data:
                                    bbox_data = [data]
                                    logger.info(f"Direct JSON object format extraction result: {len(bbox_data)} boxes found")
                                    break
                            except json.JSONDecodeError:
                                continue
                        if bbox_data:
                            break
                except Exception as e:
                    logger.error(f"Error during direct JSON parsing: {str(e)}")
            
            # If we have bbox data, convert to XLabel format and save
            if bbox_data:
                logger.info(f"Successfully extracted {len(bbox_data)} bounding boxes")
                # Generate label file path with chatbot_ prefix
                base_filename = osp.splitext(osp.basename(image_file))[0]
                label_file = f"chatbot_{base_filename}.json"
                if self.parent().output_dir:
                    label_file = osp.join(self.parent().output_dir, label_file)
                else:
                    label_file = osp.join(osp.dirname(image_file), label_file)
                
                logger.info(f"Saving labeling result to: {label_file}")
                
                # Create converter and convert
                converter = LabelConverter()
                converter.vlm_r1_ovd_to_custom(
                    input_data=assistant_content,
                    output_file=label_file,
                    image_file=image_file
                )
                
                # Use a QTimer to delay the file loading to avoid modal session conflicts
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(100, lambda: self.parent().load_file(image_file))
            else:
                logger.warning("No bounding box data found in assistant content")
                
        except Exception as e:
            # Log error but don't interrupt the chat flow
            logger.error(f"Error extracting labeling result: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_xlabel_from_assistant_response(self, assistant_response):
        """
        Generate XLABEL format file from assistant response automatically
        """
        try:
            import json
            import re
            import os.path as osp
            
            # 正则表达式匹配代码块
            code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
            
            # 查找所有代码块
            matches = re.findall(code_block_pattern, assistant_response, re.DOTALL)
            
            if not matches:
                logger.info("No JSON code block found in assistant response")
                return
                
            # 解析第一个有效的JSON代码块
            bbox_data = None
            for match in matches:
                try:
                    bbox_data = json.loads(match)
                    break
                except json.JSONDecodeError:
                    continue
                    
            if not bbox_data:
                logger.error("Failed to parse JSON from assistant response")
                return
            
            # 构建 XLABEL 格式
            image_filename = osp.basename(self.parent().filename)
            xlabel_data = {
                "version": "4.0",
                "flags": {},
                "description": "",
                "shapes": [],
                "imagePath": image_filename,
                "imageData": None
            }
            
            # 获取图像实际尺寸
            pixmap = QPixmap(self.parent().filename)
            if not pixmap.isNull():
                image_width = pixmap.width()
                image_height = pixmap.height()
            else:
                # 默认尺寸
                image_width = 1024
                image_height = 538
            
            # 添加边界框到 shapes
            for item in bbox_data:
                if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                    x1, y1, x2, y2 = item["bbox_2d"]
                    
                    # 确保坐标在有效范围内
                    x1 = max(0, min(x1, image_width))
                    y1 = max(0, min(y1, image_height))
                    x2 = max(0, min(x2, image_width))
                    y2 = max(0, min(y2, image_height))
                    
                    label = item["label"]
                    
                    # 创建矩形框的四个点（左上、右上、右下、左下）
                    shape = {
                        "label": label,
                        "points": [
                            [x1, y1],  # 左上
                            [x2, y1],  # 右上
                            [x2, y2],  # 右下
                            [x1, y2]   # 左下
                        ],
                        "shape_type": "rectangle",
                        "line_color": None,
                        "fill_color": None,
                        "group_id": None,
                        "source": "manual",
                        "attributes": {}
                    }
                    xlabel_data["shapes"].append(shape)
            
            # 设置正确的图像尺寸
            xlabel_data["imageWidth"] = image_width
            xlabel_data["imageHeight"] = image_height
            
            # 保存为 XLABEL 文件，与图像同目录
            output_dir = osp.dirname(self.parent().filename) if not self.parent().output_dir else self.parent().output_dir
            output_file = osp.join(output_dir, f"{osp.splitext(image_filename)[0]}.json")
            
            with open(output_file, 'w', encoding='utf-8') as out_f:
                json.dump(xlabel_data, out_f, ensure_ascii=False, indent=2)
            
            logger.info(f"XLABEL file generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating XLABEL from assistant response: {str(e)}")

    def handle_loading_state(self, is_loading):
        """Handle loading state changes"""
        if is_loading:
            # Disable UI components during loading
            self.set_components_enabled(False)
        else:
            # Re-enable UI components after loading
            self.set_components_enabled(True)

    def toggle_api_key_visibility(self):
        """Toggle visibility of API key"""
        if self.api_key.echoMode() == QLineEdit.Password:
            self.api_key.setEchoMode(QLineEdit.Normal)
            self.toggle_visibility_btn.setIcon(QIcon(new_icon("eye", "svg")))
        else:
            self.api_key.setEchoMode(QLineEdit.Password)
            self.toggle_visibility_btn.setIcon(
                QIcon(new_icon("eye-off", "svg"))
            )

    def navigate_image(self, direction="next", index=None):
        """Navigate to previous or next image and load its chat history

        Args:
            direction (str): Direction to navigate, either "next" or "prev"
            index (int): Index of the image to navigate to
        """
        try:
            if self.parent().image_list:
                if index is None:
                    current_index = self.parent().image_list.index(
                        self.parent().filename
                    )

                    if direction == "prev" and current_index > 0:
                        new_index = current_index - 1
                    elif (
                        direction == "next"
                        and current_index < len(self.parent().image_list) - 1
                    ):
                        new_index = current_index + 1
                    else:
                        return
                else:
                    new_index = index

                new_file = self.parent().image_list[new_index]
                self.parent().load_file(new_file)

                # Force UI and data updates
                QApplication.processEvents()
                self.update_image_preview()
                self.load_chat_for_current_image()
                self.update_import_buttons_visibility()
        except Exception as e:
            logger.error(f"Error navigating to {direction} image: {e}")

    def load_chat_for_current_image(self):
        """Load chat history for the current image"""
        try:
            while self.chat_messages_layout.count() > 0:
                item = self.chat_messages_layout.takeAt(0)
                if item and item.widget():
                    widget = item.widget()
                    widget.setParent(None)
                    widget.deleteLater()
                elif item:
                    self.chat_messages_layout.removeItem(item)

            self.chat_history = []
            self.chat_messages_layout.addStretch()
            QApplication.processEvents()
            self.load_initial_data()

        except Exception as e:
            logger.error(f"Error loading chat for current image: {e}")

    def save_chat_history_with_prefix(self):
        """Save chat history with 'chatbot_' prefix"""
        logger.info("====Saving chat history with 'chatbot_' prefix...====")
        try:
            # Get current image filename without extension
            image_filename = osp.basename(self.parent().filename)
            base_filename = osp.splitext(image_filename)[0]
            
            # Create chatbot filename with prefix
            chatbot_filename = f"chatbot_{base_filename}.json"
            
            # Determine save path
            if self.parent().output_dir:
                chatbot_filepath = osp.join(self.parent().output_dir, chatbot_filename)
            else:
                chatbot_filepath = osp.join(osp.dirname(self.parent().filename), chatbot_filename)
            
            # Load existing label file data
            label_file_data = {}
            
            # Instead of using parent().get_label_file(), we directly use the chatbot file path
            # to avoid creating the non-prefixed file
            if osp.exists(chatbot_filepath):
                with open(chatbot_filepath, 'r', encoding='utf-8') as f:
                    label_file_data = json.load(f)
            
            # Update chat history in the data
            label_file_data["chat_history"] = self.parent().other_data.get("chat_history", [])
            
            # Save to chatbot prefixed file
            with open(chatbot_filepath, 'w', encoding='utf-8') as f:
                json.dump(label_file_data, f, ensure_ascii=False, indent=2)
                
            # # Automatically generate VLM-R1-OVD format file
            # self.generate_vlm_r1_ovd_from_chat_history(label_file_data, base_filename)
                
        except Exception as e:
            logger.error(f"Error saving chat history with prefix: {e}")

    def generate_vlm_r1_ovd_file(self, assistant_content, image_file):
        """
        Generate VLM-R1-OVD format file from assistant content automatically
        """
        try:
            import os
            import json
            from anylabeling.views.labeling.label_converter import LabelConverter
            
            # Get the base name of the image file
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            
            # Determine output directory
            output_dir = self.parent().output_dir if self.parent().output_dir else os.path.dirname(image_file)
            
            # Create VLM-R1-OVD entry
            vlm_r1_ovd_entry = {
                "id": 1,  # We'll use 1 as default ID for single entry
                "image": os.path.basename(image_file),
                "conversations": [
                    {
                        "from": "assistant",
                        "value": assistant_content
                    }
                ]
            }
            
            # Generate VLM-R1-OVD filename with "VLM_R1_OVD_" prefix
            vlm_r1_ovd_filename = f"VLM_R1_OVD_{base_name}.jsonl"
            vlm_r1_ovd_file_path = os.path.join(output_dir, vlm_r1_ovd_filename)
            
            # Save to file
            with open(vlm_r1_ovd_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(vlm_r1_ovd_entry, ensure_ascii=False) + '\n')
            
            logger.info(f"VLM-R1-OVD file saved: {vlm_r1_ovd_file_path}")
            
            # Convert the VLM-R1-OVD entry to XLABEL format for visualization
            self._convert_vlm_r1_ovd_entry_to_xlabel(vlm_r1_ovd_entry, output_dir)
            
            # Also convert using the new general method
            self._convert_vlm_r1_ovd_to_xlabel(vlm_r1_ovd_file_path, output_dir)
            
        except Exception as e:
            logger.error(f"Error generating VLM-R1-OVD file: {str(e)}")

    def generate_vlm_r1_ovd_from_chat_history(self, label_file_data, base_filename):
        """
        Generate VLM-R1-OVD format file from chat history automatically
        """
        try:
            import json
            
            # Determine output directory
            output_dir = self.parent().output_dir if self.parent().output_dir else osp.dirname(self.parent().filename)
            
            # Extract chat history
            chat_history = label_file_data.get("chat_history", [])
            
            if not chat_history:
                logger.info("No chat history found")
                return
            
            # Generate VLM-R1-OVD filename with folder name prefix
            folder_name = osp.basename(output_dir) if output_dir else "default"
            vlm_r1_ovd_filename = f"VLM_R1_OVD_{folder_name}.jsonl"
            vlm_r1_ovd_file_path = osp.join(output_dir, vlm_r1_ovd_filename)
            
            # Create conversation format for VLM-R1-OVD
            conversations = []
            for msg in chat_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Map roles to VLM-R1-OVD format
                if role == "user":
                    from_role = "human"
                elif role == "assistant":
                    from_role = "assistant"
                else:
                    from_role = role
                    
                conversations.append({
                    "from": from_role,
                    "value": content
                })
            
            # Create VLM-R1-OVD entry
            entry = {
                "id": 1,  # Default ID, will be updated when writing to file
                "image": osp.basename(self.parent().filename),
                "conversations": conversations
            }
            
            # Read existing entries
            existing_entries = []
            if osp.exists(vlm_r1_ovd_file_path):
                try:
                    with open(vlm_r1_ovd_file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                existing_entries.append(json.loads(line.strip()))
                except Exception as e:
                    logger.warning(f"Could not read existing VLM-R1-OVD file: {e}")
            
            # Check if an entry for this image already exists
            image_name = osp.basename(self.parent().filename)
            entry_found = False
            for i, existing_entry in enumerate(existing_entries):
                if existing_entry.get("image") == image_name:
                    # Update existing entry
                    existing_entries[i] = entry
                    entry_found = True
                    break
            
            # If not found, add as new entry
            if not entry_found:
                existing_entries.append(entry)
            
            # Write all entries back to file
            with open(vlm_r1_ovd_file_path, 'w', encoding='utf-8') as f:
                for entry in existing_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.info(f"VLM-R1-OVD file updated: {vlm_r1_ovd_file_path}")
            
            # Convert VLM-R1-OVD to XLABEL format automatically
            self._convert_vlm_r1_ovd_to_xlabel(vlm_r1_ovd_file_path, output_dir)
            
        except Exception as e:
            logger.error(f"Error generating VLM-R1-OVD from chat history: {str(e)}")

    def _convert_vlm_r1_ovd_entry_to_xlabel(self, entry, output_dir):
        """
        Convert a single VLM-R1-OVD entry to XLABEL format for visualization
        """
        try:
            from anylabeling.views.labeling.label_converter import LabelConverter
            import json
            import re
            
            # Get image file path
            image_filename = entry.get("image", "")
            if not image_filename:
                logger.warning("No image filename found in VLM-R1-OVD entry")
                return
                
            # Find the full image path
            image_file = None
            if self.parent().filename and osp.basename(self.parent().filename) == image_filename:
                image_file = self.parent().filename
            else:
                # Try to find the image in the image list
                for img_path in self.parent().image_list:
                    if osp.basename(img_path) == image_filename:
                        image_file = img_path
                        break
            
            if not image_file or not osp.exists(image_file):
                # Try to construct path relative to output directory
                image_file = osp.join(output_dir, image_filename)
                if not osp.exists(image_file):
                    logger.warning(f"Image file not found: {image_file}")
                    return
            
            # Extract assistant content from conversations
            conversations = entry.get("conversations", [])
            assistant_content = ""
            for conv in conversations:
                if conv.get("from") == "assistant":
                    assistant_content = conv.get("value", "")
                    break
            
            if not assistant_content:
                logger.warning(f"No assistant content found for image {image_filename}")
                return
            
            # Extract bounding box data using the general method
            bbox_data = self._extract_bbox_data(assistant_content)
            
            if not bbox_data:
                logger.warning(f"No bounding box data extracted for image {image_filename}")
                return
            
            # Create label converter
            converter = LabelConverter()
            
            # Generate XLABEL filename
            xlabel_filename = osp.splitext(image_filename)[0] + ".json"
            xlabel_file_path = osp.join(output_dir, xlabel_filename)
            
            # Get image dimensions
            image_width, image_height = converter.get_image_size(image_file)
            if image_width == 0 or image_height == 0:
                # Use default dimensions if we can't get the real ones
                image_width, image_height = 1024, 538
            
            # Convert to XLABEL format
            shapes = []
            for item in bbox_data:
                label = item.get("label", "object")
                bbox = item.get("bbox_2d", [])
                
                if len(bbox) != 4:
                    continue
                    
                # Convert normalized coordinates to pixel coordinates if needed
                if all(0 <= coord <= 1 for coord in bbox):
                    # Normalized coordinates
                    x1 = bbox[0] * image_width
                    y1 = bbox[1] * image_height
                    x2 = bbox[2] * image_width
                    y2 = bbox[3] * image_height
                else:
                    # Pixel coordinates
                    x1, y1, x2, y2 = bbox
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(image_width, x1))
                y1 = max(0, min(image_height, y1))
                x2 = max(0, min(image_width, x2))
                y2 = max(0, min(image_height, y2))
                
                # Create rectangle points (top-left, top-right, bottom-right, bottom-left)
                points = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]
                
                shape = {
                    "label": label,
                    "text": "",
                    "points": points,
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "direction": 0,  # Add direction field
                }
                
                shapes.append(shape)
            
            # Create XLABEL format data
            xlabel_data = {
                "version": "1.0",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_filename,
                "imageData": None,
                "imageHeight": image_height,
                "imageWidth": image_width
            }
            
            # Write to file
            with open(xlabel_file_path, "w", encoding="utf-8") as f:
                json.dump(xlabel_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"XLABEL file generated: {xlabel_file_path}")
            
        except Exception as e:
            logger.error(f"Error converting VLM-R1-OVD entry to XLABEL: {str(e)}")
            import traceback
            traceback.print_exc()

    def _convert_vlm_r1_ovd_to_xlabel(self, vlm_r1_ovd_file_path, output_dir):
        """
        Convert VLM-R1-OVD format file to XLABEL format for visualization
        
        Args:
            vlm_r1_ovd_file_path (str): Path to VLM-R1-OVD JSONL file
            output_dir (str): Output directory for XLABEL files
        """
        try:
            from anylabeling.views.labeling.label_converter import LabelConverter
            import json
            import re
            
            # Read VLM-R1-OVD file
            with open(vlm_r1_ovd_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse JSON line in {vlm_r1_ovd_file_path}: {e}")
                        continue
                    
                    # Get image file path
                    image_filename = entry.get("image", "")
                    if not image_filename:
                        logger.warning("No image filename found in VLM-R1-OVD entry")
                        continue
                        
                    # Find the full image path
                    image_file = None
                    if self.parent().filename and osp.basename(self.parent().filename) == image_filename:
                        image_file = self.parent().filename
                    else:
                        # Try to find the image in the image list
                        for img_path in self.parent().image_list:
                            if osp.basename(img_path) == image_filename:
                                image_file = img_path
                                break
                    
                    if not image_file or not osp.exists(image_file):
                        # Try to construct path relative to output directory
                        image_file = osp.join(output_dir, image_filename)
                        if not osp.exists(image_file):
                            logger.warning(f"Image file not found: {image_file}")
                            continue
                    
                    # Extract assistant content from conversations
                    conversations = entry.get("conversations", [])
                    assistant_content = ""
                    for conv in conversations:
                        if conv.get("from") == "assistant":
                            assistant_content = conv.get("value", "")
                            break
                    
                    if not assistant_content:
                        logger.warning(f"No assistant content found for image {image_filename}")
                        continue
                    
                    # Extract bounding box data using the general method
                    bbox_data = self._extract_bbox_data(assistant_content)
                    
                    if not bbox_data:
                        logger.warning(f"No bounding box data extracted for image {image_filename}")
                        continue
                    
                    # Create label converter
                    converter = LabelConverter()
                    
                    # Generate XLABEL filename
                    xlabel_filename = osp.splitext(image_filename)[0] + ".json"
                    xlabel_file_path = osp.join(output_dir, xlabel_filename)
                    
                    # Get image dimensions
                    image_width, image_height = converter.get_image_size(image_file)
                    if image_width == 0 or image_height == 0:
                        # Use default dimensions if we can't get the real ones
                        image_width, image_height = 1024, 538
                    
                    # Convert to XLABEL format
                    shapes = []
                    for item in bbox_data:
                        label = item.get("label", "object")
                        bbox = item.get("bbox_2d", [])
                        
                        if len(bbox) != 4:
                            continue
                            
                        # Convert normalized coordinates to pixel coordinates if needed
                        if all(0 <= coord <= 1 for coord in bbox):
                            # Normalized coordinates
                            x1 = bbox[0] * image_width
                            y1 = bbox[1] * image_height
                            x2 = bbox[2] * image_width
                            y2 = bbox[3] * image_height
                        else:
                            # Pixel coordinates
                            x1, y1, x2, y2 = bbox
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(image_width, x1))
                        y1 = max(0, min(image_height, y1))
                        x2 = max(0, min(image_width, x2))
                        y2 = max(0, min(image_height, y2))
                        
                        # Create rectangle points (top-left, top-right, bottom-right, bottom-left)
                        points = [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ]
                        
                        shape = {
                            "label": label,
                            "text": "",
                            "points": points,
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {},
                            "direction": 0,
                        }
                        
                        shapes.append(shape)
                    
                    # Create XLABEL format data
                    xlabel_data = {
                        "version": "1.0",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": image_filename,
                        "imageData": None,
                        "imageHeight": image_height,
                        "imageWidth": image_width
                    }
                    
                    # Write to file
                    with open(xlabel_file_path, "w", encoding="utf-8") as f:
                        json.dump(xlabel_data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"XLABEL file generated: {xlabel_file_path}")
                    
        except Exception as e:
            logger.error(f"Error converting VLM-R1-OVD to XLABEL: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def _extract_bbox_data(self, assistant_content):
        """
        从assistant的回复中提取边界框数据
        
        Args:
            assistant_content (str): assistant的回复内容
            
        Returns:
            list: 包含边界框信息的字典列表
        """
        bbox_data = []
        
        # 方法1: 尝试直接解析整个内容为JSON
        try:
            data = json.loads(assistant_content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                        bbox_data.append(item)
            elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                bbox_data.append(data)
        except json.JSONDecodeError:
            pass
        
        # 方法2: 尝试提取代码块中的JSON
        if not bbox_data:
            code_block_pattern = r'```(?:\w+)?\s*(.*?)\s*```'
            code_blocks = re.findall(code_block_pattern, assistant_content, re.DOTALL)
            for block in code_blocks:
                try:
                    data = json.loads(block)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                                bbox_data.append(item)
                    elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                        bbox_data.append(data)
                    break  # 找到第一个有效的代码块就停止
                except json.JSONDecodeError:
                    continue
        
        # 方法3: 尝试提取<answer>标签中的内容
        if not bbox_data:
            answer_pattern = r'<answer>(.*?)</answer>'
            answer_matches = re.findall(answer_pattern, assistant_content, re.DOTALL)
            for match in answer_matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                                bbox_data.append(item)
                    elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                        bbox_data.append(data)
                    break  # 找到第一个有效的answer就停止
                except json.JSONDecodeError:
                    # 尝试在answer中查找代码块
                    code_block_pattern = r'```(?:\w+)?\s*(.*?)\s*```'
                    code_blocks = re.findall(code_block_pattern, match, re.DOTALL)
                    for block in code_blocks:
                        try:
                            data = json.loads(block)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                                        bbox_data.append(item)
                            elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                                bbox_data.append(data)
                            break  # 找到第一个有效的代码块就停止
                        except json.JSONDecodeError:
                            continue
                    if bbox_data:
                        break
        
        # 方法4: 尝试提取Qwen3-VL-Thinking格式的答案
        if not bbox_data:
            pattern = r'bbox2d\s+([^\s$]+?)\s*$(.*?)$'
            matches = re.findall(pattern, assistant_content)
            for label, coords_str in matches:
                try:
                    coords = json.loads(f"[{coords_str}]")
                    if len(coords) == 4:
                        bbox_data.append({
                            "label": label,
                            "bbox_2d": coords
                        })
                except json.JSONDecodeError:
                    # 尝试手动解析坐标
                    coord_pattern = r'(\d+(?:\.\d+)?)'
                    coord_matches = re.findall(coord_pattern, coords_str)
                    if len(coord_matches) == 4:
                        coords = [float(c) for c in coord_matches]
                        bbox_data.append({
                            "label": label,
                            "bbox_2d": coords
                        })
        
        return bbox_data
            
    def run_all_images(self):
        """Run all images with the same prompt for batch processing"""
        if len(self.parent().image_list) <= 0:
            return

        batch_dialog = BatchProcessDialog(self)
        prompt = batch_dialog.exec_()
        if prompt:
            self.current_index = self.parent().fn_to_index[
                str(self.parent().filename)
            ]
            self.image_index = self.current_index
            self.show_progress_dialog_and_process(prompt)

    def show_progress_dialog_and_process(self, prompt):
        self.cancel_processing = False
        progress_dialog = QProgressDialog(
            self.tr("Inferencing..."),
            self.tr("Cancel"),
            self.image_index,
            len(self.parent().image_list),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            ChatbotDialogStyle.get_progress_dialog_style()
        )
        progress_dialog.setFixedSize(400, 150)
        progress_dialog.setWindowFlags(
            progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        progress_dialog.setMinimumDuration(0)

        center_point = self.mapToGlobal(self.rect().center())
        dialog_rect = progress_dialog.rect()
        progress_dialog.move(
            center_point.x() - dialog_rect.width() // 2,
            center_point.y() - dialog_rect.height() // 2,
        )

        progress_dialog.canceled.connect(self.cancel_operation)
        progress_dialog.show()
        QApplication.processEvents()
        self.process_next_image(progress_dialog, prompt)

    def process_next_image(self, progress_dialog, prompt):
        total_images = len(self.parent().image_list)
        logger.info(f"====Processing image==== {self.image_index + 1}/{total_images}")
        if (self.image_index < total_images) and (not self.cancel_processing):
            template = self.tr("Processing image %d/%d...")
            display_text = template % (self.image_index + 1, total_images)
            progress_dialog.setLabelText(display_text)

            filename = self.parent().image_list[self.image_index]
            self.parent().filename = filename

            temperature = self.temp_slider.value() / 10.0
            system_prompt = self.system_prompt_input.text().strip()
            max_tokens = int(self.max_length_input.text().strip())

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if "@image" in prompt or "<image>" in prompt:
                pattern = r"@image\s*(?=\S|$)"
                prompt = re.sub(pattern, "<image>", prompt).strip()
                with open(self.parent().filename, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": prompt})

            client = OpenAI(
                base_url=self.current_api_address, api_key=self.current_api_key
            )
            response = client.chat.completions.create(
                model=self.selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            
            logger.info(f"Response from model: {response}")  # 添加日志
                    
            # 添加检查以防止索引越界
            if not response.choices or len(response.choices) == 0:
                logger.error("No choices returned from model response")
                content = ""
            else:
                content = response.choices[0].message.content
                logger.info(f"Assistant content: {content}")  # 添加日志


            self.parent().other_data["chat_history"] = [
                {
                    "role": "user",
                    "content": prompt,
                    "image": self.parent().filename,
                },
                {"role": "assistant", "content": content, "image": None},
            ]
            self.parent().set_dirty()
            logger.info(f"Line1716，save parent dirty for {self.parent().filename}")
            # Save chat history with "chatbot_" prefix for each image during batch processing
            self.save_chat_history_with_prefix()
            logger.info(f"Line1719，save chat with prefix ====")
            # Generate VLM-R1-OVD format file automatically
            self.generate_vlm_r1_ovd_from_chat_history(
                {"chat_history": self.parent().other_data["chat_history"]},
                osp.splitext(osp.basename(self.parent().filename))[0]
            )
            
            # 移除对不存在方法的调用
            # self.generate_xlabel_from_chat_history(...)  # 这个方法不存在
            progress_dialog.setValue(self.image_index + 1)
            self.image_index += 1
            self.navigate_image()

            if not self.cancel_processing:
                delay_ms = 0
                QTimer.singleShot(
                    delay_ms,
                    lambda: self.process_next_image(progress_dialog, prompt),
                )
            else:
                self.finish_processing(progress_dialog)
        else:
            self.finish_processing(progress_dialog)

    def cancel_operation(self):
        self.cancel_processing = True

    def finish_processing(self, progress_dialog):
        self.parent().filename = self.parent().image_list[self.current_index]
        self.navigate_image(index=self.current_index)
        del self.image_index
        del self.current_index
        progress_dialog.close()
        
        # Save chatbot history with prefix after batch processing
        self.save_chat_history_with_prefix()
        

    def import_export_dataset(self):
        """Import/Export the dataset"""
        option_dialog = QDialog(self)
        option_dialog.setWindowTitle(self.tr("Dataset Operations"))
        option_dialog.setFixedSize(400, 200)
        option_dialog.setStyleSheet(
            ChatbotDialogStyle.get_option_dialog_style()
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

        import_btn = QPushButton(self.tr("Import Dataset"))
        import_btn.setCursor(Qt.PointingHandCursor)

        export_btn = QPushButton(self.tr("Export Dataset"))
        export_btn.setCursor(Qt.PointingHandCursor)

        layout.addWidget(import_btn)
        layout.addWidget(export_btn)
        option_dialog.setLayout(layout)

        def export_dataset():
            option_dialog.accept()

            if not self.parent().filename:
                QMessageBox.warning(
                    self,
                    self.tr("Export Error"),
                    self.tr("No file is currently open."),
                )
                return

            current_dir = os.path.dirname(self.parent().filename)
            export_dir = QFileDialog.getExistingDirectory(
                self,
                self.tr("Select Export Directory"),
                current_dir,
                QFileDialog.ShowDirsOnly,
            )

            if not export_dir:
                return

            # Create timestamp for filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"sharegpt_mllm_data_{timestamp}"
            zip_filename = f"{export_filename}.zip"
            zip_path = os.path.join(export_dir, zip_filename)

            # Create temporary directory structure
            temp_dir = os.path.join(os.path.expanduser("~"), ".temp_export")
            temp_images_dir = os.path.join(temp_dir, "images")

            # Clean previous temp dir if exists
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_images_dir, exist_ok=True)

            try:
                json_files = [
                    f for f in os.listdir(current_dir) if f.endswith(".json")
                ]

                if not json_files:
                    QMessageBox.warning(
                        self,
                        self.tr("Export Error"),
                        self.tr(
                            "No labeling files found in the current directory."
                        ),
                    )
                    return

                export_data = []
                for json_file in json_files:
                    file_path = os.path.join(current_dir, json_file)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if not data.get("chat_history"):
                            continue

                        image_filename = data.get("imagePath")
                        if not image_filename:
                            continue

                        image_path = os.path.join(current_dir, image_filename)
                        if os.path.exists(image_path):
                            shutil.copy(
                                image_path,
                                os.path.join(temp_images_dir, image_filename),
                            )

                        messages, images = [], []
                        for msg in data.get("chat_history", []):
                            message = {
                                "role": msg.get("role"),
                                "content": msg.get("content"),
                            }
                            messages.append(message)

                            # Track images
                            if msg.get("image"):
                                rel_path = f"images/{image_filename}"
                                if rel_path not in images:
                                    images.append(rel_path)

                        if messages:
                            export_data.append(
                                {"messages": messages, "images": images}
                            )

                    except Exception as e:
                        logger.error(
                            self.tr(f"Error processing {json_file}: {str(e)}")
                        )

                if not export_data:
                    QMessageBox.warning(
                        self,
                        self.tr("Export Error"),
                        self.tr("No valid chat data found to export."),
                    )
                    return

                dataset_json_path = os.path.join(
                    temp_dir, f"{export_filename}.json"
                )
                with open(dataset_json_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                # Create zip file
                with zipfile.ZipFile(
                    zip_path, "w", zipfile.ZIP_DEFLATED
                ) as zipf:
                    zipf.write(
                        dataset_json_path, arcname=f"{export_filename}.json"
                    )
                    for img in os.listdir(temp_images_dir):
                        img_path = os.path.join(temp_images_dir, img)
                        zipf.write(img_path, arcname=f"images/{img}")

                shutil.rmtree(temp_dir)
                QMessageBox.information(
                    self,
                    self.tr("Export Successful"),
                    self.tr(f"Dataset exported successfully to:\n{zip_path}"),
                )

            except Exception as e:
                logger.error(
                    self.tr(f"An error occurred during export:\n{str(e)}")
                )
                QMessageBox.critical(
                    self,
                    self.tr("Export Error"),
                    self.tr(f"An error occurred during export:\n{str(e)}"),
                )
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        def import_dataset():
            option_dialog.accept()

            if not self.parent().filename:
                QMessageBox.warning(
                    self,
                    self.tr("Import Error"),
                    self.tr("No file is currently open."),
                )
                return

            current_dir = os.path.dirname(self.parent().filename)
            import_file, _ = QFileDialog.getOpenFileName(
                self,
                self.tr("Select Dataset File"),
                current_dir,
                "JSON Files (*.json)",
            )

            if not import_file:
                return

            try:
                with open(import_file, "r", encoding="utf-8") as f:
                    import_data = json.load(f)

                if not isinstance(import_data, list):
                    QMessageBox.warning(
                        self,
                        self.tr("Import Error"),
                        self.tr(
                            "Invalid dataset format. Expected a list of records."
                        ),
                    )
                    return

                import_base_dir = os.path.dirname(import_file)
                imported_count = 0
                for item in import_data:
                    messages = item.get("messages", [])
                    images = item.get("images", [])

                    if not messages or not images:
                        continue

                    for image_path in images:
                        image_filename = os.path.basename(image_path)

                        # Check if image exists locally or in the import directory
                        local_image_path = os.path.join(
                            current_dir, image_filename
                        )
                        import_image_path = os.path.join(
                            import_base_dir, image_path
                        )
                        import_image_dir_path = os.path.join(
                            import_base_dir,
                            os.path.dirname(image_path),
                            image_filename,
                        )

                        # Find the actual image file
                        found_image = False
                        for img_path in [
                            local_image_path,
                            import_image_path,
                            import_image_dir_path,
                        ]:
                            if os.path.exists(img_path):
                                if img_path != local_image_path:
                                    shutil.copy(img_path, local_image_path)
                                found_image = True
                                break

                        if not found_image:
                            continue

                        # Add "chatbot_" prefix to the JSON filename
                        base_filename = os.path.splitext(image_filename)[0]
                        json_filename = f"chatbot_{base_filename}.json"
                        json_path = os.path.join(current_dir, json_filename)

                        # Prepare chat history
                        chat_history = []
                        for msg in messages:
                            chat_entry = {
                                "role": msg.get("role", ""),
                                "content": msg.get("content", ""),
                                "image": None,
                            }

                            # Set image path for user messages that contain <image> tag
                            if (
                                chat_entry["role"] == "user"
                                and "<image>" in chat_entry["content"]
                            ):
                                chat_entry["image"] = local_image_path

                            chat_history.append(chat_entry)

                        # Parse image data
                        width, height = 0, 0
                        with Image.open(local_image_path) as img:
                            width, height = img.size

                        json_content = {
                            "version": __version__,
                            "flags": {},
                            "shapes": [],
                            "imagePath": image_filename,
                            "imageData": None,
                            "imageHeight": height,
                            "imageWidth": width,
                            "chat_history": chat_history,
                            "description": "",
                        }

                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(
                                json_content, f, ensure_ascii=False, indent=2
                            )

                        imported_count += 1
                        break  # (NOTE) Only support one image per file

                if imported_count > 0:
                    template = self.tr(
                        "Successfully imported {0} items to:\n{1}"
                    )
                    message_text = template.format(imported_count, current_dir)
                    QMessageBox.information(
                        self, self.tr("Import Successful"), message_text
                    )
                    self.navigate_image(
                        index=self.parent().image_list.index(
                            self.parent().filename
                        )
                    )
                else:
                    QMessageBox.warning(
                        self,
                        self.tr("Import Notice"),
                        self.tr(
                            "No valid items were found to import. Make sure images are available."
                        ),
                    )

            except Exception as e:
                logger.error(
                    self.tr(f"An error occurred during import:\n{str(e)}")
                )
                QMessageBox.critical(
                    self,
                    self.tr("Import Error"),
                    self.tr(f"An error occurred during import:\n{str(e)}"),
                )

        import_btn.clicked.connect(import_dataset)
        export_btn.clicked.connect(export_dataset)

        option_dialog.exec_()

    def eventFilter(self, obj, event):
        """Event filter for handling events"""
        # Tooltip handler for multiple buttons
        tooltip_buttons = {
            "temperature_btn": self.temperature_tooltip,
            "clear_chat_btn": self.clear_chat_tooltip,
            "open_image_file_btn": self.open_image_file_tooltip,
            "open_image_folder_btn": self.open_image_folder_tooltip,
            "prev_image_btn": self.prev_image_tooltip,
            "next_image_btn": self.next_image_tooltip,
            "run_all_images_btn": self.run_all_images_tooltip,
            "import_export_btn": self.import_export_tooltip,
        }
        for btn_name, tooltip in tooltip_buttons.items():
            if obj.objectName() == btn_name:
                if event.type() == QEvent.Enter:
                    button_pos = obj.mapToGlobal(QPoint(0, 0))
                    tooltip.move(button_pos)
                    tooltip.adjustSize()
                    tooltip_width = tooltip.width()
                    tooltip_height = tooltip.height()
                    if btn_name in ["temperature_btn"]:
                        target_x = button_pos.x() - tooltip_width + 5
                        target_y = button_pos.y() - tooltip_height - 5
                    else:
                        target_x = button_pos.x() - tooltip_width // 2 + 10
                        target_y = button_pos.y() - tooltip_height - 5
                    tooltip.move(target_x, target_y)
                    tooltip.show()
                    return True
                elif (
                    event.type() == QEvent.Leave
                    or event.type() == QEvent.Wheel
                ):
                    tooltip.hide()
                    return True

        if obj == self.message_input and event.type() == event.KeyPress:
            if (
                event.key() == Qt.Key_Return
                and event.modifiers() & Qt.ControlModifier
            ):
                self.start_generation()
                return True
            elif (
                event.key() == Qt.Key_Return
                and not event.modifiers() & Qt.ControlModifier
            ):
                # Enter without Ctrl adds a new line
                return False
        # Prevent Enter key from triggering buttons when in settings fields
        elif (
            hasattr(self, "api_address")
            and hasattr(self, "model_button")
            and hasattr(self, "api_key")
        ):
            if (
                obj in [self.api_address, self.model_button, self.api_key]
                and event.type() == event.KeyPress
            ):
                if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                    return True
        return super().eventFilter(obj, event)

    def stream_generation(self, api_address, api_key):
        """Generate streaming response from the API"""
        try:
            self.stream_handler.start_loading()
            logger.debug(
                f"Invoking model {self.selected_model}({self.default_provider}) with base URL {api_address}"
            )

            # Get temperature value from slider
            temperature = self.temp_slider.value() / 10.0

            # Get system prompt if provided
            system_prompt = self.system_prompt_input.text().strip()

            # Get max tokens if provided
            max_tokens = None
            if (
                hasattr(self, "max_length_input")
                and self.max_length_input.text().strip()
            ):
                try:
                    max_tokens = int(self.max_length_input.text().strip())
                except ValueError:
                    pass

            # Prepare messages
            messages = []

            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            for msg in self.chat_history:
                if msg["image"]:
                    try:
                        with open(msg["image"], "rb") as image_file:
                            image_data = base64.b64encode(
                                image_file.read()
                            ).decode("utf-8")
                            messages.append(
                                {
                                    "role": msg["role"],
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{image_data}"
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": msg["content"],
                                        },
                                    ],
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error reading image file: {e}")
                else:
                    messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            api_params = {
                "model": self.selected_model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens:
                api_params["max_tokens"] = max_tokens

            # Create client and prepare API call parameters
            client = OpenAI(base_url=api_address, api_key=api_key, timeout=10)

            # Create a secondary thread to periodically check for cancellation
            stop_event = threading.Event()

            def check_for_cancellation():
                while not stop_event.is_set():
                    if self.stream_handler.stop_requested:
                        stop_event.set()
                        if not self.stream_handler.get_current_message():
                            self.stream_handler.report_error(
                                "Request cancelled by user"
                            )
                        self.stream_handler.finished.emit(False)
                        self.stream_handler.stop_loading()
                        self.restore_send_button()
                        break
                    time.sleep(0.1)

            cancel_thread = threading.Thread(target=check_for_cancellation)
            cancel_thread.daemon = True
            cancel_thread.start()

            response = client.chat.completions.create(**api_params)

            # Process streaming response
            for chunk in response:
                if self.stream_handler.stop_requested:
                    break

                if (
                    hasattr(chunk.choices[0].delta, "content")
                    and chunk.choices[0].delta.content
                ):
                    content = chunk.choices[0].delta.content
                    self.stream_handler.append_text(content)

            logger.debug(f"User\n{messages[-1]['content']}")
            logger.debug(
                f"Assistant\n{self.stream_handler.get_current_message()}\n"
            )

            self.stream_handler.finished.emit(True)
            self.stream_handler.stop_loading()
            self.restore_send_button()

        except Exception as e:
            logger.debug(f"Error in streaming generation: {e}")
            self.stream_handler.report_error(str(e))
            self.stream_handler.finished.emit(False)

    def set_components_enabled(self, enabled):
        """Enable or disable UI components during streaming"""
        self.message_input.setEnabled(enabled)
        self.prev_image_btn.setEnabled(enabled)
        self.next_image_btn.setEnabled(enabled)
        self.open_image_file_btn.setEnabled(enabled)
        self.open_image_folder_btn.setEnabled(enabled)
        self.run_all_images_btn.setEnabled(enabled)
        self.import_export_btn.setEnabled(enabled)
        self.clear_chat_btn.setEnabled(enabled)

        # Update cursor for input
        if enabled:
            self.message_input.setCursor(Qt.IBeamCursor)
        else:
            self.message_input.setCursor(Qt.ForbiddenCursor)

        # Update chat message buttons
        self.set_chat_message_buttons_enabled(enabled)

    def set_chat_message_buttons_enabled(self, enabled):
        """Enable or disable all buttons in chat messages"""
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if (
                item
                and item.widget()
                and isinstance(item.widget(), ChatMessage)
            ):
                message_widget = item.widget()
                if hasattr(message_widget, "set_action_buttons_enabled"):
                    message_widget.set_action_buttons_enabled(enabled)

    def handle_error(self, error_message):
        """Handle error messages from the streaming thread"""
        # Stop the loading timer if it's still active
        if self.loading_timer and self.loading_timer.isActive():
            self.loading_timer.stop()

        if self.loading_message:
            # Remove the loading message
            self.loading_message.setParent(None)
            self.loading_message.deleteLater()
            self.loading_message = None

            # Add error message but don't add it to chat history
            self.add_message(
                "assistant",
                f"Error: {error_message}",
                delete_last_message=True,
            )

    def resizeEvent(self, event):
        """Handle window resize event, update message layout constraints"""
        super().resizeEvent(event)

        # Initialize flag if not present
        if not hasattr(self, "_middle_is_centered"):
            self._middle_is_centered = False

        is_maximized = self.isMaximized()

        if is_maximized and not self._middle_is_centered:
            while self.middle_wrapper_layout.count():
                item = self.middle_wrapper_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)

            # Add centered layout: stretch, widget, stretch
            self.middle_wrapper_layout.addStretch(1)
            self.middle_wrapper_layout.addWidget(self.middle_widget)
            self.middle_wrapper_layout.addStretch(1)
            self.middle_widget.setFixedWidth(700)
            self._middle_is_centered = True
            self.main_splitter.setStretchFactor(1, 1)

        elif not is_maximized and self._middle_is_centered:
            while self.middle_wrapper_layout.count():
                item = self.middle_wrapper_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)

            self.middle_wrapper_layout.addWidget(self.middle_widget)
            self.middle_widget.setMinimumWidth(0)
            self.middle_widget.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            self._middle_is_centered = False
            self.main_splitter.setStretchFactor(1, 1)

        # Always update chat message constraints based on middle_widget's current width
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, "update_width_constraint"):
                    widget.update_width_constraint()

        # Update geometries and scroll if needed
        self.chat_container.updateGeometry()
        QApplication.processEvents()
        QTimer.singleShot(100, self.scroll_to_bottom)

    def resubmit_edited_message(self, message_widget, new_content):
        """Handle resubmission of an edited message"""
        # Find the index of the message in the chat history
        message_widgets = []
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if (
                item
                and item.widget()
                and isinstance(item.widget(), ChatMessage)
            ):
                message_widgets.append(item.widget())

        if message_widget in message_widgets:
            message_index = message_widgets.index(message_widget)

            # Clear all messages after and including this one
            self.clear_messages_after(message_index)

            # Set the edited text to the message input
            self.message_input.setPlainText(new_content)

            # Trigger generation with the new content
            self.start_generation()

    def clear_messages_after(self, index):
        """Clear all messages after and including the specified index"""
        message_widgets = []
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if (
                item
                and item.widget()
                and isinstance(item.widget(), ChatMessage)
            ):
                message_widgets.append((i, item.widget()))

        # Identify which items to remove (in reverse order to avoid index issues)
        to_remove = []
        for i, (layout_index, widget) in enumerate(message_widgets):
            if i >= index:
                to_remove.append((layout_index, widget))

        # Remove widgets in reverse order
        for layout_index, widget in reversed(to_remove):
            self.chat_messages_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

        # Update chat history to match
        if len(to_remove) > 0:
            self.chat_history = self.chat_history[:index]

        # Make sure we still have a stretch at the end
        has_stretch = False
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if item and item.spacerItem():
                has_stretch = True
                break
        if not has_stretch:
            self.chat_messages_layout.addStretch()

        # Force update layout
        QApplication.processEvents()

    def regenerate_response(self, message_widget):
        """Regenerate the assistant's response"""
        if self.streaming:
            return

        message_widgets = []
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if (
                item
                and item.widget()
                and isinstance(item.widget(), ChatMessage)
            ):
                message_widgets.append((i, item.widget()))

        # Find the index of the message in the list
        message_index = None
        for i, (_, widget) in enumerate(message_widgets):
            if widget == message_widget:
                message_index = i
                break

        if message_index is not None:
            self.clear_messages_after(message_index)

            # Start generation process
            self.streaming = True
            self.set_components_enabled(False)
            self.add_loading_message()

            # Start streaming in a separate thread
            self.stream_handler.reset()
            self.stream_thread = threading.Thread(
                target=self.stream_generation,
                args=(self.current_api_address, self.current_api_key),
            )
            self.stream_thread.start()

    def on_text_changed(self):
        """Handle text changes in the message input, resize and highlight @image tag"""
        self.resize_input()

        # Update send button state based on whether input is empty
        current_text = self.message_input.toPlainText().strip()
        self.send_btn.setEnabled(bool(current_text))

        # Highlight @image tag
        cursor = self.message_input.textCursor()
        current_position = cursor.position()
        document = self.message_input.document()
        text = document.toPlainText()

        # Block signals temporarily to prevent recursive calls
        self.message_input.blockSignals(True)

        # Reset formatting
        cursor.select(QTextCursor.Document)
        format = QTextCharFormat()
        cursor.setCharFormat(format)

        # Find and highlight @image
        tag = "@image"
        start_index = text.find(tag)
        if start_index != -1:
            # Set highlight format
            highlight_format = QTextCharFormat()
            highlight_format.setBackground(
                QColor("#E3F2FD")
            )  # Light blue background
            highlight_format.setForeground(
                QColor("#1976D2")
            )  # Darker blue text

            # Select and format the tag
            cursor.setPosition(start_index)
            cursor.setPosition(start_index + len(tag), QTextCursor.KeepAnchor)
            cursor.setCharFormat(highlight_format)

            # Restore cursor position
            cursor.setPosition(current_position)
            self.message_input.setTextCursor(cursor)

        # Unblock signals
        self.message_input.blockSignals(False)

    def clear_conversation(self):
        """Clear all chat messages and reset history"""
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setText(
            self.tr("Are you sure you want to clear the entire conversation?")
        )
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_dialog.setDefaultButton(QMessageBox.No)
        confirm_dialog.setIcon(QMessageBox.Warning)

        # Show dialog and handle response
        response = confirm_dialog.exec_()
        if response == QMessageBox.Yes:
            while self.chat_messages_layout.count() > 0:
                item = self.chat_messages_layout.takeAt(0)
                if item and item.widget():
                    widget = item.widget()
                    widget.setParent(None)
                    widget.deleteLater()
                elif item:
                    self.chat_messages_layout.removeItem(item)

            # Reset chat history and add stretch
            self.chat_history = []
            self.chat_messages_layout.addStretch()

            # Update parent data if applicable
            if self.parent().filename:
                self.parent().other_data["chat_history"] = []
                self.parent().set_dirty()

    def hideAllTooltips(self):
        """Hide all tooltips as a safety measure"""
        tooltips = [
            self.temperature_tooltip,
            self.clear_chat_tooltip,
            self.open_image_file_tooltip,
            self.open_image_folder_tooltip,
            self.prev_image_tooltip,
            self.next_image_tooltip,
        ]
        [tooltip.hide() for tooltip in tooltips if tooltip]

        # Find and hide all tooltips in chat messages
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if (
                item
                and item.widget()
                and isinstance(item.widget(), ChatMessage)
            ):
                message = item.widget()
                for tooltip_name in [
                    "copy_tooltip",
                    "delete_tooltip",
                    "edit_tooltip",
                    "regenerate_tooltip",
                ]:
                    if hasattr(message, tooltip_name):
                        tooltip = getattr(message, tooltip_name)
                        if tooltip:
                            tooltip.hide()

    def closeEvent(self, event):
        """Handle dialog close event properly"""
        self.hideAllTooltips()

        # Stop any ongoing timers
        if self.loading_timer and self.loading_timer.isActive():
            self.loading_timer.stop()

        super().closeEvent(event)

    def wheelEvent(self, event):
        """Handle wheel events at dialog level"""
        self.hideAllTooltips()
        super().wheelEvent(event)

    def show_message_input_context_menu(self, position):
        """Show a custom styled context menu for the message input."""
        menu = self.message_input.createStandardContextMenu()
        menu.setStyleSheet(ChatbotDialogStyle.get_menu_style())
        menu.exec_(self.message_input.mapToGlobal(position))
