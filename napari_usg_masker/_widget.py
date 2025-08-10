"""
Main widget for USG masking functionality
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path

try:
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QComboBox, QSpinBox, 
        QSlider, QColorDialog, QFileDialog, QMessageBox,
        QListWidget, QListWidgetItem, QGroupBox,
        QCheckBox, QLineEdit, QTextEdit, QProgressBar
    )
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtGui import QColor, QPalette
except ImportError:
    # Fallback for PyQt5 issues on Apple Silicon
    try:
        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
            QPushButton, QLabel, QComboBox, QSpinBox,
            QSlider, QColorDialog, QFileDialog, QMessageBox,
            QListWidget, QListWidgetItem, QGroupBox,
            QCheckBox, QLineEdit, QTextEdit, QProgressBar
        )
        from PyQt5.QtCore import Qt, pyqtSignal as Signal
        from PyQt5.QtGui import QColor, QPalette
    except ImportError:
        raise ImportError("Could not import Qt widgets. Please install PyQt5 or PySide2.")

import napari
from napari.layers import Labels, Image


class USGMaskerWidget(QWidget):
    """
    A comprehensive widget for USG video masking with predefined labels
    """
    
    # Define predefined label categories with colors
    DEFAULT_LABELS = {
        'background': {'value': 0, 'color': [0, 0, 0, 255]},  # Black
        'metastatic_lymph_node': {'value': 1, 'color': [255, 0, 0, 255]},  # Red
        'clean_lymph_node': {'value': 2, 'color': [0, 255, 0, 255]},  # Green
        # '': {'value': 3, 'color': [0, 0, 255, 255]},  # Blue
        # '': {'value': 4, 'color': [255, 255, 0, 255]},  # Yellow
        # '': {'value': 5, 'color': [255, 0, 255, 255]},  # Magenta
        # '': {'value': 6, 'color': [0, 255, 255, 255]},  # Cyan
        # '': {'value': 7, 'color': [255, 128, 0, 255]},  # Orange
        # '': {'value': 8, 'color': [128, 255, 128, 255]},  # Light green
        # '': {'value': 9, 'color': [255, 128, 128, 255]},  # Light red
        # '': {'value': 10, 'color': [128, 128, 128, 255]},  # Gray
        # '': {'value': 11, 'color': [64, 64, 64, 255]},  # Dark gray
    }
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_video = None
        self.current_video_path: Optional[str] = None
        self.current_masks = {}  # {frame_index: mask_array}
        self.current_frame = 0
        self.total_frames = 0
        self.labels_config = self.DEFAULT_LABELS.copy()
        self.current_label = 1
        self.brush_size = 10
        self.mask_layer = None
        self.video_layer = None
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🔬 USG Video Masker")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px; color: #2E86C1;")
        layout.addWidget(title)
        
        # Video controls group
        video_group = QGroupBox("📹 Video Controls")
        video_layout = QVBoxLayout()
        
        # Load video button
        self.load_video_btn = QPushButton("📁 Load USG Video")
        self.load_video_btn.setStyleSheet("QPushButton { background-color: #3498DB; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }")
        self.load_video_btn.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_video_btn)
        
        # Video info label
        self.video_info_label = QLabel("No video loaded")
        self.video_info_label.setStyleSheet("color: #7F8C8D; font-style: italic;")
        video_layout.addWidget(self.video_info_label)
        
        # Frame navigation
        frame_nav_layout = QHBoxLayout()
        frame_nav_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.change_frame)
        frame_nav_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0/0")
        self.frame_label.setStyleSheet("font-weight: bold;")
        frame_nav_layout.addWidget(self.frame_label)
        
        video_layout.addLayout(frame_nav_layout)
        
        # Frame navigation buttons
        nav_buttons_layout = QHBoxLayout()
        self.prev_frame_btn = QPushButton("⬅️ Previous")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn = QPushButton("➡️ Next")
        self.next_frame_btn.clicked.connect(self.next_frame)
        
        nav_buttons_layout.addWidget(self.prev_frame_btn)
        nav_buttons_layout.addWidget(self.next_frame_btn)
        video_layout.addLayout(nav_buttons_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Masking tools group
        mask_group = QGroupBox("🎨 Masking Tools")
        mask_layout = QVBoxLayout()
        
        # Label selection
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Current Label:"))
        self.label_combo = QComboBox()
        self.update_label_combo()
        self.label_combo.currentTextChanged.connect(self.change_current_label)
        label_layout.addWidget(self.label_combo)
        mask_layout.addLayout(label_layout)
        
        # Brush size
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Brush Size:"))
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setMinimum(1)
        self.brush_size_spinbox.setMaximum(50)
        self.brush_size_spinbox.setValue(self.brush_size)
        self.brush_size_spinbox.valueChanged.connect(self.change_brush_size)
        brush_layout.addWidget(self.brush_size_spinbox)
        
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(self.change_brush_size_slider)
        brush_layout.addWidget(self.brush_size_slider)
        mask_layout.addLayout(brush_layout)
        
        # Mask manipulation buttons
        mask_buttons_layout = QGridLayout()
        
        self.create_mask_btn = QPushButton("✨ Create New Mask")
        self.create_mask_btn.setStyleSheet("QPushButton { background-color: #27AE60; color: white; padding: 6px; border-radius: 3px; }")
        self.create_mask_btn.clicked.connect(self.create_new_mask)
        mask_buttons_layout.addWidget(self.create_mask_btn, 0, 0)
        
        self.copy_prev_btn = QPushButton("📋 Copy from Previous")
        self.copy_prev_btn.clicked.connect(self.copy_from_previous)
        mask_buttons_layout.addWidget(self.copy_prev_btn, 0, 1)
        
        self.copy_next_btn = QPushButton("📄 Copy to Next")
        self.copy_next_btn.clicked.connect(self.copy_to_next)
        mask_buttons_layout.addWidget(self.copy_next_btn, 1, 0)
        
        self.clear_mask_btn = QPushButton("🗑️ Clear Current")
        self.clear_mask_btn.setStyleSheet("QPushButton { background-color: #E74C3C; color: white; padding: 6px; border-radius: 3px; }")
        self.clear_mask_btn.clicked.connect(self.clear_current_mask)
        mask_buttons_layout.addWidget(self.clear_mask_btn, 1, 1)
        
        mask_layout.addLayout(mask_buttons_layout)
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)
        
        # Label management group
        labels_group = QGroupBox("🏷️ Label Management")
        labels_layout = QVBoxLayout()
        
        # Label list
        self.labels_list = QListWidget()
        self.labels_list.setMaximumHeight(150)
        self.update_labels_list()
        labels_layout.addWidget(self.labels_list)
        
        # Label management buttons
        label_mgmt_layout = QHBoxLayout()
        self.add_label_btn = QPushButton("➕ Add Label")
        self.add_label_btn.clicked.connect(self.add_custom_label)
        self.edit_color_btn = QPushButton("🎨 Edit Color")
        self.edit_color_btn.clicked.connect(self.edit_label_color)
        
        label_mgmt_layout.addWidget(self.add_label_btn)
        label_mgmt_layout.addWidget(self.edit_color_btn)
        labels_layout.addLayout(label_mgmt_layout)
        
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)
        
        # Data management group
        data_group = QGroupBox("💾 Data Management")
        data_layout = QVBoxLayout()
        
        # Save/Load masks
        data_buttons_layout = QGridLayout()
        
        self.save_masks_btn = QPushButton("💾 Save All Masks")
        self.save_masks_btn.setStyleSheet("QPushButton { background-color: #8E44AD; color: white; padding: 6px; border-radius: 3px; }")
        self.save_masks_btn.clicked.connect(self.save_masks)
        data_buttons_layout.addWidget(self.save_masks_btn, 0, 0)
        
        self.load_masks_btn = QPushButton("📂 Load Masks")
        self.load_masks_btn.clicked.connect(self.load_masks)
        data_buttons_layout.addWidget(self.load_masks_btn, 0, 1)
        
        self.export_masks_btn = QPushButton("🖼️ Export as Images")
        self.export_masks_btn.clicked.connect(self.export_masks_as_images)
        data_buttons_layout.addWidget(self.export_masks_btn, 1, 0)

        # New: Export frames and masks as PNG + NPY
        self.export_full_btn = QPushButton("📦 Export Frames & Masks (PNG+NPY)")
        self.export_full_btn.clicked.connect(self.export_frames_and_masks_png_npy)
        data_buttons_layout.addWidget(self.export_full_btn, 2, 0)
        
        self.load_config_btn = QPushButton("⚙️ Load Config")
        self.load_config_btn.clicked.connect(self.load_label_config)
        data_buttons_layout.addWidget(self.load_config_btn, 2, 1)
        
        data_layout.addLayout(data_buttons_layout)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(100)
        self.stats_text.setReadOnly(True)
        layout.addWidget(QLabel("📊 Masking Statistics:"))
        layout.addWidget(self.stats_text)
        
        self.setLayout(layout)
        
    def connect_signals(self):
        """Connect viewer signals"""
        # Listen for layer changes
        self.viewer.layers.events.inserted.connect(self.on_layer_inserted)
        self.viewer.layers.events.removed.connect(self.on_layer_removed)
        
    def update_label_combo(self):
        """Update the label selection combo box"""
        self.label_combo.clear()
        for label_name in self.labels_config.keys():
            if label_name != 'background':  # Skip background for selection
                self.label_combo.addItem(label_name)
                
    def update_labels_list(self):
        """Update the labels list widget"""
        self.labels_list.clear()
        for label_name, config in self.labels_config.items():
            item = QListWidgetItem(f"{label_name} (value: {config['value']})")
            color = QColor(*config['color'])
            item.setBackground(color)
            # Set text color based on background brightness
            if sum(config['color'][:3]) < 384:  # Dark background
                item.setForeground(QColor(255, 255, 255))
            else:  # Light background
                item.setForeground(QColor(0, 0, 0))
            self.labels_list.addItem(item)
            
    def change_current_label(self, label_name):
        """Change the current painting label"""
        if label_name in self.labels_config:
            self.current_label = self.labels_config[label_name]['value']
            if self.mask_layer:
                self.mask_layer.selected_label = self.current_label
                
    def change_brush_size(self, size):
        """Change brush size from spinbox"""
        self.brush_size = size
        self.brush_size_slider.setValue(size)
        if self.mask_layer:
            self.mask_layer.brush_size = size
            
    def change_brush_size_slider(self, size):
        """Change brush size from slider"""
        self.brush_size = size
        self.brush_size_spinbox.setValue(size)
        if self.mask_layer:
            self.mask_layer.brush_size = size
            
    def load_video(self):
        """Load USG video file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 
            "Load USG Video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.mts *.m4v);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate progress
                
                # Save current mask before loading new video
                if self.mask_layer and hasattr(self.mask_layer, 'data') and self.current_video is not None:
                    self.current_masks[self.current_frame] = self.mask_layer.data.copy()
                
                # Try to load video using imageio
                import imageio
                
                # Read video
                reader = imageio.get_reader(file_path)
                video_data = []
                
                # Get video info
                meta = reader.get_meta_data()
                fps = meta.get('fps', 30)
                
                for i, frame in enumerate(reader):
                    if i % 10 == 0:  # Update every 10 frames
                        self.video_info_label.setText(f"Loading frame {i}...")
                    
                    # Convert to grayscale if needed
                    if len(frame.shape) == 3:
                        frame = np.mean(frame, axis=2).astype(np.uint8)
                    video_data.append(frame)
                
                video_data = np.array(video_data)
                
                # Remove existing layers
                if self.video_layer and self.video_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.video_layer)
                if self.mask_layer and self.mask_layer in self.viewer.layers:
                    try:
                        self.mask_layer.events.data.disconnect(self.on_mask_changed)
                    except:
                        pass
                    self.viewer.layers.remove(self.mask_layer)
                    
                # Add video to viewer
                self.video_layer = self.viewer.add_image(
                    video_data, 
                    name=f"USG Video ({Path(file_path).name})",
                    colormap='gray',
                    scale=(1, 1, 1)  # Ensure proper scaling
                )
                
                self.current_video = video_data
                self.current_video_path = file_path
                self.total_frames = len(video_data)
                self.current_frame = 0
                self.current_masks = {}  # Reset masks for new video
                self.mask_layer = None
                
                # Update UI
                self.frame_slider.setMaximum(self.total_frames - 1)
                self.frame_slider.setValue(0)
                self.update_frame_display()
                
                # Update video info
                shape = video_data.shape
                self.video_info_label.setText(
                    f"Loaded: {self.total_frames} frames, {shape[1]}x{shape[2]} pixels, {fps:.1f} FPS"
                )
                
                # Initialize empty mask
                self.create_new_mask()
                
                self.progress_bar.setVisible(False)
                QMessageBox.information(self, "Success", f"Loaded video with {self.total_frames} frames")
                
            except Exception as e:
                self.progress_bar.setVisible(False)
                self.video_info_label.setText("Failed to load video")
                QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
                
    def create_new_mask(self):
        """Create a new mask layer for the current frame"""
        if self.current_video is None:
            QMessageBox.warning(self, "Warning", "Please load a video first")
            return
            
        # Get current frame shape
        frame_shape = self.current_video[self.current_frame].shape
        
        # Create empty mask or use existing
        if self.current_frame in self.current_masks:
            mask_data = self.current_masks[self.current_frame].copy()
        else:
            mask_data = np.zeros(frame_shape, dtype=np.uint16)
            self.current_masks[self.current_frame] = mask_data
            
        # Remove existing mask layer
        if self.mask_layer and self.mask_layer in self.viewer.layers:
            # Disconnect events before removing to prevent issues
            try:
                self.mask_layer.events.data.disconnect(self.on_mask_changed)
            except:
                pass  # Event might not be connected
            self.viewer.layers.remove(self.mask_layer)
            
        # Create colormap for labels
        colormap = self.create_colormap()
        
        # Add mask layer
        self.mask_layer = self.viewer.add_labels(
            mask_data,
            name=f"Mask Frame {self.current_frame}",
            colormap=colormap,
            opacity=0.7
        )
        
        # Set painting properties
        self.mask_layer.selected_label = self.current_label
        self.mask_layer.brush_size = self.brush_size
        self.mask_layer.mode = 'paint'
        
        # Connect mask change event
        self.mask_layer.events.data.connect(self.on_mask_changed)
        
        # Synchronize viewer dimensions
        if self.video_layer:
            self.viewer.dims.current_step = (self.current_frame,) + (0,) * (self.video_layer.ndim - 1)
            
        # Update statistics after creating mask
        self.update_statistics()
        
    def create_colormap(self):
        """Create colormap from labels configuration"""
        # Get maximum label value to create proper colormap array
        max_value = max(config['value'] for config in self.labels_config.values())
        
        # Create colormap as array - napari expects this format
        colormap = np.zeros((max_value + 1, 4))  # RGBA format
        
        for label_name, config in self.labels_config.items():
            value = config['value']
            color = config['color']
            # Normalize colors to 0-1 and ensure RGBA format
            if len(color) == 3:
                color = color + [255]  # Add alpha
            colormap[value] = [c/255.0 for c in color]
            
        return colormap
        
    def on_mask_changed(self, event=None):
        """Handle mask data changes"""
        if self.mask_layer and hasattr(self.mask_layer, 'data'):
            # Update stored mask for current frame
            self.current_masks[self.current_frame] = self.mask_layer.data.copy()
            self.update_statistics()
            
    def change_frame(self, frame_index):
        """Change to specific frame"""
        if self.current_video is None:
            return
            
        # Save current mask before switching frames
        if self.mask_layer and hasattr(self.mask_layer, 'data'):
            self.current_masks[self.current_frame] = self.mask_layer.data.copy()
            
        self.current_frame = frame_index
        
        # Update video display by setting viewer dimensions
        if self.video_layer:
            self.viewer.dims.current_step = (frame_index,) + (0,) * (self.video_layer.ndim - 1)
            
        # Update mask display
        self.create_new_mask()
        self.update_frame_display()
        
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            # Save current mask before moving
            if self.mask_layer and hasattr(self.mask_layer, 'data'):
                self.current_masks[self.current_frame] = self.mask_layer.data.copy()
            new_frame = self.current_frame - 1
            self.frame_slider.setValue(new_frame)
            
    def next_frame(self):
        """Go to next frame"""
        if self.current_frame < self.total_frames - 1:
            # Save current mask before moving
            if self.mask_layer and hasattr(self.mask_layer, 'data'):
                self.current_masks[self.current_frame] = self.mask_layer.data.copy()
            new_frame = self.current_frame + 1
            self.frame_slider.setValue(new_frame)
            
    def update_frame_display(self):
        """Update frame counter display"""
        self.frame_label.setText(f"{self.current_frame + 1}/{self.total_frames}")
        
        # Update button states
        self.prev_frame_btn.setEnabled(self.current_frame > 0)
        self.next_frame_btn.setEnabled(self.current_frame < self.total_frames - 1)
        
    def copy_from_previous(self):
        """Copy mask from previous frame"""
        if self.current_frame > 0:
            prev_frame = self.current_frame - 1
            if prev_frame in self.current_masks:
                self.current_masks[self.current_frame] = self.current_masks[prev_frame].copy()
                self.create_new_mask()
                QMessageBox.information(self, "Success", f"Copied mask from frame {prev_frame + 1}")
            else:
                QMessageBox.warning(self, "Warning", "No mask found in previous frame")
        else:
            QMessageBox.warning(self, "Warning", "Already at first frame")
                
    def copy_to_next(self):
        """Copy current mask to next frame"""
        if self.current_frame < self.total_frames - 1:
            next_frame = self.current_frame + 1
            if self.current_frame in self.current_masks:
                self.current_masks[next_frame] = self.current_masks[self.current_frame].copy()
                QMessageBox.information(self, "Success", f"Copied mask to frame {next_frame + 1}")
            else:
                QMessageBox.warning(self, "Warning", "No mask found in current frame")
        else:
            QMessageBox.warning(self, "Warning", "Already at last frame")
                
    def clear_current_mask(self):
        """Clear the current frame's mask"""
        if self.current_video is None:
            return
            
        reply = QMessageBox.question(
            self, 'Clear Mask', 
            'Are you sure you want to clear the current mask?',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            frame_shape = self.current_video[self.current_frame].shape
            self.current_masks[self.current_frame] = np.zeros(frame_shape, dtype=np.uint16)
            self.create_new_mask()
        
    def add_custom_label(self):
        """Add a custom label"""
        from qtpy.QtWidgets import QInputDialog
        
        # Get label name
        name, ok = QInputDialog.getText(self, "Add Label", "Label name:")
        if not ok or not name:
            return
            
        # Check if name already exists
        if name in self.labels_config:
            QMessageBox.warning(self, "Warning", "Label name already exists")
            return
            
        # Get next available value
        max_value = max(config['value'] for config in self.labels_config.values())
        new_value = max_value + 1
        
        # Choose color
        color = QColorDialog.getColor()
        if color.isValid():
            self.labels_config[name] = {
                'value': new_value,
                'color': [color.red(), color.green(), color.blue(), 255]
            }
            self.update_label_combo()
            self.update_labels_list()
            
            # Update mask layer colormap if exists
            if self.mask_layer:
                self.mask_layer.colormap = self.create_colormap()
            
    def edit_label_color(self):
        """Edit color of selected label"""
        current_item = self.labels_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a label to edit")
            return
            
        # Extract label name from item text
        item_text = current_item.text()
        label_name = item_text.split(" (value:")[0]
        
        if label_name not in self.labels_config:
            return
            
        # Choose new color
        current_color = QColor(*self.labels_config[label_name]['color'][:3])
        new_color = QColorDialog.getColor(current_color)
        
        if new_color.isValid():
            self.labels_config[label_name]['color'] = [
                new_color.red(), new_color.green(), new_color.blue(), 255
            ]
            self.update_labels_list()
            
            # Update mask layer colormap if exists
            if self.mask_layer:
                self.mask_layer.colormap = self.create_colormap()
                
    def save_masks(self):
        """Save all masks to file"""
        if not self.current_masks:
            QMessageBox.warning(self, "Warning", "No masks to save")
            return
            
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Masks",
            f"usg_masks_{self.total_frames}frames.json",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Prepare data for saving
                save_data = {
                    'masks': {},
                    'labels_config': self.labels_config,
                    'video_info': {
                        'total_frames': self.total_frames,
                        'frame_shape': self.current_video[0].shape if self.current_video is not None else None,
                        'masked_frames': list(self.current_masks.keys())
                    },
                    'metadata': {
                        'plugin_version': '0.1.0',
                        'creation_date': str(np.datetime64('now'))
                    }
                }
                
                # Convert numpy arrays to lists for JSON serialization
                for frame_idx, mask in self.current_masks.items():
                    save_data['masks'][str(frame_idx)] = mask.tolist()
                    
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                    
                QMessageBox.information(self, "Success", f"Masks saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save masks: {str(e)}")
                
    def load_masks(self):
        """Load masks from file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Load Masks",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    load_data = json.load(f)
                    
                # Load labels configuration
                if 'labels_config' in load_data:
                    self.labels_config = load_data['labels_config']
                    self.update_label_combo()
                    self.update_labels_list()
                    
                # Load masks
                if 'masks' in load_data:
                    self.current_masks = {}
                    for frame_idx_str, mask_list in load_data['masks'].items():
                        frame_idx = int(frame_idx_str)
                        self.current_masks[frame_idx] = np.array(mask_list, dtype=np.uint16)
                        
                # Update display
                if self.mask_layer:
                    self.create_new_mask()
                    
                # Show info about loaded data
                masked_frames = len(self.current_masks)
                total_frames = load_data.get('video_info', {}).get('total_frames', 'unknown')
                QMessageBox.information(
                    self, "Success", 
                    f"Loaded masks from {file_path}\n"
                    f"Masked frames: {masked_frames}/{total_frames}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load masks: {str(e)}")
                
    def export_masks_as_images(self):
        """Export masks as image files"""
        if not self.current_masks:
            QMessageBox.warning(self, "Warning", "No masks to export")
            return
            
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Select Export Folder")
        
        if folder_path:
            try:
                import imageio
                
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, len(self.current_masks))
                
                for i, (frame_idx, mask) in enumerate(self.current_masks.items()):
                    self.progress_bar.setValue(i)
                    
                    # Create colored mask
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    
                    for label_name, config in self.labels_config.items():
                        value = config['value']
                        color = config['color'][:3]
                        mask_indices = mask == value
                        colored_mask[mask_indices] = color
                        
                    # Save image
                    filename = os.path.join(folder_path, f"usg_mask_frame_{frame_idx:04d}.png")
                    imageio.imwrite(filename, colored_mask)
                    
                self.progress_bar.setVisible(False)
                QMessageBox.information(self, "Success", f"Exported {len(self.current_masks)} masks to {folder_path}")
                
            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Failed to export masks: {str(e)}")

    def export_frames_and_masks_png_npy(self):
        """Export each frame (video) and each mask as PNG and NPY files.

        Structure created under chosen folder (defaults to video directory):
        - frames_png/frame_XXXX.png
        - masks_png/mask_raw_frame_XXXX.png (16-bit indices)
        - masks_png/mask_color_frame_XXXX.png (RGB colorized)
        - frames_npy/frame_XXXX.npy
        - masks_npy/mask_frame_XXXX.npy
        Also writes stacks:
        - frames_stack.npy  (shape: T,H,W)
        - masks_stack.npy   (shape: T,H,W) with zeros for missing masks
        - export_metadata.json
        """
        if self.current_video is None:
            QMessageBox.warning(self, "Warning", "Please load a video first")
            return

        # Choose export root; default to video directory if available
        default_dir = str(Path(self.current_video_path).parent) if self.current_video_path else ""
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Select Export Folder", default_dir)
        if not folder_path:
            return

        try:
            import imageio

            export_root = Path(folder_path)
            frames_png_dir = export_root / "frames_png"
            masks_png_dir = export_root / "masks_png"
            frames_npy_dir = export_root / "frames_npy"
            masks_npy_dir = export_root / "masks_npy"

            for d in [frames_png_dir, masks_png_dir, frames_npy_dir, masks_npy_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # Prepare colormap for colored masks
            colormap = {cfg['value']: cfg['color'][:3] for _, cfg in self.labels_config.items()}

            # Initialize stacks
            T = int(self.total_frames)
            H, W = int(self.current_video[0].shape[0]), int(self.current_video[0].shape[1])
            frames_stack = np.zeros((T, H, W), dtype=np.uint8)
            masks_stack = np.zeros((T, H, W), dtype=np.uint16)

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, T)

            for i in range(T):
                self.progress_bar.setValue(i)

                # Save frame PNG and NPY
                frame = self.current_video[i]
                frames_stack[i] = frame
                imageio.imwrite(str(frames_png_dir / f"frame_{i:04d}.png"), frame)
                np.save(str(frames_npy_dir / f"frame_{i:04d}.npy"), frame)

                # Save mask if exists
                if i in self.current_masks:
                    mask = self.current_masks[i].astype(np.uint16)
                    masks_stack[i] = mask

                    # Raw indices as 16-bit PNG
                    imageio.imwrite(str(masks_png_dir / f"mask_raw_frame_{i:04d}.png"), mask)

                    # Colorized PNG
                    colored = np.zeros((H, W, 3), dtype=np.uint8)
                    for value, color in colormap.items():
                        colored[mask == value] = color
                    imageio.imwrite(str(masks_png_dir / f"mask_color_frame_{i:04d}.png"), colored)

                    # NPY per-frame
                    np.save(str(masks_npy_dir / f"mask_frame_{i:04d}.npy"), mask)

            # Save stacks
            np.save(str(export_root / "frames_stack.npy"), frames_stack)
            np.save(str(export_root / "masks_stack.npy"), masks_stack)

            # Write export metadata
            export_meta = {
                "video_path": self.current_video_path,
                "total_frames": self.total_frames,
                "frame_shape": [H, W],
                "masked_frames": sorted([int(k) for k in self.current_masks.keys()]),
                "labels_config": self.labels_config,
            }
            with open(export_root / "export_metadata.json", "w") as f:
                json.dump(export_meta, f, indent=2)

            self.progress_bar.setVisible(False)
            QMessageBox.information(
                self,
                "Success",
                f"Exported frames and masks to {export_root}"
            )

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")
                
    def load_label_config(self):
        """Load label configuration from file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Load Label Configuration",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                    
                if 'labels_config' in config_data:
                    self.labels_config = config_data['labels_config']
                else:
                    self.labels_config = config_data
                    
                self.update_label_combo()
                self.update_labels_list()
                
                # Update mask layer if exists
                if self.mask_layer:
                    self.mask_layer.colormap = self.create_colormap()
                    
                QMessageBox.information(self, "Success", "Label configuration loaded")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")
                
    def update_statistics(self):
        """Update masking statistics"""
        if not self.current_masks:
            self.stats_text.setText("No masks created yet")
            return
            
        stats = []
        total_pixels = 0
        label_counts = {}
        
        for frame_idx, mask in self.current_masks.items():
            total_pixels += mask.size
            unique, counts = np.unique(mask, return_counts=True)
            
            for value, count in zip(unique, counts):
                if value not in label_counts:
                    label_counts[value] = 0
                label_counts[value] += count
                
        stats.append(f"📈 Frames with masks: {len(self.current_masks)}/{self.total_frames}")
        stats.append(f"🎯 Total pixels labeled: {total_pixels:,}")
        
        if total_pixels > 0:
            stats.append("\n🏷️ Label distribution:")
            
            for value, count in sorted(label_counts.items()):
                # Find label name
                label_name = "unknown"
                for name, config in self.labels_config.items():
                    if config['value'] == value:
                        label_name = name
                        break
                        
                percentage = (count / total_pixels) * 100
                stats.append(f"  • {label_name}: {count:,} pixels ({percentage:.1f}%)")
            
        self.stats_text.setText("\n".join(stats))
        
    def on_layer_inserted(self, event):
        """Handle layer insertion events"""
        layer = event.value
        if isinstance(layer, Image) and "video" in layer.name.lower():
            self.video_layer = layer
            
    def on_layer_removed(self, event):
        """Handle layer removal events"""
        layer = event.value
        if layer == self.video_layer:
            self.video_layer = None
        elif layer == self.mask_layer:
            self.mask_layer = None