"""
Reader functions for napari drag-and-drop functionality
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from napari.types import LayerDataTuple


def napari_get_reader(path: Union[str, List[str]]) -> Optional[Callable]:
    """
    A basic implementation of the napari get reader plugin hook specification.
    
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
        
    Returns
    -------
    function or None
        If the path is a supported format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # If path is a list, only handle single file for now
        if len(path) != 1:
            return None
        path = path[0]
    
    # Convert to Path object
    path = Path(path)
    
    # Check if it's a JSON file that might contain mask data
    if path.suffix.lower() != '.json':
        return None
        
    # Check if filename suggests it contains mask data
    filename_lower = path.name.lower()
    if any(keyword in filename_lower for keyword in ['mask', 'usg', 'label', 'annotation']):
        return read_mask_data
    
    # Try to read and check if it has mask data structure
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Check if it looks like our mask data format
        if isinstance(data, dict) and 'masks' in data and 'labels_config' in data:
            return read_mask_data
            
    except (json.JSONDecodeError, IOError):
        pass
    
    return None


def read_mask_data(path: Union[str, List[str]]) -> List[LayerDataTuple]:
    """
    Read mask data from JSON file and return napari layer data
    
    Parameters
    ----------
    path : str or list of str
        Path to the JSON file containing mask data
        
    Returns
    -------
    List[LayerDataTuple]
        List of layer data tuples for napari
    """
    if isinstance(path, list):
        path = path[0]
    
    path = Path(path)
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        if 'masks' not in data:
            return []
            
        # Load labels configuration
        labels_config = data.get('labels_config', {})
        video_info = data.get('video_info', {})
        
        # Create colormap from labels config
        colormap = {}
        for label_name, config in labels_config.items():
            if 'value' in config and 'color' in config:
                value = config['value']
                color = [c/255.0 for c in config['color'][:3]]  # Normalize to 0-1
                colormap[value] = color
        
        # Load all masks
        masks_dict = data['masks']
        if not masks_dict:
            return []
            
        # Sort frame indices
        frame_indices = sorted([int(k) for k in masks_dict.keys()])
        
        # Determine if we should create a 3D array or individual 2D layers
        if len(frame_indices) == 1:
            # Single frame - create 2D layer
            frame_idx = frame_indices[0]
            mask_2d = np.array(masks_dict[str(frame_idx)], dtype=np.uint16)
            
            metadata = {
                'labels_config': labels_config,
                'frame_index': frame_idx,
                'source_file': str(path),
                'video_info': video_info
            }
            
            layer_data = (
                mask_2d,
                {
                    'name': f'USG Mask Frame {frame_idx} ({path.stem})',
                    'colormap': colormap,
                    'metadata': metadata,
                    'opacity': 0.7
                },
                'labels'
            )
            
            return [layer_data]
            
        else:
            # Multiple frames - create 3D array
            first_mask = np.array(masks_dict[str(frame_indices[0])], dtype=np.uint16)
            mask_shape = first_mask.shape
            
            # Create 3D array with zeros for missing frames
            max_frame = max(frame_indices)
            masks_3d = np.zeros((max_frame + 1, *mask_shape), dtype=np.uint16)
            
            for frame_idx in frame_indices:
                masks_3d[frame_idx] = np.array(masks_dict[str(frame_idx)], dtype=np.uint16)
            
            metadata = {
                'labels_config': labels_config,
                'frame_indices': frame_indices,
                'source_file': str(path),
                'video_info': video_info
            }
            
            layer_data = (
                masks_3d,
                {
                    'name': f'USG Masks ({len(frame_indices)} frames) ({path.stem})',
                    'colormap': colormap,
                    'metadata': metadata,
                    'opacity': 0.7
                },
                'labels'
            )
            
            return [layer_data]
        
    except Exception as e:
        print(f"Error loading mask data from {path}: {e}")
        return []


def save_mask_data(
    masks: Dict[int, np.ndarray], 
    labels_config: Dict[str, Any],
    output_path: str,
    video_info: Optional[Dict] = None
) -> bool:
    """
    Save mask data to JSON file
    
    Parameters
    ----------
    masks : Dict[int, np.ndarray]
        Dictionary mapping frame indices to mask arrays
    labels_config : Dict[str, Any]
        Label configuration dictionary
    output_path : str
        Output file path
    video_info : Optional[Dict]
        Optional video metadata
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Prepare data for saving
        save_data = {
            'masks': {},
            'labels_config': labels_config,
            'video_info': video_info or {},
            'metadata': {
                'plugin_version': '0.1.0',
                'creation_date': str(np.datetime64('now')),
                'total_masked_frames': len(masks)
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for frame_idx, mask in masks.items():
            save_data['masks'][str(frame_idx)] = mask.tolist()
            
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error saving mask data: {e}")
        return False


def export_masks_as_images(
    masks: Dict[int, np.ndarray],
    labels_config: Dict[str, Any],
    output_dir: str,
    prefix: str = "usg_mask_frame"
) -> bool:
    """
    Export masks as colored PNG images
    
    Parameters
    ----------
    masks : Dict[int, np.ndarray]
        Dictionary mapping frame indices to mask arrays
    labels_config : Dict[str, Any]
        Label configuration dictionary
    output_dir : str
        Output directory path
    prefix : str
        Filename prefix
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        import imageio
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for frame_idx, mask in masks.items():
            # Create colored mask
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            
            for label_name, config in labels_config.items():
                if 'value' in config and 'color' in config:
                    value = config['value']
                    color = config['color'][:3]
                    mask_indices = mask == value
                    colored_mask[mask_indices] = color
                
            # Save image
            filename = output_path / f"{prefix}_{frame_idx:04d}.png"
            imageio.imwrite(str(filename), colored_mask)
            
        return True
        
    except Exception as e:
        print(f"Error exporting mask images: {e}")
        return False