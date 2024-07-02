from napari.types import FullLayerData
from napari.utils.transforms import Affine
from pathlib import Path
from typing import List


def write_transformed_layers(path: str, layer_data: List[FullLayerData]) -> List[str]:
    print("The napari_save_transformed plugin is running.")
    print(f"Processing {len(layer_data)} layers.")
    for data, attrs, type in layer_data:
        print(f"Current layer: {attrs['name']}, type: {type}")
    return path
