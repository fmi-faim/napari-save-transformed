import numpy as np
from napari.layers import Image

from napari_save_transformed import write_transformed_layers


def test_write_transformed_layers(tmp_path):
    layers = []
    shape = (100, 200)
    image0 = np.ones(shape=shape)
    layers.append(Image(image0))
    image1 = image0.copy()
    layers.append(Image(image1, affine=[[1, 0, 30], [0, 1, 10], [0, 0, 1]]))
    out_path = tmp_path / "output.tif"
    layer_data = [layer.as_layer_data_tuple() for layer in layers]
    write_transformed_layers(path=out_path, layer_data=layer_data)

    assert out_path.exists()
