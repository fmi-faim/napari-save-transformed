import numpy as np

from napari_save_transformed import write_transformed_layers


def test_write_transformed_layers(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    shape = (100, 200)
    image0 = np.ones(shape=shape)
    viewer.add_image(image0)
    image1 = image0.copy()
    viewer.add_image(image1, affine=[[1, 0, 30], [0, 1, 10], [0, 0, 1]])
    out_path = tmp_path / "output.tif"
    layer_data = [layer.as_layer_data_tuple() for layer in viewer.layers]
    write_transformed_layers(path=out_path, layer_data=layer_data)

    assert out_path.exists()
