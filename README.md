# 3D-ResNet-ONNX-for-Activity-Recognition

### Installation
```sh
pip install onnxruntime ,if have CPU \
pip install onnxruntime-gpu ,if have GPU \
pip install opencv-python
```
### Model

[kinetics-resnet-18.onnx]()



### How to run
Run the script this way:

```sh
# Video
python3 run.py /path/to/input_video /path/to/output_video_saved

# Webcam
python3 run.py webcam /path/to/output_video_saved
```
