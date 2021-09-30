# 3D-ResNet-ONNX-for-Activity-Recognition

### Installation
```sh
pip install onnxruntime ,if have CPU \
pip install onnxruntime-gpu ,if have GPU \
pip install opencv-python
```
### Model

[kinetics_moment_resnet_18.onnx](https://drive.google.com/file/d/1i3Ghm34H0Tn1Iy9Bapz_uxs8ZNesW2Ny/view?usp=sharing) \
[kinetics_moment_resnet_34.onnx](https://drive.google.com/file/d/1L1cCq37pfM91gSwetehbfSZPC6ou7MIU/view?usp=sharing)

### How to run
Run the script this way:

```sh
# Video
python3 run.py /path/to/input_video /path/to/output_video_saved

# Webcam
python3 run.py webcam /path/to/output_video_saved
```
