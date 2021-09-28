import os
import sys
from collections import deque
import traceback
import numpy as np
import cv2
import time
import onnxruntime as ort

CLASSES = open('action_recognition_kinetics_moments.txt').read().strip().split("\n")
DURATION = 16
INPUT_SIZE = 112
stream = 'act.gif'
model = 'resnet-18-kinetcis-moments.onnx'
frameskip = 2
save_output = ''

def onnx_load_model(onnx_model):
    sess = rt.InferenceSession(onnx_model,None)
    return sess

def run_inference(host_input):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    outputs = ''
    outputs = sess.run([output_name], {input_name: host_input})
    return CLASSES[np.argmax(outputs)]

if __name__ == '__main__':
    if stream == '':
        print('[Error] Please provide a valid path --stream.')
        sys.exit(0)

    if model == '':
        print('[Error] Please provide a valid path --model.')
        sys.exit(0)

    model = onnx_load_model(model)

    if not save_output == '':
        writer = cv2.VideoWriter(save_output,cv2.VideoWriter_fourcc(*'MJPG'),60, (1920, 1080))
    source = cv2.VideoCapture(0 if stream == 'webcam' else stream)
    frames = deque(maxlen=DURATION)
    skip = 0
    result = ''
    inferencetime = 0
    while True:
        ret, frame = source.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1920, 1080))
        skip += 1
        if skip % frameskip == 0:
            skip = 0
            frames.append(frame)
            if not len(frames) < DURATION:
                blob = cv2.dnn.blobFromImages(frames, (1.0/255),(INPUT_SIZE, INPUT_SIZE), (110.79, 103.3, 96.26),swapRB=True, crop=True)
                blob = np.transpose(blob, (1, 0, 2, 3))
                blob = np.expand_dims(blob, axis=0)
                blob = np.ascontiguousarray(blob)
                start = time.time()
                result = run_inference(blob)
                inferencetime = round(time.time() - start, 4)
                print('Inference Time: {} ms'.format(inferencetime), end='\r')
        overlay = frame.copy()
        display = frame.copy()
        cv2.rectangle(overlay, (560, 850), (1360, 1000), (0, 0, 0), -1)
        cv2.putText(overlay, 'Inference Time: {} s'.format(inferencetime), (600, 900), cv2.FONT_HERSHEY_COMPLEX,
            1.25, (255, 255, 255), 2)
        cv2.putText(overlay, 'Output: {}'.format(result), (600, 950), cv2.FONT_HERSHEY_COMPLEX,
            1.25, (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.7, display, 1 - 0.7, 0, display)
        cv2.imshow('Output', display)
        if not save_output == '':
            writer.write(display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    source.release()
    writer.release()
    cv2.destroyAllWindows()