#!/bin/bash
python3 -m tf2onnx.convert --saved-model efficientnet-b0 --output emnist.onnx