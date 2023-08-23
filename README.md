[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

# TensorFlow.js: YOLOv8n test

This code is built off Sascha Dittman's object detection with TensorFlow.JS demo: https://github.com/SaschaDittmann/tfjs-cv-objectdetection. The model is a YOLOv8n model, converted to the TensorFlow.JS format with this tutorial: https://docs.ultralytics.com/modes/export/#arguments. The code to convert raw model output into coherent bounding boxes comes from https://github.com/Hyuto/yolov8-tfjs/. The model is trained off the COCO dataset, so its just for general object detection.

## Setup 

Prepare the node environments:
```sh
$ yarn
```

Run the local web server script:
```sh
$ node server.js
```

Then, to get a URL to the port this has ran the server on, install localtunnel and use it:
```sh
$ npm install -g localtunnel
$ lt --port 3000
```
