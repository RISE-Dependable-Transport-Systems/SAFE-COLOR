#!/bin/bash

cd models

for i in \
  yolov3u.pt \
  yolov5nu.pt yolov5n6u.pt yolov5su.pt yolov5s6u.pt yolov5mu.pt yolov5m6u.pt yolov5lu.pt yolov5l6u.pt yolov5xu.pt yolov5x6u.pt \
  yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt \
  yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt \
  yolov9t.pt yolov9s.pt yolov9m.pt yolov9c.pt yolov9e.pt \
  yolov10n.pt yolov10s.pt yolov10m.pt yolov10b.pt yolov10l.pt yolov10x.pt; do
  wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/$i
done
