from ultralytics import YOLO

model = YOLO('yolov8m')

results = model.predict('input_videos/match_live.mp4', save = True)

print(results[0])
print('***********************************************************************************************************************')
for box in results[0].boxes:
    print(box)