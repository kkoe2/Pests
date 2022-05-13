import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

img = 'https://ultralytics.com/images/zidane.jpg'

results = model(img)

results.show()
