from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "yolo-tiny.h5"
input_path = "test1.jpg"
output_path = "newimage.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
#iterate through the detection
for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
