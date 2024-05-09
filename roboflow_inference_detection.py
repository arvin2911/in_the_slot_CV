# from inference_sdk import InferenceHTTPClient
from inference.models.utils import get_roboflow_model
import cv2

# Image path
image_path = "Strike zone overlay/test_images/IMG_5201.JPG"

# Roboflow model
model_name = "pingpong-t6uto/1"
# model_version = "18"

# Get Roboflow face model (this will fetch the model from Roboflow)
model = get_roboflow_model(
    model_id="{}".format(model_name),
    api_key="JIcwLHeghA4eR7h4ye7S"
)

# Load image with opencv
frame = cv2.imread(image_path)

# Inference image to find faces
results = model.infer(image=frame,
                        confidence=0.5,
                        iou_threshold=0.5)

if results:
  for detection in results[0].predictions:
    x0 = int(detection.x - (detection.width / 2))
    x1 = int(detection.x + (detection.width / 2))
    y0 = int(detection.y - (detection.height / 2))
    y1 = int(detection.y + (detection.height / 2))
    print(f"{x0} {x1} {y0} {y1}")

    cv2.rectangle(frame, (x0, y0), (x1, y1), (255,255,0), 10)
    cv2.putText(frame, "Ball", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# Show image
cv2.imshow('Image Frame', frame)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
cv2.waitKey(1)