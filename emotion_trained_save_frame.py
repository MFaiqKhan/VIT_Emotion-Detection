from transformers import ViTForImageClassification, ViTImageProcessor
import cv2 
from PIL import Image
import pandas as pd
import os

# List for all the predictions
output_data = []

# Trained Model on Emotion Dataset Using rtx3060

model_name = r"E:\Github\emotion_detection\models\trained"

# Load the image processor
processor = ViTImageProcessor.from_pretrained(model_name)

# Load the model
model = ViTForImageClassification.from_pretrained(model_name)

# Instantiate the video capture from the webcam
cap = cv2.VideoCapture(r"C:\Users\SHAH\Downloads\stock-footage-man-showing-different-emotions-close-up-slow-motion.webm") 

# Create output dir / images if it doesn't exist
images_path = "images"
if not os.path.exists(images_path):
   os.makedirs(images_path)

while(True): 
   
    #In the context of OpenCV's VideoCapture class, ret is a boolean value returned by the read() method. 
    #It indicates whether a frame was successfully read from the video file or camera. 
    #If ret is True, it means a frame was successfully read and stored in the frame variable. If ret is False, it means that there was an error reading the frame.
    #The line if not ret: checks if ret is False. If it is, it prints "Could not read frame" and breaks the loop, effectively ending the video capture process.
    #So, in summary, ret is used to check if a frame was successfully read from the video file or camera. If it wasn't, the loop ends and the video capture process stops 
   
   # Capture frames in the video

   ret, frame = cap.read() 
   # Make sure we got a valid frame
   if not ret:
       print("Could not read frame")
       break

   # center crop the frame to 224x224
   crop_size = 224
   height, width, channels = frame.shape
   left = int((width - crop_size) / 2)
   top = int((height - crop_size) / 2)
   right = int((width + crop_size) / 2)
   bottom = int((height + crop_size) / 2)
   frame = frame[top:bottom, left:right]

   # flip image
   frame = cv2.flip(frame, 1)

   # Convert from cv2.Mat to PIL.Image
   image = Image.fromarray(frame) 

   # Convert the PIL.Image into a pytorch tensor
   inputs = processor(images=image, return_tensors="pt")

   # Run the model on the image
   outputs = model(**inputs)

   # Get the logits (proxy for probability)
   logits = outputs.logits

   # model predicts one of the 1000 ImageNet classes
   predicted_class_idx = logits.argmax(-1).item()
   probs = logits.softmax(dim=1)

   probability = probs[0][predicted_class_idx].item()

   # Print the predicted class
   prediction = model.config.id2label[predicted_class_idx]
   print("Predicted class:", prediction)
   print("Predicted probability:", probability)

   # describe the type of font 
   # to be used. 
   font = cv2.FONT_HERSHEY_SIMPLEX 

   # Use putText() method for 
   # inserting text on video 
   cv2.putText(frame, 
               prediction,
               (50, 50), 
               font, 1, 
               (0, 255, 255), 
               2, 
               cv2.LINE_4) 

   # Display the resulting frame 
   cv2.imshow('video', frame) 

   # Save the image to len(output_data).jpg
   relative_path = os.path.join(images_path, f"{len(output_data)}.jpg")
   full_path = os.path.join(relative_path)
   image.save(full_path)

    # Create a dictionary to hold the prediction details
   prediction_details = {}
   prediction_details["prediction"] = prediction
   prediction_details["probability"] = probability
   prediction_details["file"] = relative_path

   # Append to the output data
   #prediction["file"] = relative_path
   output_data.append(prediction_details)

   # creating 'q' as the quit 
   # button for the video 
   if cv2.waitKey(1) & 0xFF == ord('q'): 
       break

# release the cap object 
cap.release() 
# close all windows 
cv2.destroyAllWindows()

# Save the csv to the output dir
df = pd.DataFrame(output_data)
df = df[["file","prediction","probability"]]
df.to_csv("predictions.csv", index=False)
