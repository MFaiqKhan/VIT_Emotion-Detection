import os
import pandas as pd

# Specify the directory containing your unstructured data
#data_dir = r'E:\Github\emotion_detection\images\images'

""" # Initialize an empty list to hold the data
data = []

# Traverse through each subdirectory in the 'train' directory
for emotion in os.listdir(data_dir):
   print(data_dir)
   emotion_dir = os.path.join(data_dir, emotion)
   print(emotion_dir)
   if os.path.isdir(emotion_dir):
       print(os.path.isdir(emotion_dir))
       # For each image file in the subdirectory, add a row to the data
       for image in os.listdir(emotion_dir):
           print(image)
           if image.endswith('.jpg'):
               data.append((os.path.join(emotion, image), emotion))

# Convert the data to a dataframe
df = pd.DataFrame(data, columns=['file', 'label'])

# Write the dataframe to a CSV file
df.to_csv('E:/Github/emotion_detection/train.csv', index=False)
 """
# Specify the directory containing your unstructured data
root_dir = r'E:\Github\emotion_detection\images'

# List of directories ('train', 'test')
directories = ['train', 'test']

for directory in directories:
   # Initialize an empty list to hold the data
   data = []

   # Full path to the current directory
   data_dir = os.path.join(root_dir, directory)

   # Traverse through each subdirectory in the current directory
   for emotion in os.listdir(data_dir):
       emotion_dir = os.path.join(data_dir, emotion)
       if os.path.isdir(emotion_dir):
           # For each image file in the subdirectory, add a row to the data
           for image in os.listdir(emotion_dir):
               if image.endswith('.jpg'):
                  data.append((os.path.join(directory, emotion, image), emotion))

   # Convert the data to a dataframe
   df = pd.DataFrame(data, columns=['file', 'label'])

   # Write the dataframe to a CSV file
   df.to_csv(f'{directory}_dataset.csv', index=False)
