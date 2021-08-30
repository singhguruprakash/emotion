
from fer import Video
from fer import FER
import matplotlib.pyplot as plt
import os
import sys
# fn contains the video file name (Ensure that you only upload one file)
videofile = "in.mp4"
# Face detection
detector = FER(mtcnn=True)
# Video predictions
video = Video(videofile)

# Output list of dictionaries
raw_data = video.analyze(detector, display=False)

# Convert to pandas for analysis
df = video.to_pandas(raw_data)
df = video.get_first_face(df)
df = video.get_emotions(df)

# Plot emotions
fig = df.plot(figsize=(20, 16), fontsize=26).get_figure()
# Filename for plot
fig.savefig('my_figure.png')