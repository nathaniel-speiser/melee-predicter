import numpy as np
import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture('replays/mangozain.avi')

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60,(int(cap.get(3)),int(cap.get(4))))

counter = 0
while(True):
    ret, frame = cap.read()

    if ret == True:

        # Write the frame into the file 'output.avi'
        out.write(frame)
        counter+=1

        # Display the resulting frame
        cv2.imshow('frame',frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  # Break the loop
    else:
        break
print(counter)
# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
