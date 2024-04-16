import cv2 
import argparse
import os
import subprocess
from random import randrange

def clear_folder(folder):
    """Clears all files and subdirectories in the specified folder."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def FrameCapture(path):     
    """Captures frames from the video and saves them as images."""
    vidObj = cv2.VideoCapture(path)     
    count = 0
    success = True
    while success and count <= 25:
        success, image = vidObj.read()
        if success:
            # Check if the image is not empty
            if image is not None:
                cv2.imwrite("frames/frame%d.jpg" % count, image) 
                count += 1
            else:
                print("Empty frame encountered, skipping...")
        else:
            print("Failed to read frame, stopping...")

def main():
    """Main function."""
    # Clear the 'frames' folder before capturing new frames
    clear_folder('frames')

    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True,
                    help="path to video file")
    args = vars(ap.parse_args())

    # Check if the input video file exists
    if not os.path.isfile(args["input"]):
        print("Error: Input video file does not exist.")
        return

    # Capture frames from the video
    FrameCapture(args["input"])

    # Randomly select an image from the captured frames
    inputImage = "frames/frame"+str(randrange(25))+".jpg"
    print("Selected image:", inputImage)

    # Run the classification script
    runvalue = "classify_image.py --image "+inputImage
    subprocess.call("python "+runvalue)

    # Run the human activity recognition script
    harActivity = "human_activity_reco.py -m resnet-34_kinetics.onnx -c action_recognition_kinetics.txt -i "+args["input"]
    print("Running human activity recognition script with command:", harActivity)
    subprocess.call("python "+harActivity)

    # Generate report
    scores = "Reportgenearation.py"
    subprocess.call("python "+scores)

if __name__ == "__main__":
    main()
