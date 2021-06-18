import os


def returnLocation():
    return os.getcwd()

#Creates text file with location and name of negative images.
def generate_negative_description_file(path_argument):
    path = returnLocation()
    path = path + path_argument
    with open('AccuracyTest/tneg.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir(path):
            f.write(filename + '\n')


generate_negative_description_file("\AccuracyTest\TNeg")

print(returnLocation())

# To create the positive images
#"C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_annotation.exe" --annotations=pos.txt --images=TPositives/

#To sample in the positive images:
#"C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_createsamples.exe" -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec

#Training the xml file:
#"C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_traincascade.exe" -data OPencvTraining/ -vec pos.vec -bg neg.txt -numPos 500 -numNeg 250 -numStages 12 -w 24 -h 24 -acceptanceRatioBreakValue 10e-5


# Training for the experiments face:
# "C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_traincascade.exe"
# -data OPencvTraining/Face/ -vec pos.vec -bg neg.txt -numPos 500 -numNeg 1000 -numStages 16 -w 24 -h 24 -acceptanceRatioBreakValue 10e-5


# training for the experiments mouth:
# "C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_traincascade.exe"
# -data OPencvTraining/mouth/ -vec posMouth.vec -bg neg.txt -numPos 500 -numNeg 1500 -numStages 14 -w 15 -h 15 -acceptanceRatioBreakValue 10e-5
