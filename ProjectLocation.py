import os


def returnLocation():
    return os.getcwd()

#Creates text file with location and name of negative images.
def generate_negative_description_file(path_argument):
    path = returnLocation()
    path = path + path_argument
    with open('neg.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir(path):
            f.write('Negatives/' + filename + '\n')


generate_negative_description_file("\\Negatives")

print(returnLocation())

# To create the positive images
#"C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_annotation.exe" --annotations=pos.txt --images=Positives/

#To sample in the positive images:
#"C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_createsamples.exe" -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec

#Training the xml file:
#"C:/Users/Arthur Vieillevoye/Downloads/opencv/build/x64/vc15/bin/opencv_traincascade.exe" -data OpencvTraining/ -vec pos.vec -bg neg.txt -numPos 500 -numNeg 250 -numStages 12 -w 24 -h 24
