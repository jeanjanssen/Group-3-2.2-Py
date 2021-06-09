import os


def returnLocation():
    return os.getcwd()


def generate_negative_description_file():
    path = returnLocation()
    path = path + "\\Negatives"
    with open('neg.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir(path):
            f.write('Negatives/' + filename + '\n')


#generate_negative_description_file()

#print(returnLocation())