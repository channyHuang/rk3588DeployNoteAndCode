import os
import argparse
import sys

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def genDataset(img_folder, output_file = 'dataset.txt'):
    file_list = sorted(os.listdir(img_folder))
    img_list = []
    with open(output_file, "w") as f:
        for path in file_list:
            if img_check(path):
                img_list.append(path)
                f.write(os.path.join(img_folder, path + '\n'))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 {} [image_folder] [output_file = './dataset.txt']".format(sys.argv[0]))
        exit(1)

    folder = sys.argv[1]
    output_file = './dataset.txt'
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    genDataset(folder, output_file)
