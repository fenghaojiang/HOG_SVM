import os
import re

def main():
    pascal_path = 'E:/HOG_SVM/Project1/Project1/INRIAPerson/Train/annotations/'
    pascal_list = os.listdir(pascal_path)
    print(len(pascal_list))
    cnt = 0

    for pascal_file in pascal_list:
        f = open(pascal_path + pascal_file, encoding='gbk')
        line_list = f.readlines()

        str_line = ''
        for line in line_list:
            if str(line).__contains__('Image filename'):
                str_line = line.strip().split('/')[2][0:-1] + '\n'   # remove the end of "
                break

        for line in line_list:
            if str(line).__contains__('Objects with ground truth'):
                nums = re.findall(r'\d+', str(line))
                str_line = str_line + str(nums[0]) + '\n'
                # print(str_line)
                break

        for index in range(1, int(nums[0]) + 1):
            for line in line_list:
                if str(line).__contains__("Center point on object " + str(index)):
                    center = re.findall(r'\d+', str(line))
                    str_line = str_line + '(' + center[1] + ',' + center[2] + ')' + '\n';
                if str(line).__contains__("Bounding box for object " + str(index)):
                    coordinate = re.findall(r'\d+', str(line))
                    str_line = str_line + '(' + coordinate[1] + ',' + coordinate[2] + ')' + '(' + coordinate[3] + ',' + coordinate[4] + ')' + '\n'
        f.close()
        cnt += 1
        print(str_line)
        print("The number of the images is {}".format(cnt))
        with open('E:/PAT/shipin/Project1/Project1/TrainAnnotation.txt', 'a', encoding='utf-8') as fh:
            fh.write(str_line)


if __name__ == "__main__":
    main()
