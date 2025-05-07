import argparse
import os
import shutil

def getAllFilesOfASuffixWithinAFolder(input_dir, suffix):
    res = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            filesuffix = file.split('.')[-1]
            if filesuffix == suffix:
                res.append(file)
    return res

def copyAllMarkingImgToNewFolder(markers, input_img_path, out_img_path):
    for m in markers:
        img_name = m.split('.')[0] + '.' + 'jpg'
        temp_folder = input_img_path + m[0:10] + '/'
        source = temp_folder + img_name
        destination = out_img_path + img_name
        #print(source)
        shutil.copy(source, destination)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Environment Settings', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp', '--marking_path', dest = 'marking_path', type = str, \
                        default = 'F://ResearchProjects/RIPCODES/Marking/Sat_Divided/', help='Path to marking results')
    
    parser.add_argument('-ip', '--input_img_path', dest = 'input_img_path', type = str, \
                        default = 'F://ResearchProjects/RIPCODES/Ortho/Selected_OneMinute_Divided_Saturated/', help='Path to save images')
    
    parser.add_argument('-op', '--out_img_path', dest = 'out_img_path', type = str, \
                        default = 'F://ResearchProjects/RIPCODES/Selected_Img/', help='Path to save images')
    
    args = parser.parse_args()
    
    marking_path, input_img_path, out_img_path = args.marking_path, args.input_img_path, args.out_img_path
    
    markers = getAllFilesOfASuffixWithinAFolder(marking_path, 'xml')
    
    os.makedirs(out_img_path, exist_ok=True)
    
    copyAllMarkingImgToNewFolder(markers, input_img_path, out_img_path)
    
    