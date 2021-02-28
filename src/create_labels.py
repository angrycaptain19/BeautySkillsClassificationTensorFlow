import csv
import os
from tqdm import tqdm
import config
from PIL import Image

def create_labels(config, folder, categ="beauty"):

    with open(config.dataset_path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        skill = os.path.basename(folder)

        if categ == "non_beauty":
            skill = "no_skills"

        for k, i in tqdm(enumerate(os.listdir(folder)), desc="Creating labels ...."):
            file_path = os.path.join(folder, i)           

            writer.writerow([file_path, categ, skill])
        
        print("Total Count: {}".format(k))


def convert_img(folder_path, extn="jpg"):
    for k, src_file in tqdm(enumerate(os.listdir(folder_path)), desc="Converting ...."):

        dstn_file = os.path.join(
                        folder_path, os.path.basename(src_file).split('.')[0] + '.jpg'
                        )
        
        src_file = os.path.join(folder_path, src_file)
        # print(src_file.split('.')[1])
        if src_file.split('.')[1] == "png":
            print(src_file)
            img = Image.open(src_file)
            rgb_im = img.convert('RGB')
            rgb_im.save(dstn_file)

    print("Done")

if __name__ == "__main__":

    # multiple folders inside a parent folder
    config,_ = config.get_config()
    if not os.path.exists(config.dataset_path):
        with open(config.dataset_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['file_path', 'isbeauty', 'skill'])

    # src = "/home/sumit/Documents/SEW/Dataset/non_beauty"

    # create_labels(config, src, categ="non_beauty")
    # for i in os.listdir(src):
    #     folder = os.path.join(src, i)
    #     create_labels(config, folder, categ="beauty")


    # convert_img(src)
    # list(filter(convert_img, src))
    # neg = "/media/sumit/Data/Workspace/SEW/dataset/non_beauty"
        
    

