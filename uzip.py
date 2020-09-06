import os
import zipfile
src_path = "data/data48778/selfie2anime.zip"
target_path = "selfie2anime"
def unzip_data(src_path,target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至data/dataset目录下
    '''
    if(not os.path.isdir(target_path)):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")
        
unzip_data(src_path,target_path)