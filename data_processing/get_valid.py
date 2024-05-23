import os
import random
import subprocess

# validation 할 파일 5000개 랜덤 샘플링하여 validation용 폴더에 옮기기

def get_text_files(path) :
    years = ['Y' + str(year) for year in range(2013, 2022)]
    months = ['M0102', 'M0304', 'M0506', 'M0708', 'M0910', 'M1112']
    
    entire_files = []
    for year in years :
        for month in months :
            folder_path = os.path.join(path, year, month)
            files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]
            entire_files += files
   
    samplelist = random.sample(entire_files, 5000)
    
    return samplelist

def mv_files(file_list) :
    dest_path = "/home/ailab/Desktop/NY/2023_ipactory/data/04_orig_valid/"
    for file in file_list :
        splitted = file.split('/')
        year, month, txt = splitted[8], splitted[9], splitted[10]
        
        command = f"mv {file} {dest_path}/{year}_{month}_{txt}"
        subprocess.call(command, shell=True)
    

valid_list = get_text_files('/home/ailab/Desktop/NY/2023_ipactory/data/03_orig_train')
mv_files(valid_list)

    