# CV
* Step 1: pip install scikit-image (along with other more common libraries; e.g: pytorch)

* Step 2: Please prepare the following folder structure for this repository. The pickle and json files can be downloaded from https://drive.google.com/file/d/1LUzzjta28ZHURvqNIl1ZZhZRBIAq_BeU/view?usp=sharing:  
-|data  
--|train (miniImagenet train folder)  
--|val (miniImagenet val folder)  
--|test (miniImagenet test folder)   
--|embedding_new.pkl  
--|imgNameToIdx.json  
--|embedding_sim.pkl  
--|centroid_by_class.json  
--|label_map.json  
-|baseline.py... 
  
* Step 3: 
Prepare the support set by running `buffer.py`

* Step 4: 
To train with curriculum learning, please run the following command:  
`python curriculum_training.py`  

# NLP
* Please follow main instructions from:  
https://github.com/zhongyuchen/few-shot-text-classification  

* To train with curriculum learning, please run `main.py`  

* To train without, please run `main_contrast.py`