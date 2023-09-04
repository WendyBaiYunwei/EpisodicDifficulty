# On the Episodic Difficulty of Few-shot Learning
## Demo the difference between curriculum training and standard training
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
Prepare the support set by running `buffer.py`. Alternatively, you can directly download an existing copy from https://drive.google.com/file/d/1nn8ry24u-SDs62QJR9mQi_4waEqZdw4S/view?usp=drive_link.

* Step 4: 
To train with curriculum learning, please run the following command:  
`python curriculum_training.py`  
To train without, please run:  
`python baseline.py`  

* This repo has been updated for code-refactoring and simplification on 2023.9.4.
* This repo implements a one-pass scheduler. There are other design choices but the key idea is the same.