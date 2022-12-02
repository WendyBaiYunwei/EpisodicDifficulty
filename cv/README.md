# CV
* Step 1: pip install scikit-image  

* Step 2: Please prepare the following folder structure for this repository. The pickle and json files can be downloaded from:  
|-data  
--|train  
--|val  
--|test  
--|embedding_new.pkl  
--|imgNameToIdx.json  
--|embedding_sim.pkl  
--|centroid_by_class.json
--|label_map.json  
|-baseline.py... 
  
* Step 3: 
To train with curriculum learning, please run the following command: 
`python curriculum_training.py`  

To train without, please run:  
`python baseline.py` 

# NLP
Please follow instructions from:  
https://github.com/zhongyuchen/few-shot-text-classification