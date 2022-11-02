# sound_source_tracking_PPO
---
Tracking sound source via Proximal Policy Optimization. Undergoing PAKDD 2023 review process. After the review process, it will be released to the public. This repository is anonymized by https://anonymous.4open.science/. 

Dependency
---
torch, torchvision  
matplotlib  
numpy  
librosa  
nprirgen (see reference)  
  
Usage
---
1. Prepare dataset - this repository only contain sample wav file because redistributing TIMIT dataset is prohibited. You can test the code with the sample wav.
2. Run "PPO.py" for training
3. Run "make_testset.py" for generating test environments.
4. Run "PPO_test.py" for testing

reference
---
https://github.com/ty274/rir-generator  
https://github.com/nikhilbarhate99/PPO-PyTorch  
https://github.com/Anjum48/rl-examples/tree/master/ppo  
