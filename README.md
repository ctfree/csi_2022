# csi_2022
AI+无线通信

### 运行算力
GPU 16G

### 依赖 

pandas  
tqdm  
scikit-learn  
numpy  
psutil  
mautil  
torchinfo 

### 目录结构
data/ 存放数据  
test/ 存放验证代码    
see.py 查看模型结构  
nn.py 模型的主要代码  
train.py 训练代码的入口  
Model_define_pytorch.py 用于提交模型  


### 训练
python train.py 或者./train.sh

### 结果
输出模型结果到output   
包含  
decoder.pth.tar-${batch}  
encoder.pth.tar-${batch}  
cfg.json  

root@container-205611993c-9acdf044:~/csi/output/  CSIPlus_KF0# tree    
.  
├── cfg.json  
├── decoder.pth.tar  
├── decoder.pth.tar-101  
├── decoder.pth.tar-102  
├── decoder.pth.tar-103  
├── decoder.pth.tar-104  

### 测试验证  
cd test  
./update.sh ${batch} 将需要测试的模型拷贝到Modelsave 中  
./test.sh 验证得到测试结果  