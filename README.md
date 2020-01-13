# DeepSniffer
DeepSniffer is a model extraction framework that predicts the model architecture of the victim models based on the architecture hints during their execution. Specifically, this project mainly provides the most important part: layer sequence prediction. The key concept of DeepSniffer is to transform the layer sequence to a sequence-to-sequence prediction problem.
# Installation
1) Install the Tensorflow v1.13, Pytorch v0.4.1, and Python 3.6.2.
2) Download the model checkpoint files from the [google drive](https://drive.google.com/drive/folders/1JrTkT9C0klWFMK4x-KSMqvvPJ7k3TL6U?usp=sharing).
# Workflow
This project comprises of two parts: 1) Model extraction part: we provide the source code and data set for training and testing the layer sequence predictor which is the fundamental step for model extraction. 
2) Adversarial attack example: In the further step, we also provide the source code and trained substitute model checkpoints to evaluate the effectiveness of the extracted models on adversarial attacks. 
## Model Extraction
#### Layer Sequence Predictor Inference 
* **Predictors**: We provide the trained layer sequence predictor in /DeepSniffer/ModelExtraction/validate_deepsniffer/predictors， which can be used for predicting the layer sequence of the victim models with their architecture hints. 
* **Dataset**: We provide architecture hint feature file of several commonly-used DNN models (profiling on K40), in the following directory: DeepSniffer/ModelExtraction/dataset/typicalModels.
* **Scripts**: To infer the layer sequence of these victim models, run 
DeepSniffer/ModelExtraction/scripts/.infer_predictor_typicalmodels.sh. The results log files are stored in DeepSniffer/Results/Table4/logs. Run DeepSniffer/Results/Table4/results_analysis.py to display the final prediction error rate results.

#### Layer Sequence Predictor Training
* **Dateset**: We randomly generate computational graphs and profile the GPU performance counter information (kernel latency, read volume, write volume) during their execution to train the layer sequence predictor. The training and testing dataset is in the following directory: DeepSniffer/ModelExtraction/dataset/training_randomgraphs.

* **Scripts**: To train the layer sequence predictor for model extraction, run DeepSniffer/ModelExtraction/scripts/train_predictor.sh. The trained model is in the directory of DeepSniffer/ModelExtraction/training_deepsniffer and training log is under the model checkpoint file directory.

* **Results**: The training log files are in the following directory: DeepSniffer/Results/Figure6/logs.

## Adversarial Attack with DeepSniffer
We show an example of targeted adversarial attack on ResNet18 (Golden model). DeepSniffer adopts the extracted neural network architecture to build the substitute models. For comparison, the baseline examines the substitute models established from following networks: VGG family, ResNet family, DenseNet family, SqueezeNet, and Inception.

* **Substitute model checkpoints**: The checkpoints for the trained substitute model can be downloaded from [google drive](https://drive.google.com/drive/folders/1JrTkT9C0klWFMK4x-KSMqvvPJ7k3TL6U?usp=sharing). Unzip the files in the ''models'' to the directory of  DeepSniffer/AdversarialAttack/. 
* **Datasets**: We provide a smaller dataset from ImageNet for adversarial attack experiments, which can be downloaded from [google drive](https://drive.google.com/drive/folders/1JrTkT9C0klWFMK4x-KSMqvvPJ7k3TL6U?usp=sharing). Unzip the files in ''data_100'' under the directory of DeepSniffer/AdversarialAttack/data_100. 
* **Workflow**: To reproduce the results of Table6, go to the directory of DeepSniffer/AdversarialAttack/ and run ```python AdversarialAttack.py --iter_number=20 ``` , where iter_number indicates how many iterations (In every iteration, 100 image examples are being adversarial attack) you would like to run.

* **Results**: The log files of attack results locate in DeepSniffer/AdversarialAttack/attack_result/, go to the directory and run ```python parser_ae.py``` to display the attack success rate of different substitute models.

Additional results: If you would like to validate more cases with random selected substitute models (to reproduce the results of Figure 10), run ```python auto_attack_labels_random.py 750 245 ```, where 750 and 245 is the source label and the 245 is the target label. You can change them as you need. The attack results are shown in demofile_random.txt in the same directory (DeepSniffer/AdversarialAttack/), including the attack results of 50 randomly-ensembled substitute models.

**If you have any questions, please feel free to open issue in this project. Thanks!**


