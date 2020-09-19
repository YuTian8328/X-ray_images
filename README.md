# X-ray_images


Theoreically, computer-aided [patients triage](https://en.wikipedia.org/wiki/Triage) system can perform a automatic initial interpretation of the periodontal condition of patients in a dental clinic. It can help to classify patients into different categories, severe periodontitis (more than 1/2 bone around the teeth root is lost, left image), moderate periodontits (1/3-1/2 bone is lost, middle image), mild periodontitis or healthy periodontal tissue (less than 1/3 bone is lost, right image). Based on this classification patients will be refered to an oral hygienist (mild periodontitis) or a general dentist (moderate periodontitis) or a specialist (severe periodotits).

# Problem Formulation: 
- It's a Multi-class classification problem
- One set of X-ray image is one observation
- Label is a categorical variable, 0 for mild or healty, 1 for moderate and 2 for severe
- Train a PCA(or NMF) + SVC model by applying Hinge loss function to measure the quality of SVC
- Or implement a Neural Network based on a pretrained model (transfer learning). Use cross entropy loss function as train.

# Method
- Images are collected from my prior work, for deep learning it's a small dataset, so image augmentation and transfer learning is necessary process
- K-fold cross validation will be applyed
  - Shuffle the dataset randomly
  - Split the dataset into k groups 
- manually analyze miss classified samples and modify evaluation metrics or loss function, as severe samples being classified as moderate or mild should be avoided，otherwise moderate samples being classified as severe won't hurt that much
- Libraries: Tensorflow, keras, imageio, PIL, numpy, matplotlib...