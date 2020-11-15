# Data-driven-induction-of-fuzzy-sets-in-forensics
Bachelor's Thesis in Computer Science,  
Advisor: Prof. Dario Malchiodi,  
University of Milan, 2020.
## Overview
The scope of this repository is to illustrate the work behind my Bachelor's Thesis in Computer Science. The project describes the use of data driven induction of fuzzy sets for a classification problem in the field of forensic medicine.

Questo repository ha lo scopo di illustrare il codice realizzato durante l'esperienza di tirocinio interno triennale presso il dipartimento d'informatica dell'Università Degli Studi di Milano. Il lavoro svolto è descritto e commentato all'interno della tesi di laurea "Induzione di insiemi fuzzy in ambito medico-legale".  
The problem consists in correctly classifying people, i.e. observations of a sample of which we know personal characteristics and characteristics concerning the injuries on the body suffered in an accident, such as hit by heavy or light vehicles.
## Hardware specs
The experiments were performed with a hp model 15-aw010nl laptop which has the following features:
- CPU AMD Quad-Core A10-9600P (2,4 GHz, up to 3,3 GHz, 2 MB cache).
- 16 GB SDRAM DDR4-2133 (2 x 8 GB) memory.
- Graphic card AMD Radeon™ R7 M440 (DDR3 2 GB dedicated).  

The computation of the training of the learning model, given a single configuration of hyperparameters, takes on average 2 minutes.
## Notebook descriptions
In this repository there are several jupyter notebooks. Each of them refers to a specific part of the project and / or a specific view of the dataset on which the experiments were performed. Particularly:
- experiments\_all-feature.ipynb contains the experiments performed on the original dataset.
- first\_experiments\_20feature.ipynb contains the first experiments performed during the internship experience. The notebook has been inserted with the sole purpose of illustrating the first steps of my work and therefore presents experiments performed with simpler grids and in no particular order. I do not recommend running this notebook on your own devices.
- experiments\_20feature.ipynb contains the most significant experiments on the view of the dataset obtained by considering 20 lesion variables chosen by the coroners.
- experiments\_13feature.ipynb contains the most significant experiments on the view of the dataset obtained by considering 13 lesion variables chosen by the coroners..
- experiments\_datasetWithFeatureExtraction.ipynb contains the experiments performed on a dataset view constructed by aggregating the lesion variables by sensible areas, using a dimensionality reduction technique such as PCA.
- experiments\_dxsx.ipynb contains experiments with dataset views constructed from a subdivision of lesions on the right or left side of the body.
- experiments\_umap.ipynb contains experiments where UMAP is used as a dimensionality reduction technique.
- data\_augmentation.ipynb illustrates two techniques for oversampling the dataset in order to improve model accuracy.
- datavis.ipynb presents a possible data visualization.
- exploratory\_analysis\_for\_defuzzification.ipynb presents the exploratory analysis performed in order to defuzzy the results properly. 
- defuzzification\_20feature.ipynb contains the best defuzzificated results on the view of the dataset obtained by considering 20 lesion variables chosen by the coroners.  

If you are interested only in knowing the best results of my work, together with the variables, techniques, operators and parameters of the learning model to obtain them, I recommend reading the notebook best\_result.ipynb.
## Additional materials
Google Slides presentation is available [here](https://docs.google.com/presentation/d/1GH-OsCUFrqLLk-CFYR8U-zfByxMaecM4rlz58dKerbE/edit?usp=sharing)(ITALIAN ONLY).
