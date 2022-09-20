# DeepLID

# About
This manuscript  introduces DeepLID, aiming at proposing effective Surprise Adequacy (SA) metric and Surprise Coverage (SC) criteria for analyzing test adequacy. This work calculates the neuron importance in terms of the neuron's contribution to the final decision and proposes the Local Intrinsic Dimensionality (LID)-based SA (LidSA) metric by adopting the LID estimate based on activation traces of important neurons. Two instances of LID-based SC criteria are proposed based on the LidSA value ranges of the test suite. The empirical evaluation on four image datasets demonstrates the effectiveness and efficiency of the LidSA measurement in measuring the relative surprise of test inputs. Compared with state-of-the-art coverage criteria, LID-based SC criteria are more sensitive to different error-revealing inputs including natural errors and adversarial examples.

# Repository
This repository includes details about the artifact corresponding to implementation of DeepLID. Our implementation is publicly available in DeepLID repository. This artifact allows reproducing the experimental results presented in the paper. 

# Details
craft_adv_examples.py is used to generates adversarial examples by implement five state-of-the-art adversarial attack strategies.

neural_networks: DNN models trained on four image datasets, i.e., MNIST, Fashion-MNIST, CIFAR-10, and Udacity Driving Dataset.

coverage: compute the coverage of DNN models, including, NC, KMNC, TKNC, NBC, LSC, DSC, LDSC, and TNSC. For each criterion, we utilize the hyperparameters recommended in their original research.

class: measure the LidSA values of test inputs on classification models.

reg: measure the LidSA values of test inputs on regression models.
