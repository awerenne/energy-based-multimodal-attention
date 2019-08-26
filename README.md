# Energy-based Multi-Modal Attention (EMMA)

This repository contains the implementation for my thesis 'Energy-based Multi-Modal Attention' at the Montefiore (Computer & Electrical Engineering) Institute of the University of Liège (2018-2019). The report can be found [here](https://github.com/Werenne/energy-based-multimodal-attention/blob/master/report/main.pdf).

Author: Aurélien Werenne<br />
Supervisor: Raphaël Marée  
 
___


A multi-modal neural network exploits information from different channels and in different terms (e.g., images, text, sounds, sensor measures) in the hope that the information carried by each mode is complementary, in order to improve the predictions the neural network. Nevertheless, in realistic situations, varying levels of perturbations can occur on the data of the modes, which may decrease the quality of the inference process. An additional difficulty is that these perturbations vary between the modes and on a per-sample basis. This work presents a solution to this problem. 

___

The neural networks are implemented using [Pytorch](https://pytorch.org/). Notice that there is still room for improvement of the code; I will try to improve it if I can find the time.

If you find this thesis or code useful, please cite according to the following bib entry,
```
@MastersThesis{Werenne:Thesis:2019,
    author  =  {Aurélien Werenne},
    title  =  {Energy-based Multi-Modal Attention},
    school  =  {University of Liège},
    address  =  {Belgium},
    year  =  {2019}
    }
```


