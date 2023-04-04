# Generative Adversarial Networks for anonymous Acneic face dataset generation

It is well known that the performance of any classification model is effective if the dataset used for the training process and the test process satisfy some specific requirements. In other words, the more the dataset size is large, balanced, and representative, the more one can trust the proposed model's effectiveness and, consequently, the obtained results. Unfortunately, large-size anonymous datasets are generally not publicly available in biomedical applications, especially those dealing with pathological human face images. This concern makes using deep-learning-based approaches challenging to deploy and difficult to reproduce or verify some published results. In this paper, we suggest an efficient method to generate a realistic anonymous synthetic dataset of human faces with the attributes of acne disorders corresponding to three levels of severity (i.e. Mild, Moderate and Severe). Therefore, a specific hierarchy StyleGAN-based algorithm trained at distinct levels is considered. To evaluate the performance of the proposed scheme, we consider a CNN-based classification system, trained using the generated synthetic acneic face images and tested using authentic face images. Consequently, we show that an accuracy of 97,6\% is achieved using InceptionResNetv2.
As a result, this work allows the scientific community to employ the generated synthetic dataset for any data processing application without restrictions on legal or ethical concerns. Moreover, this approach can also be extended to other applications requiring the generation of synthetic medical images. We can make the code and the generated dataset accessible for the scientific community.



## Dataset used for training StyleGAN2 model with all levels of acne severity
https://mega.nz/file/DlAREKZa#AZpAIG3loCZVSONKVSYqc9JROcw21cMLiuBvfHrSY4I

## StyleGAN2
https://github.com/NVlabs/stylegan2

## Mild Acne StyleGAN2 Model - Tensorflow (Legacy)
https://mega.nz/file/Dg53WS4A#e28CDkZ6vIbafBglHkSx34BNtq9_NEAYZG3IbjXXe-c

## Moderate Acne StyleGAN2 Model - Tensorflow (Legacy)
https://mega.nz/file/igRkELwC#-k4cmu0nJ5AaV0sAPTxlhjLvej8LB7t6-ARIs371Zxs

## Severe Acne StyleGAN2 Model - Tensorflow (Legacy)
https://mega.nz/file/TkIQFRZK#XA1LjypcKrSgoHoF-igTel3rafNpo2isg6k-tuVYkw8

## Dataset used in training CNN Models
https://mega.nz/file/G45xBTKb#BQZjqomqE69UlHBHIrHg1BPhgrBS2tYC4lD8CzoA0-U

## PyQT5 AcneGAN GUI
https://mega.nz/file/e8xEmZxQ#rbkt1cPqiBvLYdcEHbLnPvr-tAHsEO7QYwYdyCYJXGE

## StyleGAN2 Models Converted to PyTorch
https://mega.nz/folder/DoJHSYQL#Lm_tJY5huMu_MFJrieqBmg

# Citation
If you use or extend our work, please cite the following paper:


