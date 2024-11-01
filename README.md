# Generative Adversarial Networks for anonymous Acneic face dataset generation

It is well known that the performance of any classification model is effective if the dataset used for the training process and the test process satisfy some specific requirements. In other words, the more the dataset size is large, balanced, and representative, the more one can trust the proposed model's effectiveness and, consequently, the obtained results. Unfortunately, large-size anonymous datasets are generally not publicly available in biomedical applications, especially those dealing with pathological human face images. This concern makes using deep-learning-based approaches challenging to deploy and difficult to reproduce or verify some published results. In this paper, we propose an efficient method to generate a realistic anonymous synthetic dataset of human faces, focusing on attributes related to acne disorders at three distinct levels of severity (Mild, Moderate, and Severe). Notably, our approach initiates from a small dataset of facial acne images, leveraging generative techniques to augment and diversify the dataset, ensuring comprehensive coverage of acne severity levels while maintaining anonymity and realism in the synthetic data. Therefore, a specific hierarchy StyleGAN-based algorithm trained at distinct levels is considered. Moreover, the utilization of generative adversarial networks for augmentation offers a means to circumvent potential privacy or legal concerns associated with acquiring medical datasets. This is attributed to the synthetic nature of the generated data, where no actual subjects are present, thereby ensuring compliance with privacy regulations and legal considerations. To evaluate the performance of the proposed scheme, we consider a CNN-based classification system, trained using the generated synthetic acneic face images and tested using authentic face images. Consequently, we show that an accuracy of 97.6\% is achieved using InceptionResNetv2. As a result, this work allows the scientific community to employ the generated synthetic dataset for any data processing application without restrictions on legal or ethical concerns. Moreover, this approach can also be extended to other applications requiring the generation of synthetic medical images.

## Datasets
- [Dataset used for training StyleGAN2 model with all levels of acne severity](https://figshare.com/articles/dataset/StyleGAN2_Acne_Dataset/25033925)
- [Dataset used in training CNN Models](https://figshare.com/articles/dataset/StyleGAN2_Generated_Dataset_for_CNN_training/25033928)

## StyleGAN2 Trained Models Weights
- [StyleGAN2 Acne Models - Tensorflow (Legacy)](https://figshare.com/articles/software/StyleGAN2_Acne_Models_-_Tensorflow_Legacy_/25033946)
- [StyleGAN2 Acne Models - PyTorch](https://figshare.com/articles/software/StyleGAN2_Acne_Models_-_PyTorch/25033955)

## PyQT5 GUI
- [PyQT5 AcneGAN GUI - Tensorflow (Legacy)](https://figshare.com/articles/software/PyQT5_AcneGAN_GUI_-_Tensorflow/25033943)
- [PyQT5 AcneGAN GUI - Pytorch](https://figshare.com/articles/software/PyQT5_AcneGAN_GUI_-_Pytorch/27324408)

## Citation
If you use this code for your research, please cite our paper [Generative adversarial networks for anonymous  acneic face dataset generation](https://doi.org/10.1371/journal.pone.0297958)
```
@article{10.1371/journal.pone.0297958,
    doi = {10.1371/journal.pone.0297958},
    author = {Zein, Hazem AND Chantaf, Samer AND Fournier, Régis AND Nait-Ali, Amine},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Generative adversarial networks for anonymous acneic face dataset generation},
    year = {2024},
    month = {04},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0297958},
    pages = {1-15},
}
```


