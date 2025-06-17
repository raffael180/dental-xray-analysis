# YOLO12n for Dental Disease Detection on the DENTEX Dataset

[English](#english-version) | [Português](#versão-em-português)

---

## English Version

### Project Overview

This project presents a pipeline for dental pathology detection using the **YOLO12n** model by Ultralytics. The core of this work involved adapting the public **DENTEX** dataset from its original Detectron2 format to the YOLO format, suitable for training with the latest YOLO models. The goal is to leverage a state-of-the-art, real-time object detector for the accurate and efficient identification of various dental conditions in panoramic X-ray images.

### Key Contributions

- **Dataset Conversion:** Development of a custom pipeline to convert the DENTEX dataset annotations from Detectron2's JSON format to the YOLO plain text format.
- **Model Training:** Training and validation of the YOLO12n model on a converted subset of the DENTEX dataset (approx. 700 images).

### Model: YOLO12n

The detection model is based on YOLO12n, a recent, highly efficient real-time object detection model from Ultralytics. This choice was motivated by its balance of high accuracy and low inference latency, making it suitable for practical applications.

### Dataset: DENTEX

This work utilizes the **DENTEX (Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays)** dataset. All credits for the data collection, annotation, and original publication belong to its creators.

- **Original Repository:** [https://github.com/ibrahimethemhamamci/DENTEX](https://github.com/ibrahimethemhamamci/DENTEX)
- **Format Used:** A subset of the original dataset was converted to YOLO format for this project.

### Getting Started

#### Prerequisites

- Python 3.9+
- PyTorch
- Ultralytics YOLO library

#### Installation

1.  Clone the repository:
    ```sh
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repository-Name].git
    ```
2.  Navigate to the project directory:
    ```sh
    cd [Your-Repository-Name]
    ```
3.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

To run inference on a new image using the trained model, use the following example script:

```python
!pip install ultralytics
from ultralytics import YOLO

# Load the custom-trained YOLO12n model
model = YOLO("yolo12n.pt")

# Train
results = model.train(
    data="/content/dentex_yolo12n/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    name='dentex_train'
    )

# The results object contains detections, masks, etc.
# To visualize the results with bounding boxes:
results = model.predict(
    source="/content/dentex_yolo12n/images/val/train_655.png",
    conf=0.25,
    iou=0.7,
    save=True,
    save_crop=True,
    save_txt=True
)

import matplotlib.pyplot as plt

plt.imshow(results[0].plot())
plt.axis('off')
plt.show()

```

### Citations and License

#### Academic Citation

If you use DENTEX in your research, the original authors request references to the following papers:

```bibtex
@article{hamamci2023dentex,
  title={DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and Yuksel, Atif Emre and Gultekin, Sadullah and Ozdemir, Serife Damla and Yang, Kaiyuan and Li, Hongwei Bran and Pati, Sarthak and Stadlinger, Bernd and others},
  journal={arXiv preprint arXiv:2305.19112},
  year={2023}
}

@inproceedings{hamamci2023diffusion,
  title={Diffusion-based hierarchical multi-label object detection to analyze panoramic dental x-rays},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and Sekuboyina, Anjany and Gundogar, Mustafa and Stadlinger, Bernd and Mehl, Albert and Menze, Bjoern},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={389--399},
  year={2023},
  organization={Springer}
}
```

#### License

- The original DENTEX dataset is provided under the **CC BY-SA 4.0 License**.
- All custom code in this repository, including the conversion scripts and training pipeline, is provided under the **MIT License**.

---

## Versão em Português

### Resumo do Projeto

Este projeto apresenta um pipeline para detecção de patologias dentárias utilizando o modelo **YOLO12n** da Ultralytics. O foco deste trabalho foi a adaptação do dataset público **DENTEX**, convertendo-o de seu formato original (Detectron2) para o formato YOLO, compatível com o treinamento dos modelos YOLO mais recentes. O objetivo é aplicar um detector de objetos de última geração e em tempo real para a identificação precisa e eficiente de diversas condições dentárias em imagens de raios-X panorâmicos.

### Principais Contribuições

- **Conversão do Dataset:** Desenvolvimento de um pipeline customizado para converter as anotações do dataset DENTEX do formato JSON (Detectron2) para o formato de texto plano (YOLO).
- **Treinamento do Modelo:** Treinamento e validação do modelo YOLO12n em um subconjunto convertido do dataset DENTEX (aprox. 700 imagens).

### Modelo: YOLO12n

O modelo de detecção é baseado no YOLO12n, um recente e altamente eficiente modelo de detecção de objetos em tempo real da Ultralytics. Esta escolha foi motivada pelo seu equilíbrio entre alta acurácia e baixa latência de inferência, tornando-o adequado para aplicações práticas.

### Dataset: DENTEX

Este trabalho utiliza o dataset **DENTEX (Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays)**. Todos os créditos pela coleta, anotação e publicação original dos dados pertencem aos seus criadores.

- **Repositório Original:** [https://github.com/ibrahimethemhamamci/DENTEX](https://github.com/ibrahimethemhamamci/DENTEX)
- **Formato Utilizado:** Um subconjunto do dataset original foi convertido para o formato YOLO para este projeto.

### Como Começar

#### Pré-requisitos

- Python 3.9+
- PyTorch
- Biblioteca Ultralytics YOLO

#### Instalação

1.  Clone o repositório:
    ```sh
    git clone [https://github.com/](https://github.com/)[Seu-Usuario]/[Nome-Do-Repositorio].git
    ```
2.  Navegue até o diretório do projeto:
    ```sh
    cd [Nome-Do-Repositorio]
    ```
3.  Instale as dependências necessárias:
    ```sh
    pip install -r requirements.txt
    ```

### Como Usar

Para executar uma inferência em uma nova imagem utilizando o modelo treinado, use o seguinte script como exemplo:

```python
!pip install ultralytics
from ultralytics import YOLO

# Carrega o modelo YOLO12n treinado
model = YOLO("yolo12n.pt")

# Realiza o treino do modelo
results = model.train(
    data="/content/dentex_yolo12n/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    name='dentex_train'
    )

# O objeto 'results' contém as detecções, máscaras, etc.
# Para visualizar os resultados com as caixas delimitadoras (bounding boxes):
results = model.predict(
    source="/content/dentex_yolo12n/images/val/train_655.png",
    conf=0.25,
    iou=0.7,
    save=True,
    save_crop=True,
    save_txt=True
)

import matplotlib.pyplot as plt

plt.imshow(results[0].plot())
plt.axis('off')
plt.show()

```

### Citações e Licença

#### Citação Acadêmica

Se você utilizar o DENTEX em sua pesquisa, os autores originais solicitam referências aos seguintes artigos:

```bibtex
@article{hamamci2023dentex,
  title={DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and Yuksel, Atif Emre and Gultekin, Sadullah and Ozdemir, Serife Damla and Yang, Kaiyuan and Li, Hongwei Bran and Pati, Sarthak and Stadlinger, Bernd and others},
  journal={arXiv preprint arXiv:2305.19112},
  year={2023}
}

@inproceedings{hamamci2023diffusion,
  title={Diffusion-based hierarchical multi-label object detection to analyze panoramic dental x-rays},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and Sekuboyina, Anjany and Gundogar, Mustafa and Stadlinger, Bernd and Mehl, Albert and Menze, Bjoern},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={389--399},
  year={2023},
  organization={Springer}
}
```

#### Licença

- O dataset original DENTEX é fornecido sob a licença **CC BY-SA 4.0**.
- Todo o código customizado neste repositório, incluindo os scripts de conversão e o pipeline de treinamento, é fornecido sob a **Licença MIT**.
