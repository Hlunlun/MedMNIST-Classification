# Classification of Chest X-Ray images: Pneumonia, Turberculosis and Normal 

## Get Started
1. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

2. Start the system
    ```bash
    streamlit run app.py
    ```

## Configuration
Nvidia GeForce RTX 4060 (8GB)


## Dataset
| Dataset | Samples | Classes |
|-|-|-|
| [PneumoniaMNIST](https://github.com/Hlunlun/MedMNIST-Classification) | 5,856 | 2 |
|Tuberculosis Chest X-Ray Images | 1,000 | 3 |

### Preprocessing
```graphviz
digraph DataPreprocessing {
    rankdir=TB;
    node [shape=rectangle, style="filled, rounded", fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];

    // Data Sources
    subgraph cluster_sources {
        label = "Data Sources";
        style = dashed;
        color = lightgrey;
        tb_dir [label="Local TB Directory\n(Normal / Tuberculosis)", fillcolor="#FFF3E0"];
        mnist_api [label="MedMNIST API\n(PneumoniaMNIST)", fillcolor="#E1F5FE"];
    }

    // Processing TB Data
    load_tb [label="Load File Paths & Map Labels\n(Normal: 0, TB: 2)", fillcolor="#FFE0B2"];
    tb_df [label="tb_df (Pandas DataFrame)", fillcolor="#FFE0B2"];
    split_tb [label="train_test_split\n(test_size=0.2)", fillcolor="#FFCC80"];
    
    // Processing MNIST Data
    mnist_train [label="mnist_train (split='train')", fillcolor="#B3E5FC"];
    mnist_test [label="mnist_test (split='test')", fillcolor="#B3E5FC"];

    // Custom Dataset Wrapper
    wrapper_tb_train [label="MultiDiseaseDataset\n(is_mnist=False)", fillcolor="#C8E6C9"];
    wrapper_tb_test [label="MultiDiseaseDataset\n(is_mnist=False)", fillcolor="#C8E6C9"];
    wrapper_mnist_train [label="MultiDiseaseDataset\n(is_mnist=True)", fillcolor="#C8E6C9"];
    wrapper_mnist_test [label="MultiDiseaseDataset\n(is_mnist=True)", fillcolor="#C8E6C9"];

    // Final Output
    concat_train [label="ConcatDataset\n(train_set)", fillcolor="#4CAF50", fontcolor="white", style="filled, bold"];
    concat_test [label="ConcatDataset\n(test_set)", fillcolor="#2196F3", fontcolor="white", style="filled, bold"];

    // Connections
    tb_dir -> load_tb;
    load_tb -> tb_df;
    tb_df -> split_tb;
    
    split_tb -> wrapper_tb_train [label="train_tb_df"];
    split_tb -> wrapper_tb_test [label="test_tb_df"];

    mnist_api -> mnist_train;
    mnist_api -> mnist_test;

    mnist_train -> wrapper_mnist_train;
    mnist_test -> wrapper_mnist_test;

    wrapper_mnist_train -> concat_train;
    wrapper_tb_train -> concat_train;

    wrapper_mnist_test -> concat_test;
    wrapper_tb_test -> concat_test;
}
```


## Model
|Backbone|ACC|
|-|-|
|[ResNet50](https://arxiv.org/abs/1512.03385)|0.92|
|[EfficientNet-B0](https://arxiv.org/abs/1905.11946)|0.90|
|[ConvNeXt-Tiny](https://arxiv.org/abs/2201.03545)|0.93|
|[Vision Transformer](https://arxiv.org/abs/2010.11929)||

### RestNet50:
- Performance on every class
  |precision|recall|f1-score|support|
  |-|-|-|-|
  |Normal|1.00|0.88|0.94|937|
  |Pneumonia|0.78|1.00|0.88|390|
  |Tuberculosis|0.98|0.99|0.99|137|
- Overall Performance
  |precision|recall|f1-score|support|
  |-|-|-|-|
  |micro avg|0.92|0.96|0.93|1464|
  |weighted avg|0.94|0.92|0.93|1464|

### EfficientNet-B0
- Performance on every class
  ||precision|recall|f1-score|support|
  |-|-|-|-|-|
  |Normal|1.00|0.85|0.92|937|
  |Pneumonia|0.73|1.00|0.85|390|
  |Tuberculosis|1.00|0.99|0.99|137|
- Overall Performance
  ||precision|recall|f1-score|support|
  |-|-|-|-|-|
  |macro avg|0.91|0.94|0.92|1464|
  |weighted avg|0.93|0.90|0.91|1464|

### ConvNeXt-Tiny
- Performance on every class
  ||precision|recall|f1-score|support|
  |-|-|-|-|-|
  |Normal|1.00|0.89|0.94|937|
  |Pneumonia|0.80|1.00|0.89|390|
  |Tuberculosis|0.99|1.00|0.99|137|
- Overall Performance
  ||precision|recall|f1-score|support|
  |-|-|-|-|-|
  |macro avg|0.93|0.96|0.94|1464|
  |weighted avg|0.95|0.93|0.93|1464|
  

### Vision Transformer




## Training Scripts
You can compare different model using the training scripts in [JupyterNotebook/](https://github.com/Hlunlun/MedMNIST-Classification/tree/master/JupyterNotebook)

```bash
JupyterNotebook
  |-- main.ipynb
  |-- ensemble.ipynb
  |-- inference.ipynb
```

## Results
- ResNet50
![image](static/restnet50.png)

- EfficientNet-B0
![image](static/efficientnet-b0.png)

- ConvNeXt-Tiny
![image](static/convnext-tiny.png)

- Vision Transformer
![image](static/vit.png)




