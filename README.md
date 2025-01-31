# DeepNetBeam (DNB)
This repository contains the code and database associated with the paper "[Applications of scientific machine learning for the analysis of functionally graded porous beams]([.......](https://www.sciencedirect.com/science/article/pii/S0925231224018903?via%3Dihub))".

## Paper Title:
**[Applications of scientific machine learning for the analysis of functionally graded porous beams](https://www.sciencedirect.com/science/article/pii/S0925231224018903?via%3Dihub)**

### Abstract:
This study introduces a framework, named DeepNetBeam (DNB), based on Scientific Machine Learning (SciML) for the analysis of functionally graded (FG) porous beams. The beam material properties are assumed to vary as an arbitrary continuous function. The DNB framework considers the output of a neural network/operator as an approximation to the displacement fields and derives the equations governing beam behavior based on the continuum formulation. The framework is formulated by three approaches: 
1. The **vector approach** leads to Physics-Informed Neural Network (PINN),
2. The **energy approach** brings about Deep Energy Method (DEM),
3. The **data-driven approach** results in a class of Neural Operator methods.

Finally, a neural operator has been trained to predict the response of FG porous beams under any porosity distribution pattern and any arbitrary traction condition. The results are validated with analytical and numerical reference solutions.

---

## Overview:
The code and database provided in this repository are related to the paper **"Applications of scientific machine learning for the analysis of functionally graded porous beams"**, which explores the application of various machine learning techniques for the analysis of functionally graded porous beams. 
The three approaches (PINN, DEM, Neural Operator) are implemented to allow flexibility and extension for future use.

## Requirements:
The required packages have been mentioned in each section separately. 

## Datasets:
The datasets used in the paper are available in the [Link](https://seafile.cloud.uni-hannover.de/d/299afa7ad11545cb9a01/)  . 
The datasets cover the required database for sections 2 and 3.

You can also generate the datasets using the provided scripts.

## Contributing
We welcome contributions to improve the implementation and extend the framework. Please fork the repository, create a feature branch, and submit a pull request with your changes.

## Contact:

For any inquiries or issues regarding this repository, please feel free to reach out:

**Mohammad Sadegh Eshaghi**  
[eshaghi.khanghah@iop.uni-hannover.de]  
[GitHub](https://github.com/eshaghi-ms)  
[LinkedIn](https://www.linkedin.com/in/mohammad-sadegh-eshaghi-89679b240/) 

**Prof. Ph.D. Xiaoping Zhuang**  
[zhuang@iop.uni-hannover.de]  
[IOP](https://www.iop.uni-hannover.de/de/zhuang)  
[LinkedIn](https://www.linkedin.com/in/xiaoying-zhuang-5306a073/) 

## How to Cite:
If you use this code in your research, please cite the following paper:

```bibtex
@article{eshaghi2025applications,
  title={Applications of scientific machine learning for the analysis of functionally graded porous beams},
  author={Eshaghi, Mohammad Sadegh and Bamdad, Mostafa and Anitescu, Cosmin and Wang, Yizheng and Zhuang, Xiaoying and Rabczuk, Timon},
  journal={Neurocomputing},
  volume={619},
  pages={129119},
  year={2025},
  publisher={Elsevier}
}
```
 
