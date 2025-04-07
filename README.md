# NGP-Net
Official Pytorch implementation of NGP-Net, from the following paper:

**NGP-Net: A Lightweight Growth Prediction Network for Pulmonary Nodules**

[Xinkai Tang](https://xinkai-tang.github.io)<sup>1+</sup>, Zhiyao Luo<sup>2+</sup>, Feng Liu<sup>1#</sup>, Wencai Huang<sup>3</sup>, Jiani Zou<sup>3#</sup>

> <sup>1</sup> School of Computer Science, Wuhan University, China .  
<sup>2</sup> Department of Engineering Science, University of Oxford, UK .  
<sup>3</sup> Department of Radiology, General Hospital of Central Theater Command of the PLA, China .  
<sup>+</sup> Tang and Luo are the co-first authors.  
<sup>#</sup> Liu and Zou are the corresponding authors.  


## Introduction
We propose NGP-Net, a novel W-shaped deep learning architecture specifically designed for dynamic nodule growth prediction by directly modelling irregularly spaced longitudinal CT data. Unlike previous methods reliant on fixed growth metrics such as volumetric or mass-change rates, NGP-Net innovatively integrates temporal sensitivity into its predictive framework through a Spatial-Temporal Encoding Module (STEM) based on dilated depthwise separable convolutions. Furthermore, NGP-Net employs a dual-branch decoder to reconstruct high-fidelity textures and shapes of nodules at arbitrary future time intervals, significantly enhancing interpretability and clinical applicability. 


## Prediction Results

### Overall Performance
Comparison of methods in terms of Dice Similarity Coefficient (DSC), Sensitivity (SEN), Positive Predictive Value (PPV), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and Mean Square Errors in overall (MSE<sub>ROI</sub>) and nodule (MSE<sub>PN</sub>) regions. The values in the table are obtained by averaging the 5-fold cross-validation results, respectively.
| Methods     | Param. | DSC(%)↑               | SEN(%)↑  | PPV(%)↑  | PSNR(dB)↑ | SSIM(%)↑ | MSE<sub>ROI</sub>↓ | MSE<sub>PN</sub>↓ |
|-------------|-------:|----------------------:|---------:|---------:|----------:|---------:|-------------------:|------------------:|
| NoFoNet     | 0.91 M | 50.78                 | 52.29    | 56.19    | 21.63     | 63.17    | 9.81×10⁻³          | 7.87×10⁻⁴         |
| WarpNet     | 0.46 M | 50.78                 | 52.29    | 56.19    | 21.62     | 63.08    | 9.81×10⁻³          | 7.86×10⁻⁴         |
| PredNet     | 0.47 M | 46.45                 | 49.62    | 52.67    | 19.99     | 56.34    | 12.3×10⁻³          | 6.77×10⁻⁴         |
| GM-AE       | 35.7 M | 61.42                 | 69.84    | 60.10    | 22.83     | 71.47    | 6.37×10⁻³          | 3.73×10⁻⁴         |
| **NGP-Net** | 1.33 M | **72.19**             | **75.00**| **74.95**| **23.18** | **71.74**| **6.13×10⁻³**      | **1.28×10⁻⁴**     |

### Performance on Distinct Growth Patterns
Growth prediction results of pulmonary nodules with different growth patterns (**34** dilation cases; **41** shrinkage cases; **75** stability cases) on our dataset. The values in the table are obtained by averaging the 5-fold cross-validation results, respectively.
| Patterns      | Methods     | DSC(%)↑   | SEN(%)↑   | PPV(%)↑   | PSNR(dB)↑ | SSIM(%)↑  | MSE<sub>ROI</sub>↓ | MSE<sub>PN</sub>↓ |
|---------------|-------------|----------:|----------:|----------:|----------:|----------:|-------------------:|------------------:|
| **Dilation**  | GM-AE       | 63.54     | 62.31     | 68.90     | **23.07** | 69.48     | **6.63×10⁻³**      | 4.19×10⁻⁴         |
|               | **NGP-Net** | **64.85** | **64.68** | **71.40** | 22.85     | **70.74** | 6.83×10⁻³          | **2.52×10⁻⁴**     |
| **Shrinkage** | GM-AE       | 56.17     | **78.01** | 45.75     | 22.57     | **71.19** | 6.67×10⁻³          | 2.65×10⁻⁴         |
|               | **NGP-Net** | **67.34** | 75.25     | **64.57** | **22.58** | 69.93     | **6.46×10⁻³**      | **2.28×10⁻⁵**     |
| **Stability** | GM-AE       | 64.82     | 70.48     | 64.74     | 22.86     | 72.52     | 6.08×10⁻³          | 3.96×10⁻⁴         |
|               | **NGP-Net** | **81.02** | **82.10** | **82.76** | **23.75** | **74.66** | **5.45×10⁻³**      | **1.25×10⁻⁴**     |


## Acknowledgement
This work is supported by National Natural Science Foundation of China (NSFC No.62172309).
