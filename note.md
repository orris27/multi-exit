https://www.intel.ai/fast-inference-with-early-exit/#gs.oc5n5e

 a confidence measure will determine if a prediction made at a certain stage can exit early from the entire deep learning topology, thus saving unnecessary processing in the subsequent layers.
 
+ Leverage the variance of difficulty among real-world data and thus uses only part of the network to handle recognition tasks: Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition
+ Multi-Scale Dense Network (MSDNet) 
+ selectively inserting exits between specific layers. Checks for an ability to exit were done after some amount of extra processing on the exit branches themselves: BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks.
+ perform dynamic routing of the data and thus skip certain layers of processing along the way: SkipNet: Learning Dynamic Routing in Convolutional Networks




Confidence Measure:

CrossEntropy
+ Training: weighted (multiple exit points) => overall loss
+ Inference: entropy < threshold => exit ok


[Distiller](https://ai.intel.com/compressing-deep-learning-models-with-neural-network-distiller/): nn compression research


[Early exit](https://nervanasystems.github.io/distiller/algo_earlyexit/index.html) is a new feature in Distiller and is available as an Open Source package on [Github](https://github.com/NervanaSystems/distiller).




### BranchyNet

additional side branch classifiers

+ For many simple test examples can exit the network early via these branches when they are infered with high confidence.
+ For difficult examples, deeper networks are still utilized.


Jointly Optimization: 
+ optimizes the weighted loss of all exit points
+ each exit point provides regularization on others?
+ The final loss function (N is the total number of exiting point): 
```
$L_{branchynet}(\hat{\mathbf{y}}, \mathbf{y}; \theta) = \sum_{n=1}^N { w_n L(\hat{\mathbf{y}}_{exit_n}, \mathbf{y}; \theta)  }$
```


Structures of Branches:
+ The computation cost of branch should be less than that of exiting at a later exit point
+ Earlier branch has more layers, and later branch has fewer layers.


How do we choose weight `$w_n$` for each exit point?
+ higher weights at early exiting point

How do we messure the confidence of an exiting point?
+ Compute the entropy value at the exiting point: `$\text{entropy}(\mathbf{y}) = \sum_{c\in C}{ y_c \log {y_c} }$`, where `$C$` is the set of classes.
+ `y_c` is obtained by forward pass on the sub-networks

How do we determine the exiting threshold `T_n` for exiting point?
+ determined by application, i.e., as long as the accuracy and speed meets the requirements.

How do we choose the exiting location?
+ Depends on the difficulty of the task (dataset, etc.)

Do we forward from the first layer again for the future exiting points? I think there is more computation loss.



### SkipNet
Modify ResNet

Gating Network:
+ Add to each residual network: takes the outputs of the previous layer as inputs, and outputs 0/1 to determine whether to skip the block (1: no-skip; 0: skip)
+ non-differentiable
    - ~~gradient descent~~
    - softmax softer? make it differentiable
    - fidelity + penalty (=reward)
    
Loss Function:
$$
L_{\theta}(g, X) = L(\hat{y}(X, F_{\theta}, g), y) - \frac{\alpha}{N} \sum_{i=1}^N{(1 - g_i ) C_i},
$$
where $g_i$ is the $i$th decision (0/1), $N$ is the number of decisions, $C_i$ is hyperparamter to measure the importance of $F_{\theta}^i$ (the authors select $C_i$ as 1), $\alpha$ is another hyperparameter, $F_{\theta}^i$ is the set of network parameters for $i$th layer including gating network, $F_{\theta} = [F_{\theta}^1, F_{\theta}^2, \cdots, F_{\theta}^N]$

