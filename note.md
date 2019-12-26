https://www.intel.ai/fast-inference-with-early-exit/#gs.oc5n5e

 a confidence measure will determine if a prediction made at a certain stage can exit early from the entire deep learning topology, thus saving unnecessary processing in the subsequent layers.
 
 
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



How do we choose weight `$w_n$` for each exit point?

How do we messure the confidence of an exiting point?
+ Compute the entropy value at the exiting point: `$\text{entropy}(\mathbf{y}) = \sum_{c\in C}{ y_c \log {y_c} }$`, where `$C$` is the set of classes.
+ `y_c` is obtained by forward pass on the sub-networks

How do we determine the exiting threshold `T_n` for exiting point?

Do we forward from the first layer again for the future exiting points? I think there is more computation loss.






