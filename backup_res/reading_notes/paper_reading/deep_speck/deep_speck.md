# 预备知识

## 密码

### SPECK 密钥拓展算法

 ![密钥扩展](D:\pics\typora_pics\speck_key_expand.png)



## 深度学习

### 神经网络差分区分器

神经网络差分区分器最早由 Gohr提出, 由一个残差神经网络构成

#### 深度残差神经网络

 残差神经网络是深度学习的模型之一. 利用该网络, 对给定输入差分的某算法的输出差分分布特征进行学习, 并利用训练的网络对给定的密文进行预测, 看该密文是否符合其输出差分分布. 区分器进行预测时, 利用了输出差分的分布特征, 这 其中很多特征对于传统的纯差分区分器来说是不可见的, 并且该区分器在用于密钥恢复攻击时, 大大降低 了所需数据复杂度和时间复杂度. 下文我们将该区分器简称为单差分神经网络区分器.



#### ResNet 

假设有单个图像$x_0$通过一个卷积神经网络。网络包含L个层，每层都执行一个非线性转换 $H_l(\cdot)$，这里 $l$ 是层的索引， $ H_{l}(\cdot) $ 可以是诸如 卷积，池化，Relu激活，batch normalizatioin 的复合函数。我们将输出的第 $l$ 层表示为 $x_l$ 。

传统的卷积前馈网络连接第$l$ 层的输出作为第 $l+1$层的输入，这引起了如下的层转换：$X_l = H_l ( x_{l − 1 })$ 。

而ResNet增加一个跳过连接skip connection，绕过非线性变换，通过一个身份函数：
$$
X_l = H_l ( x_{l − 1 }) + X_{l - 1}\quad (1)
$$
ResNets的一个优点是，梯度可以直接通过身份函数从后面的层流到前面的层。但是，身份函数和 $H_l$ 的输出通过求和相结合，可能会阻碍网络中的信息流。

**Dense connective**： 为了进一步改善层之间的信息流，我们提出了一种不同的连接模式：引入了从任何层到所有后续层的直接连接。图1显示了所得DenseNet的布局。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200204112234907.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4ODYzNDEz,size_16,color_FFFFFF,t_70)

因此，第 $l$ 层接收所有先前层的特征图：$x_0 ， ⋯ x_l$作为输入：
$$
x_l = H_l ([x_0 , x_1 , \cdots   , x_{l − 1}] ) \quad(2)
$$
这里$[x_0 , x_1 , \cdots   , x_{l − 1}]$是前面$0,1,2,\cdots l-1$ 的层的级联。由于其紧密的连接性，我们将此网络架构称为“密集卷积网络”（DenseNet）。为了易于实现，我们将等式（2）中的 $H_l(\cdot) $ 的多个输入串联到一个张量中。



## 名词解释

### Markov property 马尔可夫性质

（英语：Markov property）是[概率论](https://zh.wikipedia.org/wiki/概率论)中的一个概念，因为俄国数学家[安德雷·马尔可夫](https://zh.wikipedia.org/wiki/安德雷·馬可夫)得名[[1\]](https://zh.wikipedia.org/wiki/马尔可夫性质#cite_note-1)。当一个[随机过程](https://zh.wikipedia.org/wiki/随机过程)在给定现在状态及所有过去状态情况下，其未来状态的条件[概率分布](https://zh.wikipedia.org/wiki/概率分布)仅依赖于当前状态；换句话说，在给定现在状态时，它与过去状态（即该过程的历史路径）是[条件独立](https://zh.wikipedia.org/wiki/条件独立)的，那么此[随机过程](https://zh.wikipedia.org/wiki/随机过程)即具有马尔可夫性质。具有马尔可夫性质的过程通常称之为[马尔可夫过程](https://zh.wikipedia.org/wiki/马尔可夫过程)

**马尔可夫链**

https://zhuanlan.zhihu.com/p/26453269

Markov cipher

 a Markov cipher is an iterated block cipher in which the probability of the individual differential transitions is independent of the concrete plaintext values if the subkeys applied to
each round are chosen in a uniformly random manner



### **double-precision arithmetic**

单精度和双精度精确的范围不一样

单精度（float），一般在计算机中存储占用4字节，也32位，有效位数为7位；

双精度（double）在计算机中存储占用8字节，64位，有效位数为16位

32位浮点数就是单精度，64位浮点数就是双精度。



### downsamp 下采样

缩小中间数据=DownSampling（下采样）=SubSampling（子采样），方法有：Max Pooling、Average Pooling、Sum Pooling等。
增大中间数据=UpSampling（上采样）=SuperSampling（超采样），方法有：UnPooling、Deconvolution、Bilinear等。



### Ridge Regression 岭回归
最小二乘回归 Least squares，又称最小平方法
“最小二乘法”的核心就是**保证所有数据偏差的平方和最小。**



岭回归(Ridge Regression)是在平方误差的基础上增加正则项.
可以认为是在回归分析中，用一种方法改进回归系数的最小二乘估计后所得的回归
$$
\sum_{i=1}^{n}\left ( y_i-\sum_{j=0}^{p}w_jx_{ij} \right )^2+\lambda \sum_{j=0}^{p}w^2_j
$$


通过确定的$\lambda$值可以使得在方差和偏差之间达到平衡：随着$\lambda$的增大，模型方差减小而偏差增大。（方差指的是模型之间的差异，而偏差指的是模型预测值和数据之间的差异。我们需要找到方差和偏差的折中）

**岭回归是对最小二乘回归的一种补充，它损失了无偏性，来换取高的数值稳定性，从而得到较高的计算精度。**





### **Neutral Bits**

neutral bits, i.e., they do not affect the differences of the intermediate data for 15–20 rounds.

![截屏2022-10-11 17.35.42](/Users/grhunhun/Library/Application Support/typora-user-images/截屏2022-10-11 17.35.42.png)



设 $M$ 和 $M'$ 是一对符合 $\sigma_r$ 的message（对某些轮数$r \geq 16$）
如果添加了 $i'$ 位后的 $M$ 和 $M'$ 仍满足 $\sigma_r$ ，
则消息的第 $i'$  位 ( $i' \in \{0, \cdots , 511\}$  ) 对于 $M$ 和 $M'$ 分别是一个中性位



### 最高有效位

**最高有效位**（英语：**Most Significant Bit**，**msb**），是指一个n位二进制数字中的n-1位，具有最高的权值![2^{n-1}](https://wikimedia.org/api/rest_v1/media/math/render/svg/80c2cb3e3a7de902c9503fb34a17641df5896539)。与之相反的称之为[最低有效位](https://zh.wikipedia.org/wiki/最低有效位)。在[大端序](https://zh.wikipedia.org/wiki/字节序#大端序)中，msb即指最左端的位。

对于[有符号二进制数](https://zh.wikipedia.org/wiki/有符號數處理)，负数采用[反码](https://zh.wikipedia.org/wiki/反码)或[补码](https://zh.wikipedia.org/wiki/补码)形式，此时msb用来表示符号，msb为1表示[负数](https://zh.wikipedia.org/wiki/负数)，0表示[正数](https://zh.wikipedia.org/wiki/正数)。



### Multi-armed bandit problem

一个赌徒，要去摇老虎机，走进赌场一看，一排老虎机，外表一模一样，但是每个老虎机吐钱的概率可不一样，他不知道每个老虎机吐钱的概率分布是什么，那么每次该选择哪个老虎机可以做到最大化收益呢？这就是**多臂赌博机问题 (Multi-armed bandit problem, K- or N-armed bandit problem, MAB**



### Knowledge Distillation 知识蒸馏

知识蒸馏指的是模型压缩的思想，通过一步一步地使用一个较大的已经训练好的网络去教导一个较小的网络确切地去做什么



背景：高性能的深度学习网络通常是计算型和参数密集型的，难以应用于资源受限的边缘设备. 为了能够在低资源设备上运行深度学习模型，需要研发高效的小规模网络. 

知识蒸馏是获取高效小规模网络的一种新兴方法， 其主要思想是将学习能力强的复杂教师模型中的“知识”迁移到简单的学生模型中. 同时，它通过神经网络的互学习、自学习等优化策略和无标签、跨模态等数据资源对模型的性能增强也具有显著的效果.

知识蒸馏是一种教师-学生(Teacher-Student)训练结构，通常是已训练好的教师模型提供知识，学生模型通过蒸馏训练来获取教师的知识. 它可以以轻微的性能损失为代价将复杂教师模型的知识迁移 到简单的学生模型中



### **Bayesian Optimization** 贝叶斯优化

[用简单术语让你看到贝叶斯优化之美](https://www.jiqizhixin.com/articles/2020-10-05-2)

[一文看懂贝叶斯优化/Bayesian Optimization](https://cloud.tencent.com/developer/article/1810524)

Bayesian optimization,
which acts as a very effective global optimization algorithm, has been widely applied in designing problems. By structuring the probabilistic surrogate model and the acquisition function appropriately, Bayesian optimization framework can guarantee to obtain the optimal solution under a few numbers of function evaluations, thus it is very suitable to solve the extremely complex optimization
problems in which their objective functions could not be expressed, or the functions are non-convex, multimodal and computational expensive



**假设我们把acquisition function简写为aq(x)，整个贝叶斯优化可以概括为：**

1.  基于Gaussian Process，初始化替代函数的先验分布；
2. 根据当前替代函数的先验分布和aq(x)，采样若干个数据点。
3. 根据采样的x得到目标函数c(x)的新值。
4. 根据新的数据，更新替代函数的先验分布。 并开始重复迭代2-4步。
5. 迭代之后，根据当前的Gaussian Process找到全局最优解。

**由于目标函数无法/不太好 优化，那我们找一个替代函数来优化，为了找到当前目标函数的合适替代函数，赋予了替代函数概率分布，然后根据当前已有的先验知识，使用acquisition function来评估如何选择新的采样点（trade-off between exploration and exploitation），获得新的采样点之后更新先验知识，然后继续迭代找到最合适的替代函数，最终求得全局最优解。**

\1. **定义一种关于要优化的函数/替代函数的概率分布**，这种分布可以根据新获得的数据进行不断更新。我们常用的是Gaussian Process。

\2. **定义一种acquisition function。**这个函数帮助我们根据当前信息决定如何进行新的采样才能获得最大的信息增益，并最终找到全局最优。
