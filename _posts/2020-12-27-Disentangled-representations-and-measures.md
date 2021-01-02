# Feature Representations, Circuits, and Abstract Nonsense.
Currently, two of the biggest areas of research in the realm of deep learning are [representation learning](https://www.deeplearningbook.org/contents/representation.html) and [interpretablity](https://christophm.github.io/interpretable-ml-book/). Loosely speaking, representation learning (applied to deep learning) is the study of the behaviour of *feature representations* in neural networks, and how to design systems which learn more effective representations. Interpretability on the other hand can be seen as attempting to understand exactly what leads an individual network to make decisions about samples in some dataset. While these two are distinct in many ways, the tools used in interpretability tend to be very useful for examining ideas in representation learning in a more "empirical" way.

This application of interpretability to empirically studying representation learning in neural networks has resulted in many interesting discoveries. One of the most interesting things to come out of this paradigm is the idea of *circuits*. Circuits  (originally presented by [Olah et al.](https://distill.pub/2020/circuits/zoom-in/#claim-2)) can be thought of as essentially subnetworks within a network which describe how features in earlier layers interact to create more abstract features in deeper layers. Due to the complicated nature of neural network dynamics, studying such interactions empirically is the natural approach.

While a rigorous, elegant mathematical theory of circuits would be nice, it seems that the number of moving components in such systems means any such theory which captures all the dynamics of circuits rigorously would be rather complex.

This does not mean however that we can not reason abstractly about them! In this post, we will create a slightly more mathematical way of thinking about circuits. This is not meant to be a formal, rigorous theory and should be taken with a grain of salt, however, it does highlight both potential routes for developing a more formalized theory, and why such a theory in full generality might still depend on some sort of inductive bias placed upon the system.


## Approaching Disentangled Representations Theoretically
In order to discuss the problem, we want to think formally about what an image is. A greyscale image can be thought of as a compactly supported continuous function $f : \mathbb{Z}^2 \to \mathbb{R}$. We can then think abstractly about the feature space of greyscale images as the space of compactly supported continuous functions $C_c(\mathbb{Z}^2, \mathbb{R})$ which we simply write as $C_c(\mathbb{Z}^2)$. We can then generalize this to stacks of feature maps of arbitrary depth using the [tensor product](https://en.wikipedia.org/wiki/Tensor_product). Letting $V$ be some $c$ dimensional vector space, the space of feature maps of depth $c$ is given via the tensor product $C_c(\mathbb{Z}^2) \otimes_\mathbb{R} V$.

The space $C_c(\mathbb{Z}^2) \otimes_\mathbb{R} V$ has the structure of a $C_c(\mathbb{Z}^2)$-(bi-)[module](https://en.wikipedia.org/wiki/Module_(mathematics)) (for those unfamiliar, a module is a more general version of a vector space). The elements $\vec{f} \in C_c(\mathbb{Z}^2) \otimes_\mathbb{R} V$ have the form:
$$
\vec{f} = \sum_i^c f_i \otimes_\mathbb{R} \vec{e}_i
$$
$$
= \left[ {\begin{array}{c}
   f_1 \\
   \vdots \\
   f_c \\
  \end{array} } \right]
$$
where $\vec{e}_i$ is a basis vector. Now since the ring of scalars of this space is $C_c(\mathbb{Z}^2)$, scalar multiplication is given by
$$
\alpha \vec{f} = \left[ {\begin{array}{c}
   \alpha*f_1 \\
   \vdots \\
   \alpha*f_c \\
  \end{array} } \right]
$$
where $*$ is the convolution and $\alpha \in C_c(\mathbb{Z}^2)$. This means the the action via translations and multiplication by scalars in $\mathbb{R}$ are both well defined on the space. This can be seen by simply defining a translation as the function $t$ which takes the desired translation (which is an element of $\mathbb{Z}^2$) to the value $1$. Convolution with $t$ then defines the desired translation. Furthermore, $\mathbb{R}$ scalar multiplication is given by a function taking the trivial action (the point (0,0)) in $\mathbb{Z}^2$ to the scalar multiple $s$. Convolution with functions of this type is how scalar multiplication is performed.

Given this abstract interpretation of feature spaces in convolutional networks, we need a description of the actual layers which transform these feature spaces. The module viewpoint of these feature spaces actually provides us with arguably the most "natural" description of convolutional layers. Since convolutional layers are translation equivariant and $\mathbb{R}$-linear they are, by definition, $ C_c(\mathbb{Z}^2)$-[module homomorphisms](https://en.wikipedia.org/wiki/Module_homomorphism). That is, a convolutional layer $\Psi$ with $c$ input channels and $d$ output channels is an element of the space 
$$
\textit{Hom}_{C_c(\mathbb{Z}^2)}(C_c(\mathbb{Z}^2) \otimes_\mathbb{R} V, C_c(\mathbb{Z}^2) \otimes_\mathbb{R} W)
$$ 
where $V$ and $W$ are $c$ and $d$ dimensional vector spaces respectively. We now give a theorem which relates this idea of homomorphisms to the classical idea of convolutional layers as being described by a collection of filters:

**Theorem 1** :
$$
\textit{Hom}_{C_c(\mathbb{Z}^2)}(C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^c, C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^d)
$$ 
is isomorphic to 
$$
\bigoplus_{i=0}^d \textit{Hom}_{C_c(\mathbb{Z}^2)}(C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^c, C_c(\mathbb{Z}^2))
$$.

**Proof** :
Let $\Psi$ be an element of 
$$
\textit{Hom}_{C_c(\mathbb{Z}^2)}(C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^c, C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^d)
$$ 
which takes the stack of feature maps 
$$ 
\vec{f}_1 
$$ 
to 
$$ 
\vec{f}_2 
$$. 
We can write 
$$ 
\vec{f}_2 
$$ 
as the linear combination
$$
\vec{f}_2 = \sum_{j}^c f_{2,j} \otimes \vec{e}_j
$$
Now notice that every $f_{2,j}$ can be given by a function $\psi_j : C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^c \to C_c(\mathbb{Z}^2)$ so we can rewrite the above as
$$
\vec{f}_2 = \sum_{j}^d \psi_j(\vec{f}) \otimes \vec{e}_j
$$
Each $\psi_j$ defines a module homomorphism (in specific, it defines an element of the [dual module](https://en.wikipedia.org/wiki/Dual_module)). It then follows that since we have $d$ output channels, the map $\Psi$ is described by a collection $(\psi_1, \dots, \psi_d)$ which gives the desired result. $\square$


When discussing feature representations in neural networks, one of the things that tends to be focused on is how such networks respond to transformations of the input space. Most works consider transformations which are given via a group action on the space, such as [Higgins et al.](https://arxiv.org/abs/1812.02230). There are many reasons why this is both a natural and powerful paradigm. Such group actions have a relatively nice mathematical theory which can be exploited, and some transformations whose impact we might wish to understand/exploit are given by a group action. Sadly though, many useful transformations are not given by a natural group action on the space. We thus need a more general, well-defined type of transformation. Key to developing this is the space of [measurable functions](https://en.wikipedia.org/wiki/Measurable_function) from $C_c(\mathbb{Z}^2)$ to itself, which we will denote $M(C_c(\mathbb{Z}^2))$. This is a more general space of transformations than the space of module homomorphisms from $C_c(\mathbb{Z}^2)$ to itself since it allows for non-linear transformations of $C_c(\mathbb{Z}^2)$. We can generalize this space of transformations beyond the single channel case using the direct sum, so for $C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^c$ the space of these transformations is $\Mu^c = \bigoplus_{i}^c M(C_c(\mathbb{Z}^2))$. Now, the application of $\Lambda \in \Mu^c$ to some $\vec{f} \in C_c(\mathbb{Z}^2) \otimes_\mathbb{R} \mathbb{R}^c$ is given in a pointwise fashion. That is, given that $\Lambda$ is defined by a collection of measurable functions $(\lambda_1, \dots, \lambda_c)$ we define the composition
$$
\Lambda \vec{f} = \left[ {\begin{array}{c}
   \lambda_1(f_1) \\
   \vdots \\
   \lambda_c(f_c) \\
  \end{array} } \right].
$$

## Filter Families
In [the original post on circuits](https://distill.pub/2020/circuits/zoom-in/#claim-2-curves) they define a *family* of filters in a layer to be "a collection of filters in a layer which detect small variations of the same thing". We will give a more mathematical definition of families using the idea of [small categories](https://en.wikipedia.org/wiki/Category_(mathematics)). A small category is essentially a generalization of a group which is defined as a set of functions which must contain the identity function, and must have a defined associative function composition between elements. However, it need not be closed, so we may have compositions of elements which is not in the small category. When quantifying if a filter "detects" its associated filter we usually discuss the value given by a non-linearity applied to the output of a filter. To avoid any confusion about when a filter does or does not detect its associated feature(s), we will give a mathematical definition.

** Definition 1 :**
We say a filter $\psi_i$ is $p$-activated with threshold $\delta$ if $\|\sigma(\psi_i(\vec{f})) \|_p > \delta$ where $\sigma$ is a non-linearity.

Now, the choice of $\delta$ depends on activation, so for simplicity we will consider our activation on the unit interval $[0,1]$, and set a threshold $\delta > 0.5$, and take the $p$-norm to be the $L_2$ norm. This captures the intuitive notion of a filter being activated. We can now define a family in a more mathematical way:

** Definition 2 : **
Let $\Gamma$ be a small category of transformations of the input. We call a collection of filters $\mathcal{F} = (\psi_p, \dots, \psi_q)$ a * $\Gamma$-family* if for all $\Lambda \in \Gamma$, if $\psi_i(\vec{f})$ activates for some $\psi_i \in \mathcal{F}$, then $\| \psi_i(\vec{f}) \|_2 \approx \| \psi_j(\Lambda \vec{f}) \|_2 + \psi_k(\Lambda \vec{f}) > \delta$ for some pair $\psi_j, \psi_k$, where one of the pair is allowed to be $0$.

This definition essentially says that the small category $\Gamma$ shifts the activation around the family $\mathcal{F}$. This phenomena is observed in the curve detectors of [Olah et al.](https://distill.pub/2020/circuits/zoom-in/#claim-2-curves) where transformations by the group $SO(2)$ (which is a small category) on input features shifted the activations within a family of curve detectors. It also captures the case of the invariant dog-head detector by defining $\mathcal{F}$ to be the invariant detector $\psi$, since $\| \psi(\vec{f}) \|_2 \approx \| \psi_j(\Lambda \vec{f}) \|_2$ for any $SO(2)$ transformation $\Lambda$. The definition given above thus captures the classic idea of group equivariance, but allows a more formal treatment of actions on the input space which are not group actions.


To see an example of this, consider the small category $\Gamma_C$ which is the set of color transformations $\{I, \text{RB}, \text{BR} \}$ where $I$ is the identity, $\text{RB}$ maps red to blue, and $\text{BR}$ maps blue to red. This set forms a group isomorphic to the cyclic group of order 3. However, its action on the space of of RGB images is not a group action, since it is not invertible. This can be seen since if we take an image which is half red and half blue, and apply either $RB$ or $BR$ to this image, there is no way to recover the original image using only the actions of the group.

Before continuing, we must highlight a small issue with working in such a general way. For any pair of filters, we can find a set of transformations $\Gamma$ which makes them a family. So, when considering families of filters, we want to consider families which are related by transformations which we deem "natural" in some way. For instance things like perspective shifts, changing in lighting, shifts in objects in an image (i.e replacing an adult with a child in an image).

## Circuits and Transformations
The definitions used for families of filters allows for a more mathematical discussion of the concept of *circuits*.

To discuss this, we first observe how convolution is actually calculated. For the each filter functional $\psi_i$, we have that
$$
\psi_i(\vec{f}) = psi_{i,1} * f_1 + \dots + \psi_{i,c}*f_c
$$
Let's consider now how circuits combine families, which will give us some insight into possible reasons why certain feature representations form. Suppose we have two sets of input channels in 
$$
\vec{f}
$$ 
which form families 
$$
\mathcal{F}_1
$$ 
and 
$$
\mathcal{F}_2
$$ with small categories 
$$
\Gamma_1
$$ 
and 
$$
\Gamma_2
$$
. Suppose then we wish to combine these two families in a circuit which maintains "equivariance" with respect to both small categories. Now, if we suppose 
$$
f_p \in \mathcal{F}_1
$$ 
and 
$$
f_q \in \mathcal{F}_2
$$
, to create a family of output features which forms a family which preserves equivariance under combinations of elements of the small categories, for every possible combination of 
$$
f_p
$$ 
and 
$$
f_q
$$ 
we must have a filter 
$$
\psi_{pq}
$$
such that
$$
\psi_{pq}(\vec{f}) = \psi_{pq, p} * f_p + \psi_{pq, q} * f_q
$$
where all channels 
$$
f_i
$$ 
which do not belong to either 
$$
\mathcal{F}_1
$$
or $$
\mathcal{F}_2
$$
are convolved against a zero filter (in practice, the other filters are not zero, which we will soon explain) In order to preserve the equivariance we also require that every 
$$
\psi_{pq, p}
$$ 
(and 
$$
\psi_{pq, q}
$$
) be related to each other under the corresponding small category of the input feature family.

This is essentially an ["equivariant-equivariant circuit"](https://distill.pub/2020/circuits/equivariance/). However, notice that as we have presented them, they are very parameter inefficient. For example, given two families, the number of filters required to maintain equivariance to both families through the circuit is $\| \mathcal{F}_1 \| \cdot \| \mathcal{F}_2 \|$, furthermore, each filter only has two active sub-filters.

This parameter inefficiency isn't just within these sort of circuits, but can also be seen to be related to the idea of "pure features". We can think of pure features as being filters which only activate if a specific feature is present in an image. In order for a filter $\psi_i$ to be pure, one would suspect that that only a subset of its subfilters would be non-zero (since subfilters with substantial weights for input features which do not strongly correlate to the presence of the pure feature would cause noise in the output.) In our circuit discussion, each filter in the filter family was a "pure feature", so discussing properties of pure features also gives insight into properties of circuits.

This parameter expense actually gives some insight into the formation of [polysemantic filters](https://distill.pub/2020/circuits/zoom-in/#claim-2). Consider two pure features given by filters $\psi_i$ and $\psi_j$ and suppose $\psi_i$ and $\psi_j$ have very low probability of both activating at the same spatial location in any given input (or even occurring together in one input). Notice that combining the two pure filters into one combined filter $\psi_k$ increases the parameter efficiency without losing much information. While $\psi_k$ outputs only a single channel, one can discern which pure feature was detected by either observing the activation pattern, or by examining the activations of other filters.

Something like this is indeed observed in [the original circuits post](https://distill.pub/2020/circuits/zoom-in/#claim-2) where they show a filter which detects cat heads, cat legs, and car hoods. One would expect that cat heads, cat legs, and car hoods tend to occur in different spatial locations in an image, and it is probably rare that car hoods and cats appear in the same image in most datasets. Furthermore cat legs and cat heads having the same filter actually makes sense. Indeed, legs and heads will almost always fall in a pattern which avoids much overlap between the two features, so having them activated within the same filter makes sense. This might also encourage more parameter efficiency since, for instance, both have fur, and sharing a subfilter which activates for the presence of the fur feature could be utilized more efficiently by the filter if used in the detection in more than one feature.

Before concluding, it is worth making some remarks about how things behave in practice. For instance, our assumptions that some subfilters are zero tends not to be true in practice. Usually, almost all subfilters in a filter have non-zero weights, though many of these tend to be very small. This could be due to simply noise during training, or possibly that some feature deeper in the network activates when a certain set of shallower filters have small activation values, or a number of other things. However, it doesn't cause any contradictions with the ideas presented here. It would be interesting to see what role such small activation values play in a network.

In conclusion, by presenting a slightly more formal way of thinking about very general transformations of inputs, we can discuss the observed properties of circuits and families in a slightly more mathematical way. While informal, it provides some useful tools for thinking about how features might potentially interact in a network.

Hopefully this read has been interesting!
Thanks,
Max

