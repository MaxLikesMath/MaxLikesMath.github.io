# An Information Bottleneck Perspective on Latent Bootstrapping

## Introduction
In many ways, strong self-supervised learning algorithms are the "holy grail" of modern ML. With unlabelled data being cheap and readily available, being able to effectively utilize it to create effective models for fintetuning on downstream tasks is a great way of creating cost-efficient, deployment-ready models. From a business perspective, it is clear why this is important. Labelling data can be expensive, so the less data you need to label the better. However, this is not the only reason they are important. They belong to a paradigm of learning which needs to be exploited if we wish to achieve human level performance across a wide-variety of tasks, and will almost certainly be a key component in the quest for artificial general intelligence.

SSL really rose to prominence in natural language processing, where there exists a natural *pretext task* for a model to learn from. Letting $$f_{\theta}$$ be a model, given a sequence of words $$s = [w_1, \..., w_n ]$$, we create a new sequence $$s^\prime$$ by removing a random word $$w_k$$ from the sequence $$s$$. We then train $$f_\theta$$ by having it make a prediction $$f_\theta(s^\prime) = w^\prime_k$$ about the target word $$w_k$$. If we assume our model knows a fixed amount of words (or word pieces) $$W$$, we can view this as a $$|W|$$-class classification problem, (where $$|W|$$ is the number of words our model knows) and train $$f_\theta$$ by minimizing the cross-entropy loss. 

For problems in the vision domain, there is not some immediately obvious pretext task that dominates all others. Many pretext tasks have been attempted in vision, with many achieving very strong results (for an overview of these methods, the blog post [Self-Supervised Representation Learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#contrastive-predictive-coding) is excellent). Most of the SOTA methods previously were so called "contrastive methods", which functioned by reducing the distance in latent space between transformed views of the same image (called positive pairs), while increasing the distance between augmented views of different images (negative pairs). Intuitively, this makes sense, and feels like a natural pretext task to use. Complications do arise though. Firstly, dealing with negative pairs can be difficult and they generally require either special mining techniques to find good pairs, or some type of memory mechanism to function properly. Second, such techniques tend to be sensitive to the choice of augmentations.

Coming out near the end of 2020, the paper [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf) showed that one could do away with negative samples, and achieve SOTA scores across a wide variety of tasks. Furthermore, their method was more robust towards different augmentation techniques than previous SOTA methods. Following the success of BYOL, it was shown in [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566), that the methodology used in BYOL could actually be simplified further and still give competitive results. Extending and understanding these "Latent Bootstrapping" methods has very quickly become a hot topic in many research communities in ML, since they provide a relatively simple method for learning from large, unstructured unlabeled data. Here, we will introduce these methods and explore, from a mathematical view, how and why they work.

## The Basics of Latent Bootstrapping
Suppose we have a feature space $$X$$, which we can consider as a random variable with probability distribution $$P(X)$$. Now, let $$T_1, T_2$$ be two (possibly equivalent) sets of transformations on elements of our feature space. That is, for $$\tau \in T_i$$, $$\tau$$ is a function $$\tau : X \to X$$. We then describe two probability distributions $$D(T_1), D(T_2)$$ from which we sample such transformations. 

Next let $$f_\theta$$ be the model we would like to learn useful representations of our data. We then take another model (usually the same architecture) $$f_\zeta$$ to form a pair of models $$(f_\theta, f_\zeta)$$. Attached to each model are *projection networks* $$(g_\theta, g_\zeta)$$, with a *prediction network* $$h_\theta$$ attached to $$g_\theta$$.

### SimSiam and BYOL
During training, given a datapoint $$x$$ we sample two transformations $$\tau_1, \tau_2$$ according to distributions $$D(T_1)$$ and $$D(T_2)$$ respectively. We then produce two transformed views of $$x$$ given by $$\tau_1(x), \tau_2(x)$$. Given these two augmented views, we compute the latent representations $$r_1 = f_\theta(\tau_1(x))$$ and $$r_2 = f_\zeta(\tau_2(x))$$ which get passed into the projection networks giving $$z_1 = g_\theta(r_1)$$ and $$z_2 = g_\zeta(r_2)$$. Finally we compute the prediction $$y = h_\theta(z_1)$$. In order to simplify notation, we will denote the function composition $$h_
\theta \circ g_\theta \circ f_\theta$$ as $$M_\theta$$, which we will call the *online network*, and $$g_\zeta \circ f_\zeta$$ as $$N_\zeta$$, referred to as the *target network*.

Now, recall that we mentioned that in contrastive learning methods, when learning with positive pairs, we try to minimize the distance in latent space between augmented views of the same image. This is also the goal of latent bootstrapping methods, so we must use a loss function during training which minimizes this distance. We give this loss as the symmetric MSE:

$$L_{\theta, \zeta}(\tau_1(x), \tau_2(x)) = \| M_\theta(\tau_1(x)) - N_\zeta(\tau_2(x)) \|_2^2 + \| M_\theta(\tau_2(x)) - N_\zeta(\tau_1(x)) \|_2^2$$

It is worth noting that in practice, we generally normalize the outputs as $$\hat{M}(x) = \frac{M(x)}{\|M(x)\|_2}$$ before passing them into the loss function.

Where SimSiam and BYOL differ is in their treatment of the target network $$N_\zeta$$. In BYOL, they used a *momentum encoding* method for the weights of $$\zeta$$. That is, during training, we update the weights of the online model normally:

$$\theta \leftarrow \text{optimizer}(\theta, \nabla_\theta L^{\text{sg}}_{\theta, \zeta})$$

The loss term however is modified with the *stopgrad* operator:

$$L^{\text{sg}}_{\theta, \zeta}(\tau_1(x), \tau_2(x)) = \| M_\theta(\tau_1(x)) - \text{sg}(N_\zeta(\tau_2(x))) \|_2^2 + \| M_\theta(\tau_2(x)) - \text{sg}(N_\zeta(\tau_1(x))) \|_2^2$$

We then update the weights of the target network as an exponential moving average:

$$\zeta \leftarrow \lambda \zeta + (1-\lambda)\theta$$

where $$\lambda$$ is the decay rate.

In SimSiam they observe that it is sufficient to update the weights of the target network simply as $$\zeta \leftarrow \theta$$ as long as the stopgrad operator is used in the loss. If one is not familiar with the idea of the stopgrad operator, it may not be clear what it is doing, or what the purpose of it is. In short, the stopgrad operator lets us update our weights without considering outputs of the target network to aid in avoiding trivial (collapsed) solutions. 

Let us consider an example of such a collapsed solution. Notice that $$\theta$$ is an optimal solution to our loss if $$M_\theta(x) = c = N_\theta(x)$$ for all $$x$$ where $$c$$ is some constant vector. In [SimSiam](https://arxiv.org/abs/2011.10566) they observe that if one trains without stopgrad, that the standard deviation of the output channels of $$f_\theta$$ quickly collapses to zero, implying that the network converges on a collapsed solution.

## Entropy, Mutual Information, and the Information Bottleneck

These latent bootstrapping methods can be thought of as an unsupervised method for finding good representations of our data. But what is a good representation? In [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/pdf/1503.02406.pdf), a criteria for a good representation is proposed based upon the idea of mutual information. Before we can discuss this idea though, we will first go over some basic ideas from information theory.

First, we consider the *entropy* $$H(X)$$ of a discrete random variable $$X$$, which is defined as 

$$H(X) = \mathbb{E}[-\ln(P(X))]$$

$$= -\sum_{i=0}^n p(x_i) \ln(p(x_i))$$

The entropy of a random variable effectively measures how "surprising" or "informative" outcomes obtained by sampling $$X$$ are, and can be thought of as the average measure of information about a distribution obtained by sampling that distribution. Given that entropy is defined in terms of probability distributions, it can be naturally extended to the conditional case:

$$H(Y|X) = -\sum_{i,j} p(y_i, x_j) \ln(\frac{p(y_i, x_j)}{p(x_j)})$$

and the joint case:

$$H(X, Y) = -\sum_{i,j} p(x_i, y_j) \ln(p(x_i, y_j))$$

As with probability distributions, we also have a chain rule relating joint entropy and conditional entropy:

$$H(Y|X) = H(X,Y) - H(X)$$

Entropy can then be used to define the *mutual information* between two random variables, defined as:

$$I(X ; Y) = H(X) - H(X|Y)$$

The mutual information between two random variables effectively tells us how much we can learn about one random variable by sampling from the other.

The restriction to the discrete case is common in ML literature, since the idea of [entropy for continuous distributions](https://en.wikipedia.org/wiki/Limiting_density_of_discrete_points) is not a direct generalization of entropy for discrete distributions, and due to the discrete nature of computation, we are in effect usually dealing with discrete distributions.

We can now introduce the [Information Bottleneck Method](https://www.cs.huji.ac.il/labs/learning/Papers/allerton.pdf). If we think about general supervised learning tasks, we usually have a feature space $$X$$ and some set of labels $$Y$$, and some joint probability distribution $$P(X,Y)$$ which we wish to learn from some finite sample generated according to $$P(X,Y)$$.

So given some sample $$x$$ there is some minimum amount of relevant information contained in $$x$$ which can be used to predict the corresponding label $$y$$. Notice that this is exactly the mutual information between the distributions on $$X$$ and $$Y$$, $$I(X ; Y)$$. The distribution on $$Y$$ thus determines the relevant and irrelevant features in data sampled from $$X$$. An optimal representation for $$X$$ (according to $$Y$$) would thus be one which compressed data points in $$X$$ to some minimal set of features, and ignored any noise caused by irrelevant signals in the input.

Let us define a parametrized representation mapping $$f_\theta : X \to V$$ where $$V$$ is some "representation space" (which in deep learning is almost always some vector space). The probability distribution on $$X$$ thus determines a distribution on $$V$$. We can then define a representation of $$X$$ on $$V$$ as optimal if $$I(X, Y) = I(f_\theta(X), Y)$$ and $$I(f_\theta(X), X)$$ is minimal. If we think about this as an optimization problem, we can formulate this as minimizing the following Lagrangian:

$$\mathcal{L}[p(f_\theta(x)|x)] = I(X;f_\theta(X)) - \beta I(f_\theta(X);Y)$$

where $$\beta$$ is a positive Lagrange multiplier which acts as a tradeoff between how complex the representation is, and how much information is preserved. As we will show, this view of learning representations can shed some light on what is going on when perform latent bootstrapping.

## The Information Bottleneck in Latent Bootstrapping

In the recent paper [Barlow Twins: Self-Supervised Learning Via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf) , they show that their method effectively acts by minimizing a specific instance of the information bottleneck objective. As we will show, latent bootstrapping methods do something similar. Let $$V_\theta$$ be the random variable representing the latent representation of our online model, and let $$\hat{V}_\zeta$$ be the latent representation of our target model. We will now prove the following theorem:

**Theorem:** Latent bootstrapping methods minimize the value

$$\mathcal{L}B_\theta = (1-\beta)H(V_\theta) + \beta H(V_\theta | \hat{V}_\zeta) - H(T)$$

where $$\beta$$ determines the tradeoff between preserved information and transformation invariance, and $$H(T)$$ is a constant determined by the choice of possible transformations.

**Proof:** In latent bootstrapping, notice we can can create a version of the information bottleneck for our problem as

$$\mathcal{L}B_\theta = I(X ; V_\theta) - \beta I(V_\theta ; \hat{V}_\zeta)$$

This can be seen since our goal is that given an input image $$x$$, we generate two augmented views of $$x$$, $$\tau_1(x)$$ and $$\tau_2(x)$$. We then attempt to predict the latent representation of $$\tau_2(x)$$, $$N_\zeta(\tau_2(x)) = \hat{v}_\zeta$$ of our target network from the augmentation $$\tau_1(x)$$ using our online network $$M_\theta$$. Taking our latent representation of our predictor as $$f_\theta(\tau_1(x))$$ gives us the above form of the bottleneck.

Using the definition of mutual information we then get

$$\mathcal{L}B_\theta = H(V_\theta) - H(V_\theta|X) - \beta(H(V_\theta)-H(V_\theta|\hat{V}_\zeta))$$

Now, from the chain rule for entropy know that 

$$H(V_\theta|X) = H(X, V_\theta) - H(X)$$
 
By the definition of joint entropy we know 

$$H(X, V_\theta) = \sum_{i,j} p(x_i, v_{\theta,j}) \ln(p(x_i, v_{\theta,j}))$$

The chain rule for probability distributions then tells us that

$$p(x_i, v_{\theta,j}) = p(v_{\theta,j}|x)p(x)$$

The value of $$v_\theta$$ depends on the values of $$x, \tau_1$$, and $$\theta$$. Since $$\theta$$ is always known, we have that the value $$p(v_{\theta,j}|x)=d(\tau_1)$$, the probability of the transformation $$\tau_1$$. This tells us that

$$p(x_i, v_{\theta,j}) = d(\tau_1)p(x)$$

Notice that this is the probability of getting any transformed sample. We now have that that

$$H(X, V_\theta) = H(T) + H(X)$$

This then implies that

$$H(V_\theta|X) = (H(T) + H(X)) - H(X) = H(T)$$

So we have

$$\mathcal{L}B_\theta = H(V_\theta) - H(T) - \beta(H(V_\theta)-H(V_\theta|\hat{V}_\zeta))$$

it then follows straightforwardly that

$$\mathcal{L}B_\theta = (1-\beta)H(V_\theta) + \beta H(V_\theta | \hat{V}_\zeta) - H(T)$$

as claimed.
 $$\square$$

Let's think about if the above theorem actually makes sense, and examine what it tells us. An important aspect of latent bootstrapping methods is that they are built in such a way that the learned representations will be invariant to the set of transformations we selected. This gives rise to the $$\beta H(V_\theta | \hat{V}_\zeta)$$ term, which is encourages our network towards invariant representations. Notice as well that we get a constant term $$H(T)$$ based on our set of transformations. This tells us that our choice of transformations (and the distribution over them) directly impacts the minimum value our Lagrangian can obtain. The last term is the $$(1-\beta)H(V_\theta)$$ which attempts to minimize the information contained in the latent representation. Notice as well that this actually captures the phenomena of collapsed solutions, and shows that during optimization, steps must be taken to avoid this collapse. This is because we can see that given any $$\beta$$ and any set of transformations $$T$$, the Lagrangian admits optimal solutions when $$H(V_\theta | \hat{V}_\zeta)$$ and $$H(V_\theta)$$ are both zero, meaning that the information content of our latent representations is zero. However, as observed in [BYOL](https://arxiv.org/pdf/2006.07733.pdf), given an appropriate weight update and loss function, such collapsed representations tend to be unstable.

In the [Barlow Twins Paper](https://arxiv.org/abs/2103.03230), they propose a Siamese network architecture with a special loss function which allows them to avoid using the stopgrad operator. They also prove that their loss function is related to the information bottleneck principle. The theorem given here can be seen as a generalization of this idea, showing that any latent bootstrapping method minimizes a similar Lagrangian. It also tells us that to create new latent bootstrapping methods, we must find a system which minimizes the Lagrangian but avoids collapsed representations.

These insights may not be practically useful, but hopefully you at least found them interesting.

Thanks for reading.

Max
