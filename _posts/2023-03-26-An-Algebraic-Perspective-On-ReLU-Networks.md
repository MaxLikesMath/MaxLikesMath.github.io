# An Algebraic Perspective on ReLU Networks
## Introduction and Preliminaries
In this post, we are going to establish a relationship between the study of ReLU networks, geometry, abstract algebra, and logic. To do this, we give an algebraic formulation of how information flows through a ReLU network. We then show that this algebraic formulation can associate simplicial complices to inputs to the network. Finally we show that this algebraic formulation actually behaves like a type of logic.

### Motivation for this Post
The study of fully connected ReLU networks is one of the most explored areas of theoretical deep learning, with many interesting results coming out constantly, making it somewhat hard to keep up with what's-what in the field. However, one of the biggest results in the field is that of the so called ["*Spline Theory of Neural Networks*" ](https://proceedings.mlr.press/v80/balestriero18b/balestriero18b.pdf) which shows that a large class of neural network architectures effectively partition the input space with splines.This result gives a nice bit of insight into what neural networks are "doing", and allows for some intuitive thinking about these networks. 

While listening to an episode of Machine Learning Street Talk ([this episode I believe](https://open.spotify.com/episode/414NCcPr6mxRZU2FQgEHXR)) someone said that neural networks effectively are putting a structure of a simplicial complex on the input space in reference to this partition via splines. At the time I thought that it made intuitive sense. There is, for instance, a canonical map from a standard $$n-1$$ simplex onto a polytope with $$n$$ vertices, and convex polytopes can be defined by a system of linear inequalities. Since one can write a ReLU layer as a system of linear inequalities, we can give a natural correspondence between a ReLU layer and simplicial complex. I did a bit of digging to see if much had been done examining networks from the simplicial viewpoint more explicitly, but I didn't find much (though I did find [some work using simplices for universal approximation](https://eduph.github.io/ANNandSC/)).

 I thought it was a bit of a shame, as simplicial complexes are interesting, especially since they provide a natural gateway from geometry to algebra, and I like the abstraction that algebraic theories tend to provide. I did a bit of playing around with these ideas, I hit upon a bit of an interesting algebraic theory (that to) me gave some insight into how these networks operate that wasn't exactly captured in the spline theory work explicitly. Given I have a bit of free time in these cold winter months, I thought it might be interesting to share these cobbled together ideas somewhere, as they might be of interest to others. There's admittedly probably a much nicer way of getting this sort of structure, but the way in which it was derived uses relatively simple tools, so hopefully it is easy enough to understand.

In the following sections, we will derive an algebraic viewpoint of ReLU networks. We will then examine how this relates to simplicial complexes, and how it shows that ReLU networks actually perform a very general type of logic.

*Note*: While this post is a bit math heavy, I did try to keep it a bit informal at points for pedagogy.

### Preliminaries
I wanted to try and make this post accessible to those without a formal background in abstract algebra, as it's not a common part of the education of people in the machine learning world. However, some knowledge of some basic ideas from undergraduate abstract algebra is necessary due to the objects being used. Assuming readers are familiar with basic concepts of linear algebra, I will do my best to introduce the main objects of study here. These are **semigroups**, **rings**, and **semirings**.

We start with the idea of a [semigroup](https://en.wikipedia.org/wiki/Semigroup) . A semigroup is a set $$S$$ equipped with an associative binary operation $$\otimes : S \times S \to S$$. That is to say, if we let $$a,b,c \in S$$ then we have that $$a\otimes(b \otimes c) = (a\otimes b) \otimes c$$. An example of a semigroup that many people are familiar with is that of the set of possible strings over some finite alphabet $$\Sigma$$ where the binary operation is string concatenation. If a semigroup has an element $$1$$ (which we call the identity) such that $$1 \otimes a = a$$ for all $$a \in S$$ this semigroup is then called a [monoid](https://en.wikipedia.org/wiki/Monoid). Any semigroup can be turned into a monoid in the obvious way: simply add an extra element and define it to behave as the identity. A monoid where for every non-identity element $$a$$ there exists an element $$a^{-1}$$ such that $$aa^{-1} = 1$$ is called a [group](https://en.wikipedia.org/wiki/Group_(mathematics)). 

The idea of a [ring](https://en.wikipedia.org/wiki/Ring_(mathematics)) has more moving parts than that of a semigroup. A ring is a set $$R$$ equipped with two binary operations of addition and multiplication $$(+, \cdot)$$ with the following properties:

1. The addition operation $+$ is associative: $$a+(b+c) = (a+b)+c$$.
2. Addition is commutative: $$a+b = b+a$$.
3. There is an element $$0$$ (called the additive identity/zero) such that $$a+0 = a$$.
4. For every $a$ there is an element $$-a$$ such that $$a+(-a) = 0$$.
5. Multiplication is associative: $$(a \cdot b ) \cdot c = a \cdot (b  \cdot c)$$
6. There is an element $$1$$ (called the multiplicative identity) such that $$1 \cdot a = a$$.
7. Multiplication distributes over addition: $$a \cdot (b + c)= a \cdot b +a \cdot b$$.

The idea of a ring can also be generalized to the idea of a [semiring](https://en.wikipedia.org/wiki/Semiring). A semiring is the same as a ring, except without property 4. So, an example of a ring would be the set of integers $$\mathbb{Z}$$, while an example of a semiring would be the non-negative integers $$\mathbb{Z}^+$$.

With these preliminaries out of the way, we can start looking at the algebraic structure of neural networks.

## Giving Algebraic Structure to ReLU Networks
In this section we are going to construct an algebraic representation of ReLU networks. This does require a fair bit of symbolic manipulation, but the general end result is simple. It is obviously encouraged to read the section, but for those wishing to skim, the summary provides a TLDR version of what is covered.
### Basics of a ReLU Layer
 Let's consider some input space $$X$$ as an $$n$$-dimensional vector space. A single fully-connected ReLU layer is given by the equation 
 
 $$\mathcal{N}(x) = \sigma(Wx + b)$$
 
  where $$W$$ is the weight matrix, $$x$$ is the input, $$b$$ is a bias term, and $$\sigma(z) = \text{max}(z,0)$$ is the ReLU function, applied elementwise to the vector produced by $$Wx + b$$.  A layer can be broken down into its constituent parts (i.e its "neurons"), where the ith neuron can be defined by a function 
 
 $$\mathcal{N}_i(x) = \sigma(w_i \cdot x + b_i)$$
 
 where the $$w_i$$ is the $$i$$th row of the weight matrix $$W$$ and $$b_i$$ is the ith bias term. 

Borrowing from the aforementioned spline theory of deep learning, we can interpret each $$w_i$$ as a so called *feature template*, where the dot product produces a similarity measure of the input $$x$$ with this feature template. The bias term effectively works by setting the minimum activation threshold for the ReLU function. For our purposes, we are going to drop the bias term as leaving it in would add a fair bit of notational clutter, while not meaningfully impacting the results.

Throughout the following sections you will notice that instead of treating ReLU layers strictly as operations on vector spaces, we "abstract out" and consider more broadly how they operate on the information contained in their inputs. Let's think about why this type of abstraction makes sense. In this feature template view, the dot product of a feature template with the input gives information about the the strength of the feature in that input. This can be thought of as retrieving information from the input, which is then passed down through the network. 

### Parametrization and Growing Layer Widths
To get the type of abstraction we want, we start by considering how we define a neural network layer. The key component of a fully-connected layer is the weight matrix. Most people are familiar with matrices as essentially the "rectangles of numbers" you work with in linear algebra, but we can give a bit of a weirder version of them than that. We can take a matrix $W$ and relate to it a function $$W : \mathbb{N} \to \mathbb{R}^n$$ on (a subset of) the natural numbers which contains all of the relevant information about the matrix. This can be done rather simply by defining $$W(i) = w_i$$ where $$i$$ is a row in the matrix we wish to construct. This works for the normal case of finite matrices by considering such functions on only finite subsets of $$\mathbb{N}$$, and can be used to describe a matrix with infinite columns as well.

Given this, a fully-connected neural network layer is defined by such a function which we call a **parametrization scheme**.  We use this concept to define the action of "growing" a hidden layer. We start by selecting some upper bound on the possible width of the layer (we could allow it to be infinite if we wanted), and defining some parametrization scheme $$W : U \to \mathbb{R}^n$$ where $$U \subseteq \mathbb{N}$$ which defines the set of possible neurons. The ith neuron output can be described by the computation 

$$\mathcal{N}_i(x) = \sigma(W(i) \cdot x)$$

Letting $$\mathcal{N}_\text{i}$$
denote a neuron, let us define a concatenation operation $\otimes$ between the neurons. With this operation, the parametrization scheme defines a semigroup by simply treating the $$\mathcal{N}_{i}$$ as members of a (possibly infinite) alphabet. Let us denote this semigroup of as $S(\mathcal{N})$. Elements of this semigroup are objects of the form $$\bigotimes_{i \in K}^K \mathcal{N}_i$$ for some set of indices $K$. This operation of concatenation allows us to "grow" a hidden layer to some desired width. 

*Remark*:
Where it is clear, we will drop the concatenation symbol $$\otimes$$, and use Einstein-style notation: $$\bigotimes_{i \in K}^K\mathcal{N}_i = \mathcal{N}_K$$.

### Basic Properties of the Layer Semigroup

We can outline some properties of the layer semigroup by looking at the elements as representing pieces of information about the features encoded by their feature templates.

First, this semigroup is **idempotent**. By this, we mean that it contains non-trivial elements with the property that $$\mathcal{N}_i \mathcal{N}_i = \mathcal{N}_i$$.
Why is this? Well, consider what the value $$\mathcal{N}_i(x)$$ actually represents: the strength of the feature defined by the template $$W(i)$$. No extra information is encoded by concatenating an element with itself (since a single copy of $$\mathcal{N}_i$$ contains all the relevant information about the feature encoded by the parametrization scheme ). Thus we have that $$\mathcal{N}_i  \mathcal{N}_i = \mathcal{N}_i$$. This has an important consequence because it means that if we restrict ourselves to a semigroup $$S(\mathcal{N})$$ which has finitely many generating elements $$\{\mathcal{N}_i \text{|} i=1,...n\}$$ then the $$S(\mathcal{N})$$  must be finite.

Just to give a bit more justification for this, we can clearly construct a neural network where multiple neurons have the same weights. This however almost never happens in practice in modern deep learning. In fact, most weight initialization schemes are designed and used explicitly to avoid this situation. We will call such networks with identical neurons **degenerate**.  We will assume through much of this that networks we work with are non-degenerate. There is are however cases where weights are shared across locations in an input, such as convolutional layers. This can be handled by including with each $$\mathcal{N}_i$$ some extra location term (e.g, coordinates).

Next, we can take this semigroup to be **commutative**. That is, we can take $$\mathcal{N}_i  \mathcal{N}_j = \mathcal{N}_j \mathcal{N}_i$$. While in practice we consider the output of a neural network layer to have a canonical ordering for the ease of downstream computations, there is no reason to fix this ordering when thinking about such things in the abstract. In fact, it makes more sense to let the semigroup be commutative since the ordering of the outputs does not change the information content. This can be understood by simply considering a fully-connected ReLU network since it's rather easy to show that each permutation of the output of the network corresponds to some permutation of the weights in the network. Each permutation ultimately outputs the same information, just ordered differently.

Given the commutativity of the semigroup $$S(\mathcal{N})$$, if $$S(\mathcal{N})$$ represents a layer with finitely many generating elements, it has a unique maximal element. This can be seen by simply considering the element consisting of all possible symbols in $$S(\mathcal{N})$$. The fact that the semigroup is both commutative and idempotent means that this semigroup is technically a [semilattice](https://en.wikipedia.org/wiki/Semilattice).

We also have that for each $$\mathcal{N}_i$$ we have an **idempotent subsemigroup** which relates different scales of the parameter given by $$W(i)$$. The construction of this is rather straightforward. Suppose we have indices $$i,j$$ such that $$W(i) = cW(j)$$ where $$c$$ is a positive constant. It's then clear that $$\mathcal{N}_i$$ and $$\mathcal{N}_j$$ contain the same information, just at different scales. So the composition $$\mathcal{N}_i \mathcal{N}_j$$ contains no more information than either individual element. When defining these subsemigroups, we can make a choice of canonical operation between the scaled elements as either the minimum or maximum (under some norm on the vectors). So without loss of generality, we have $$\mathcal{N}_i \mathcal{N}_j = \mathcal{N}_i$$. For every $$i$$ this subsemigroup has at least one element, namely $$\mathcal{N}_i$$ itself.

We can actually make a stronger assertion than this. Let $$\mathcal{N}_K$$ be an arbitrary element of the semigroup and consider the value obtained by $$\mathcal{N}_K \otimes \mathcal{N}_j$$. If we have that $$j \in K$$, we have 

$$
\mathcal{N}_K \otimes \mathcal{N}_j = \mathcal{N}_K
$$

since the semigroup is commutative and idempotent. Consider then the set 

$$
G(A_K) = \{ \mathcal{N}_K \otimes \mathcal{N}_j \text{|} j \notin K \}
$$

and then let $$A_K$$ be the set generated by concatenating elements in $$G(A_K)$$. The set $$A_K$$ is then an [ideal](https://en.wikipedia.org/wiki/Semigroup#Subsemigroups_and_ideals) of the semigroup. A (right) ideal is defined as a subset $$A$$ of a semigroup $$S$$ such that for any $$s \in S$$ and any $$a \in A$$ we have $$as \in A$$. It is easy to see that $$A_K$$ is indeed and ideal. 

First, letting $$a$$ denote an element of $$A_k$$ we have that if $$j \in K$$ then $$a \otimes \mathcal{N}_j = a$$. If $$j \notin  K$$ then $$\mathcal{N}_K \otimes \mathcal{N}_j \in G(A_K)$$ so we have that $$a \otimes \mathcal{N}_j \in A_K$$. Since the $$\mathcal{N}_K$$ term will always behave idempotently, the ideal behaves as the (sub)semigroup with generating set $$\{\mathcal{N}_j \text{|} j \notin K \}$$.
Importantly, we can actually write this as a quotient $$\mathcal{N} / \mathcal{N}^{\lbrack K \rbrack}$$ with $$\mathcal{N}^{\lbrack K \rbrack}$$ being the subsemigroup generated by the $$\{\mathcal{N}_k \text{|} k \in K \}$$.
Being able to construct these ideals turns out to have important consequences in the structure of neural network feature spaces, as we shall see later.

### Computing with Layers and the Layer Semiring
For a network with some depth $d$, we can denote the semigroup of the $$l$$th layer as $$S(\mathcal{N}^l)$$. Given a layer and some input $$x$$, we can define a map, which we will refer to as the **valuation at x**, by the computation 

$$\mathcal{N}^l(x) = \sum_{i \in I}^I\mathcal{N}^l_i(x)N^l_i$$

 where the $$N^l_i$$ are treated as elements of the layer semigroup. What then would a map $$\mathcal{N}^{l+1}(\mathcal{N}^l(x))$$ look like?

First consider a single neuron $$\mathcal{N}^{l+1}_j((\mathcal{N}^l(x)))$$. We first compute the inner product, which we can write as

$$
\sum_{i \in I}^IW^{l+1}(j)_i\mathcal{N}^l_i(x)
$$

This can then be used to give us a term

$$
\sigma(\sum_{i \in I}^IW^{l+1}(j)_i\mathcal{N}^l_i(x)) 
( \mathcal{N}^l_{I})
$$

which is looks a lot like a monomial with variables $$\mathcal{N}^l_{I}$$. Before continuing, let's just take a moment to think intuitively about what this is saying. A neuron computes a similarity between its feature template and its input, and uses the ReLU function to set the feature activation to 0 if the template gives a negative match. The $$\mathcal{N}^l_{I}$$ term is essentially holding the input variables responsible for activating the neuron. We can also encode sparsity of feature templates into the output by letting $$I'$$ denote the set of indices such that the term $$W^{l+1}(j)_i = 0$$, and taking $$\mathcal{N}^l_{I/I'}$$ instead of $$\mathcal{N}^l_{I}$$ in the neuron output.

Using this construction of the output of an individual neuron we can give the output of $$l+1$$ layer as

$$
\mathcal{N}^{l+1}(\mathcal{N}^l(x)) = \sum_{j \in J}^J \sigma(\sum_{i \in I}^IW^{l+1}(j)_i\mathcal{N}^l_i(x)) 
(\mathcal{N}^l_{I})
$$

which, given that each neuron produces a monomial term, is effectively a polynomial.

There is one problem with this approach, which is that if our templates aren't sparse, the sum given by $$\mathcal{N}^{l+1}(\mathcal{N}^l(x))$$ can collapse into a single term under normal addition. In fact, even if we have sparse templates, if any two templates share the same non-zero entries, they will collapse into one term in this output polynomial, which makes applying subsequent layers according to the above definition not work properly. There is a rather simple way around this however, which is to add an extra term to each monomial:

$$
\mathcal{N}^{l+1}(\mathcal{N}^l(x)) = \sum_{j=1}^J \sigma(\sum_{i=1}^IW^{l+1}(j)_i\mathcal{N}^l_i(x)) 
( \mathcal{N}^l_I)\mathcal{N}^{l+1}_j
$$

This new term represents the neuron used in the computation. This can always be brought back to the representation in terms of only the input features by "modding out" the new terms. Furthermore, notice that we do not lose the structure of the layer semigroup by adding these new terms, since each monomial term can be thought of as belonging to a new semigroup where the new symbols have been added to the generating set.

This also allows gives us a natural shorthand for the output of the layer as

$$
\mathcal{N}^{l+1}(\mathcal{N}^l(x)) = \sum_{j=1}^J\mathcal{N}^{l+1}_j(x)N^{l+1}_j
$$

We will frequently work with the first hidden layer which takes input from some input space $$X$$, which we can write as

$$
\mathcal{N}^1(x) = \sum_{j \in J}^J \sigma(\sum_{i \in I}^IW^{1}(j)_ix_i) 
( \mathcal{X}_I)\mathcal{N}^1_j
$$

This tells us a few things. First, each feature in the output is encoded by a squarefree monomial. A squarefree monomial is a monomial where no variable appears more than once. This must be the case since the layer semigroup is idempotent.

Given the above, we have a semiring for the layer, which we are going to denote as $$\mathbb{R}^+_{\mathcal{I}}[\mathcal{N}^l]$$, but, to the annoyance of those familiar with rings, is not actually the same as the semiring of the positive real multiples of the semigroup (hence the subscript).
They are related, but this semiring is going to be a bit backwards, in that we are going to take classical addition as the multiplication operation of the semiring, and take our concatenation operation $$\otimes$$ as the addition operation within the semiring. It can easily be checked that by including the zero neuron in the layer semigroup to make it a monoid that we can indeed make this swap. This means that this semiring is what is known as an idempotent semiring, a fact that will become important later.

### Ideals of the Semiring
When talking about the layer semigroup, we showed that each element had an associated ideal. We have something similar for our semiring given above. There exists the notion of an [ideal of a ring](https://en.wikipedia.org/wiki/Ideal_(ring_theory)). An ideal of a ring is similar to an ideal of a semigroup. For a ring $$R$$, a subset $$A$$ is an ideal of $$R$$ if $$A$$ if the additive group $$(A, +)$$ is a subgroup of $$(R, +)$$, and if for any $$r \in R$$ we have that $$ra \in A$$ for all $$a$$. A weaker version of this is a subring, which is not closed under multiplication by elements of the ring, but only under elements of the sub-ring. Note though that we are not working with rings, but instead semirings, so instead of requiring an additive group structure on $$A$$, we weaken this to an additive semigroup structure. 

Here we restrict ourselves to considering networks with some upper bound on the width, so the layer semigroup is finite. It suffices to consider elements of the first hidden layer semialgebra 

$$
\mathcal{N}^1(x) = \sum_{j=1}^J \mathcal{N}^1_j(x)\mathcal{N}^1_j
$$

$$ = \sum_{j=1}^J \mathcal{N}^1_j(x) \mathcal{X}_{I_j}\mathcal{N}^1_j$$

where the $$I_j$$ are the index sets. We will show that every input has an associated (non-unique) subsemiring. 

Given $$\sum_{j \in J}^J \mathcal{N}^1_j(x) \mathcal{X}_{I_j}\mathcal{N}^1_j$$,
we can consider the new terms $$\mathcal{N}_j^1$$ as new elements of the generating set for the original layer semigroup,
so we can just let it be represented by some $$\mathcal{X}$$. Thus we can give a set $$\{\mathcal{X}_k \text{|} k \notin I_j \forall j \in J\}$$,
which is the set of all $$\mathcal{X}_k$$ which do not appear in any of the monomial terms of $$\mathcal{N}^1(x)$$. 
This set forms a (sub)semigroup, and we know from the basic properties of of our semigroup that this must have some maximal element $$\mathcal{X}_K$$. This element defines the semigroup ideal generated by $$\{\mathcal{X}_K \otimes \mathcal{X}_i \text{|} i \notin K \}$$,
which maps onto the semigroup generated by the $$\mathcal{X}_i$$ which we can recall is the quotient $$X/X^{[K]}$$. We now have that $$\sum_{j=1}^J \mathcal{N}^1_j(x) \mathcal{X}_{I_j}$$ is an element of the subsemiring $$\mathbb{R}^+_{\mathcal{I}}[X/X^{[K]}]$$.
This quotient effectively factors out the monomial terms from our semiring which do not appear as features present in the input $$x$$.

### Section Summary
The above results show that a ReLU network defines a semiring. This semiring consists of squarefree monomial terms. Each input is mapped onto an element of this semiring, meaning each input has an associated polynomial with squarefree monomial terms.

## Simplicial Complexes and The Geometric Structure of the Layer Semiring

### An Informal Introduction to Simplices
In the previous section we gave a bit of an algebraic structure to the internal mechanisms of a neural network. These structures have a geometric interpretation in the form of [simplicial complexes](https://en.wikipedia.org/wiki/Simplicial_complex#), but before we can show that, we must first give a bit of an explanation as to exactly what a simplicial complex is.

Anyone who has played any older 3D games will be familiar with objects constructed by gluing together different polygons. This is similar to the idea of a simplicial complex, which can be defined as an object constructed by gluing together [simplices](https://en.wikipedia.org/wiki/Simplex). So, what exactly is a simplex? Put simply, a simplex is an n-dimensional triangle. A 0-dimensional simplex is a point, a 1-dimensional simplex is a line, a 2-dimensional simplex is a triangle, a 3-dimensional simplex is a tetrahedron, and so on. There is a formal definition in terms of affinely independent points, but a more intuitive overview of constructing simplices will provide more meaningful insights into how our algebraic structure encodes some geometry.

So, suppose we have some abstract point $$x_1$$, we can consider the set $$\{x_1\}$$ as 0-simplex. Imagine then that we plop this point down at an arbitrary point in the 1-dimensional space $$\mathbb{R}$$. It really doesn't matter where on the number line we place $$x_1$$. Now, suppose we pick another point arbitrary point $$x_2$$, and we draw a line between them. This line is defined by the two points $$\{x_1, x_2 \}$$, which is a 1-simplex, and the points $$x_1$$ and $$x_2$$ are called the vertices of the simplex. Next, we can take this line and put it in the two dimensional space $$\mathbb{R}^2$$. We then select a third point $$x_3$$ which does not lie on the line defined by the endpoints $$\{x_1, x_2 \}$$. If we draw a line from $$x_1$$ to $$x_3$$ and $$x_2$$ to $$x_3$$ then we have a triangle $$\{x_1,x_2,x_3\}$$. This defines a 2-simplex containing all the points on the interior of the triangle. If we move up to $$\mathbb{R}^3$$, we can once again place the triangle somewhere within the space, and select a fourth point $$x_4$$ that does not lie on the same plane as the triangle defined by $$\{x_1,x_2,x_3\}$$.  We then draw lines from each vertex of the triangle $$\{x_1,x_2,x_3\}$$ to the new point $$x_4$$, we get a tetrahedron with vertices $$\{x_1,x_2,x_3, x_4\}$$ which defines a 3-simplex containing all the points in the interior of the tetrahedron. One can continue this type of simplex construction into arbitrary dimensions. Notice that each n-simplex has a border consisting of $$n-1$$ dimensional simplices. We call these the **faces** of the simplex. We can see that each face itself has faces, so sometimes one might refer to these subfaces as $$m$$-faces where $$m$$ is simply the dimension of the face.

### Constructing Simplicial Complexes

Given our definition of a simplex, let's consider a triangle $$\{x_1,x_2,x_3\}$$ in the plane, and let's attach a line from a new point $$x_4$$ (where $$x_4$$ is not in the interior of the triangle $$\{x_1,x_2,x_3\}$$) to $$x_1$$ which gives the 1-simplex $$\{x_4, x_1\}$$. Geometrically, this looks like a triangle with a horn, which we can write using our current notation as $$\{\{x_1,x_2,x_3\}, \{x_4, x_1 \} \}$$. We could also include another point $$x_5$$, which doesn't attach to anything, so we would get the geometric object consisting of our horned triangle and a single disjoint point, which in our above notation is  $$\{\{x_1,x_2,x_3\}, \{x_4, x_1 \} , \{x_5\}\}$$. Objects created in this fashion are  **simplicial complexes**.

There are two corresponding formal notions of simplicial complexes, **geometric simplicial complexes** and **abstract simplicial complexes**. An abstract simplicial complex is a collection of sets $$\Delta$$ such that for every $$X \in \Delta$$ if $$Y \subset X$$ then $$Y \in \Delta$$. In our example above, we did not include every subset of each simplex to avoid confusion. However, such sets still represent simplicial complexes since each simplex is maximal, so it is clear that we can simply include all possible $$m$$-faces of each simplex in the set to meet the formal definition.

A geometric simplicial complex is defined similarly as a set of simplices $$\Delta$$ such that every face of a simplex in $$\Delta$$ is also in $$\Delta$$, and that any non-empty intersection between simplices is a face of both simplices. From these two definitions it shouldn't  be too surprising that every geometric simplicial complex has a corresponding abstract simplicial complex, and that every abstract complex can be realized geometrically.

### Monomials and the Algebraic Form of Simplicial Complexes 

Suppose once again we have the set $$\{\{x_1,x_2,x_3\}, \{x_4, x_1 \} , \{x_5\}\}$$ which as we discussed above represents a simplicial complex $$\Delta$$ by its maximal simplices. We can represent this set by the polynomial

$$
P_\Delta = x_1x_2x_3 + x_4x_1 +x_5
$$
 
which consists of square free monomial terms. If we have some finite vertex set $$V = \{x_1, x_2, ..., x_n \}$$ every simplicial complex which can be constructed by combining these vertices can be represented by such polynomials consisting of only square-free monomial terms.

A polynomial like $$P_\Delta$$ coming from such a finite vertex set can be viewed as an element of the polynomial ring  $$\mathbb{R}[x_1, ..., x_n]$$. Now consider the ideal $$I_\Delta$$ generated by all square-free monomial terms which do not appear as faces of any of the terms of $$P_\Delta$$ (it is straightforward to check that this is indeed an ideal). We can then produce a quotient ring $$\mathbb{R}[x_1, ..., x_n]/I_\Delta$$ which contains the polynomial $$P_\Delta$$. This quotient ring has a particular name, called the [Stanley-Reisner Ring](https://en.wikipedia.org/wiki/Stanley%E2%80%93Reisner_ring), and the ideal $$I_\Delta$$ is known as the Stanely-Reisner ideal of $$\Delta$$. This ring is one of the objects that establishes the connection between the fields of polyhedral geometry and commutative combinatorial algebra. For a more in-depth overview of Stanley-Reisner theory, [this survey](https://math.okstate.edu/people/mermin/papers/A_survey_of_Stanley-Reisner_theory.pdf) serves as a good introduction for those with some experience in abstract algebra.

### Returning to Neural Networks
Returning back to considering some layer semiring. It's easy to see that this is not a Stanley-Reisner ring. Despite this, it still captures the structure of simplicial complex. As we saw earlier, an input $$x$$ is associated to some subsemiring which has a generating set factored by squarefree monomial terms that represent the features killed off by the ReLU function. The input $$x$$ is then associated to some polynomial consisting of squarefree monomial terms which do not get factored which defines a simplicial complex in a similar way to the Stanley-Reisner Ring. So, a neural network consisting of ReLU layers does, in fact, map inputs to simplicial complexes in a sense. An important thing to note though is that the layer semiring we have constructed does not necessarily capture the geometric or topological properties of the associated simplicial complex the way the Stanley-Reisner ring does. Rather, it just tells us that there is a way to associate some simplicial complex with each input.

## Neural Networks as Logic Machines
For those familiar with the idea of an [information algebra](https://en.wikipedia.org/wiki/Information_algebra) some of the ideas presented in the algebraic section might look familiar. One might think then that there is a sense in which the layer algebras defined above are in fact information algebras. Indeed, one can [induce an information algebra from a semiring](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=05c62d16d117cd9d4ba2355ae8e227b7fa8905ff) . However, we are not going to explore this directly here as we do not need to. Instead, we will show how neural networks encode a type of logic, and that this logic is effectively captured by the algebraic structure. We will start with a simple example for intuition, then provide the actual result.

### Encoding Boolean Logic in a Neural Network
In this section, we are going to encode a type of boolean logic within a neural network with ReLU layers. There are ways of encoding logic into a neural network by designing specific activation functions, such as functions with a learnable threshold, but it's not as straightforward to do this with a ReLU network. To see how we can approach this problem, start by letting input $$x$$ be some input of binary variables, let's suppose that we have a logical statement $$C_1$$ which  asks the question "does input $$x$$ belong to class 1?"

We might have some set of conditions that if $$x$$ meets, then the statement $$C_1$$ is true. We will denote logical "and" by $$\land$$ and "or" by $$\lor$$. So we might have something like $$P_1 \land (P_2 \lor P_3) \implies C_1$$ where the $$P$$ terms are some logical statements. We can consider that the truth values of each of the $$P$$ terms represents the presence of some feature in the input $$x$$.

Let's think about how to encode this sort of logic within a neural network. Consider the logical statement $$P_1$$. Let's take an example input vector 

$$ x =\begin{bmatrix}
           1 \\
           1 \\
           0 \\
           1
         \end{bmatrix}$$
         
and let's take $P_1$ to be the logical statement $x_1 \land x_2$. We then encode this statement as the vector

$$
P_1 =\begin{bmatrix}
           1 \\
           1 \\
           0 \\
           0
         \end{bmatrix}
$$

so we can get the truth value for $$P_1$$ by the equality $$P_1 \cdot x = 2$$. We can actually encode this into a ReLU layer by $$\sigma(P_1 \cdot x -1)$$, as one can see that if $$P_1$$ is not true then $$\sigma(P_1 \cdot x -1) = 0$$ and 1 otherwise.
We can generalize this to any $$\land$$ between any number of input variables by $$\sigma(P \cdot x + (1-\text{||} P \text{||}_1 ))$$
where the logical statement $$P$$ is encoded by vector with entries of $$0$$s and $$1$$s.

Next let's think of how to encode the negation $$\neg$$. For a logical statement $$P_2 = \neg x_3$$ this can be encoded in a straightforward way by taking

$$
P_2 =\begin{bmatrix}
           0 \\
           0 \\
           -1 \\
           0
         \end{bmatrix}
$$

and then computing the value $$\sigma(P_2 \cdot x +1)$$ since this will zero if $$x_3=1$$ and one if $$x_3=0$$. This same formula encodes any logical and of negations. For instance, if we change the value of $$P_2$$ to
 
$$
P_2 =\begin{bmatrix}
           -1 \\
           0 \\
           -1 \\
           0
         \end{bmatrix}
$$

then $$P_2$$ is true if and only if $$P_2 \cdot x = 0$$.

Now what about $$\land$$ statements which containing both negations and not? We can encode these as well. Since the negation can always be expressed as $$\sigma(P \cdot x +1)$$ if $$P$$ contains both negated and non-negated statements, the linearity of the dot product allows us to separate the positive and negative parts as

$$P =P^+ \cdot x + P^- \cdot x$$

and it is easy to see that

$$
P^+ \cdot x - ||P^+|| = 0
$$

implies that $P^+$ is true as a logical statement. Furthermore, we know that $P^-$ is true if and only if $P^- \cdot x = 0$, so $P$ is true if

$$
P^+ \cdot x - ||P^+||  + P^- \cdot x +  1 = 1
$$

which means we have the ReLU encoding as
 
$$
\sigma(P \cdot x +(1- ||P^+||))
$$

What about the or $$\lor$$? Can this be expressed in a similar way? Yes, but not as straightforwardly and in fact, it seems to require two layers to do so. One could encode such statements in a neural network using a version of ReLU which is bounded above, but this does not provide much insight into ReLU networks.

To get a $$\lor$$ from a ReLU network, it's useful to think about how to express a the logical or in different ways. From elementary predicate logic we know that we have

$$
\neg (\neg x_1 \land \neg x_2) = x_1 \lor x_2
$$

but encoding this operation is not possible with a single layer in our scheme. But, it is straightforward if we allow for a second layer. Let $$P$$ represent the logical statement $$\neg x_1 \land \neg x_2$$ which we know we can encode in a single neuron in a ReLU layer. We can also represent a negation with a single neuron. So we have that logical statement $$P$$ has its truth value encoded in the output of the first ReLU layer. Clearly then, we can just define a negation neuron in the second layer and compute $$\neg P$$, which is equivalent to the logical or.  One can get the exclusive or in a similar way. This means we can encode basic Boolean logic in ReLU networks.

### Neural Networks as Generalized Fuzzy Logic Machines
From the previous section we have that ReLU networks are capable of capturing Boolean logic, but this is not how they are actually used. Indeed, they seem to capture something more general. It seems like they are almost operating on a "dequantized" logic. One thing that can be somewhat overlooked is that logic is ultimately a game of sets. Given some set of predicates $$\{P_1, ..., P_n\}$$ which we assign to each some truth value, evaluating any logical statement is effectively just asking if that statement belongs to the set of "true" statements according to our logical rules, or if it does not and is thus false. This is quantized in the sense that logical statements take on binary values. It is well known though that logic can be generalized to notions of [fuzzy logic](https://en.wikipedia.org/wiki/Fuzzy_logic) which operates on [fuzzy sets](https://en.wikipedia.org/wiki/Fuzzy_set). Fuzzy sets are effectively sets where there is a "degree" of membership, where the degree is some value in the interval $$[0,1]$$, which is effectively a dequantized logic. However, neither Boolean nor fuzzy logic capture how ReLU networks process information.

However, there is a notion of [generalized fuzzy sets](https://arxiv.org/abs/1209.1718). To define these, one starts with what is called a "universe" of events $$\Omega$$, an idempotent semiring $$\mathcal{S}$$ and $$\mathcal{F}$$ the set of functions $$f : \Omega \to \mathcal{S}$$. The set $$\mathcal{F}$$ is also an idempotent semiring, and its elements are called *generalized fuzzy sets*.

Our ReLU networks are generalized fuzzy sets. We can consider the input space $$X$$ as the "universe", and we know that a ReLU layer maps its input to an idempotent semiring, so any stack of such layers maps from the input space to some idempotent semiring $$\mathbb{R}^+_{\mathcal{I}}[\mathcal{N}]$$. This means a ReLU network defines a function $$f : X \to \mathbb{R}^+_{\mathcal{I}}[\mathcal{N}]$$.

To get that $$\mathcal{F}$$ as being idempotent, we assume that all networks in $$\mathcal{F}$$ share the same parametrization scheme $$W$$, and differ in which neurons they choose to include in each layer. One way of thinking about this is that  $$\mathcal{F}$$ is the set of all subnetworks of some larger network. For any two elements of $$\mathcal{f}$$, they can be added and multiplied layerwise according to idempotent layer semiring. The idempotency then comes from the fact that they are both elements of some larger network which defines the idempotent semiring $$\mathcal{F}$$, which we know to be idempotent. Thus, our ReLU networks encode generalized fuzzy sets.

## Conclusion
Now, this all seems a little disjointed, but we can tie it all together here. Ultimately we have really just shown that ReLU networks do a type of generalized logical operations on the information contained in their input. This information also gives a geometric structure, similar to what one gets in the spline theory. The logical structure and the geometric structure both emerge from the same algebraic structure of the information being processed in a network.

Now whether or not any of this is useful is debatable, but I think it's kind of a nifty way to look at these systems. In a future post, we will try and derive some results from this algebraic framework.

Thanks,
Max
