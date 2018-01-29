
<center>
#### **A Short Note on Support Vector Machine**

##### Wenbo

##### Jan-2018

</center>
**Introduction**

This is part of my self-note on understanding the derivation of support vector machine (SVM). This short note aims to answer the following question

-   What motivats the SVM
-   How we formulate the motivation into an mathematical optimization problem
-   How to simplify the original optimization problem and solve it by Quadratic Programming (QP)

This short note is based on the understanding of Prof. Hsuan-Tien Lin's Video on Machine Learning Techniques

**Motivation**

<center>
<img src="C:\Users\Wenbo%20Ma\Desktop\Capture.jpg" alt="Figure is from Prof. Hsuan-Tien Lin&#39;s Slides on ML Techniques" style="width:60.0%" />

</center>
Given the three plots above, there are three classifiers all can separate the positive and negative case. If you can choose one from the three, which one do you want? I guess most of the people will choose the third one. At least I will choose the third one because it is robust to small perturbation of the data. (Think of a scenario that x,y axis represents people's height and weight. It is highly possible that the data is noisy, say, sometimes you report your height as 184cm or 185cm. In this case a robust(stable) classifier will still make right prediction)

**Formulate the Question in Plain English**

To achieve such goal above, we have to define what is a good classifier. In SVM, we firstly define margin as the smallest distance between any point and the decision boundary. Then we could say a classifier is better than the other if the its margin is larger than the others' (Intuitively it means all points can at least perturbate the margin without change in prediction). Another requirement is the usual one that all prediction must be correct. Therefore we can formulate a maximization problem as follows:

<center>
**max**: margin

**s.t.** margin defined as minimum distance(sample point i, boundary line/hyperplane)

every sample is correctly classified
</center>
**Formulate/Translate the Question into Mathematics**

In order to solve the optimization problem using mathematics, we have to firstly translate it into mathematics (modeling).

Before that , we have to equip ourselves with some geometry knowledge.

-   A hyper plane in *R*<sup>*n*</sup> can be expressed as (w,b) where *w*<sup>*T*</sup>*x* + *b* = 0
-   A line perpendicular to this plane can be expressed as *w*<sup>*T*</sup>
-   The distance between a point *x*<sub>*i*</sub> in *r*<sup>*n*</sup> and the hyperplane can be expressed as $\\frac{1}{\\|w\\|} |w^{T}x\_{i}+b|$

Given sample point is denoted *x*<sub>*i*</sub> and class is *y*<sub>*i*</sub> ∈ { − 1, +1}, the above optimization problem can be translated as

<center>
max $min\_{i} \\frac{1}{\\|w\\|} |w^{T}x\_{i}+b|$

s.t. *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)&gt;0
</center>
The above problem can be simplified (and equivalent)

<center>
max $min\_{i} \\frac{1}{\\|w\\|} y\_{i}(w^{T}x\_{i}+b)$

s.t. *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)&gt;0
</center>
because if *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)&gt;0 holds, then *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)=|*y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)| = |*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*| given *y* ∈ { − 1, 1}

**Narrow our goal to Simplify the Question**

The above maxmium minimum problem is still complicated. By taking a closer look at the problem, we can find if (*w*<sup>\*</sup>, *b*<sup>\*</sup>) is a solution, then (*k**w*<sup>\*</sup>, *k**b*<sup>\*</sup>) is also the solution. (i.e. satisfying all constraints and achieves the same optimal value for target function). Therefore, we narrow our goal that we are only interested in the solution making *m**i**n*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)=1. To illustrate the logic, let's say we assume (*w*<sup>\*</sup>, *b*<sup>\*</sup>) is an optimal solution making *m**i**n*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)=*M*. By scaling *w*<sup>\*</sup>, *b*<sup>\*</sup> we can always have a new solution $\\frac{w^{\*}}{M}, \\frac{b^{\*}}{M}$ making *m**i**n*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)=1. Therefore the narrowed version only explores that sort of solution space. And consequently the narrowed version can be much simplified as

<center>
max $\\frac{1}{\\|w\\|}$

s.t. min *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)=1
</center>
which is equivalent to

<center>
min $\\frac{1}{2}{w^{T}w}$

s.t. min *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)=1
</center>
after the following treatment (all of them preserve the equivalence)

-   maximization becomes minimization reciprocal,
-   adding $\\frac{1}{2}$
-   minimize ∥*w*∥ is equivalent to minimize ∥*w*∥<sup>2</sup> = *w*<sup>*t*</sup>*w*

**Relaxation**

Unfortunately, the current optimization problem is still hard to solve (due to minimization in constraints). We therefore have to relax the constraint (remove minimization) to *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*<sub>*i*</sub> &gt; =1. To illustrate the equivalence of this relaxation. Assume under the new relaxed constraint, we achieve a solution *w*<sup>′</sup>, *b*<sup>′</sup> and all *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*<sub>*i*</sub> &gt; 1. By scaling invariance (illustrated above), we can easily get a new $(\\frac{w^{'}}{k},\\frac{b^{'}}{k})$ making a smaller $\\frac{1}{2}{\\|w\\|}$. Therefore *w*<sup>′</sup>, *b*<sup>′</sup> when all *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*<sub>*i*</sub> &gt; 1 cannot be an optimal solution. There has to be a *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*<sub>*i*</sub>)=1. Hence the relaxation is equivalent to the original problem.

**Solving the Probelm by Quadratic Programming**

Now current optimization problem after relaxation is stated as follows:

<center>
min $\\frac{1}{2}{w^{T}w}$

s.t. *y*<sub>*i*</sub>(*w*<sup>*T*</sup>*x*<sub>*i*</sub> + *b*)&gt; = 1
</center>
The above is a Quadratic Programming problem which can be easily solved with a QP solver.

<center>
<img src="C:\Users\Wenbo%20Ma\Downloads\QPLin.JPG" alt="Figure is from Prof. Hsuan-Tien Lin&#39;s Slides on ML Techniques" style="width:50.0%" />

</center>
