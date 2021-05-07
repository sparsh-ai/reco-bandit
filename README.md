# RecoBandit
Building recommender Systems using contextual bandit methods to address cold-start issue and online real-time learning

## App 1
`Thompson Sampling`, `Single-user Multi-product Simulation`, `Multi-armed Bandit`

The objective of this app is to apply the bandit algorithms to recommendation problem under a simulated envrionment. Although in practice we would also use the real data, the complexity of the recommendation problem and the associated algorithmic challenges can already be revealed even in this simple setting.

[![RecoBandit - Thompson Sampling Simulation](https://img.youtube.com/vi/ND04S_0fRjs/0.jpg)](https://www.youtube.com/watch?v=ND04S_0fRjs)

Inspired by the following works:
- [Simulation](https://learnforeverlearn.com/bandits/)
- [Colab notebook](https://colab.research.google.com/github/yfletberliac/rlss-2019/blob/master/labs/MAB.RecoSystems.ipynb)
- [Blog post](https://peterroelants.github.io/posts/multi-armed-bandit-implementation/)
- [Blog post](https://dataorigami.net/blogs/napkin-folding/79031811-multi-armed-bandits)


## App 2
`Multi-user Multi-product Contextual Simulation`, `Contextual Bandit`, `Vowpal Wabbit`

The objective of this app is to apply the contextual bandit algorithms to recommendation problem under a simulated envrionment. The recommender agent is able to quickly adapt the changing bahavior of users and change the recommendation strategy accordingly.

[![VW Contextual Bandit Simulation](https://img.youtube.com/vi/9t0-FZIWMRQ/0.jpg)](https://www.youtube.com/watch?v=9t0-FZIWMRQ)

## App 3 (next release)
`Image Embeddings`, `Offline Learning`

The objective is to recommend products and adapt the model in real-time using user's feedback using Actor-critic algorithm. Suppose, we observed users’ behavior and acquired some products they clicked on. It is fed into the Actor Network which decides what we would like to read next. It produces an ideal product embedding. It can be compared with other product embeddings to find similarities. The most matching one will be recommended to the user. The Critic helps to judge the Actor and help it find out what is wrong.

Inspired by the following works:
- [Blog post](https://towardsdatascience.com/deep-reinforcement-learning-for-news-recommendation-part-1-architecture-5741b1a6ed56)

## App 4 (next release)
`Offline Learning`

The core intuition is that we couldn't just blindly apply RL algorithms in a production system out of the box. The learning period would be too costly. Instead, we need to leverage the vast amounts of offline training examples to make the algorithm perform as good as the current system before releasing into the online production environment. An agent is first given access to many offline training examples produced from a fixed policy. Then, they have access to the online system where they choose the actions.

Inspired by the following works:
- [Blog post](https://blog.insightdatascience.com/multi-armed-bandits-for-dynamic-movie-recommendations-5eb8f325ed1d)
- [Blog post](https://abhishek-maheshwarappa.medium.com/multi-arm-bandits-for-recommendations-and-a-b-testing-on-amazon-ratings-data-set-9f802f2c4073)
- [RecoGym](https://github.com/criteo-research/reco-gym)

![Offline then online](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bb18246a-c536-4cbc-919f-0d1108c9432b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210507%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210507T115825Z&X-Amz-Expires=86400&X-Amz-Signature=13fc58ac43e09545d5f334ca5c99a62cd9c83eee40f394b37b9fc53477f9ae82&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## What is Bandit based Recommendation?
Traditionally, the recommendation problem was considered as a simple classification or prediction problem; however, the sequential nature of the recommendation problem has been shown. Accordingly, it can be formulated as a Markov decision process (MDP) and reinforcement learning (RL) methods can be employed to solve it. In fact, recent advances in combining deep learning with traditional RL methods, i.e. deep reinforcement learning (DRL), has made it possible to apply RL to the recommendation problem with massive state and action spaces.

### Use case 1: Personalized recommendations

Goal: Quickly help users find products they would like to buy

In e-commerce and other digital domains, companies frequently want to offer personalised product recommendations to users. This is hard when you don’t yet know a lot about the customer, or you don’t understand what features of a product are pertinent. With limited information about what actions to take, what their payoffs will be, and limited resources to explore the competing actions that you can take, it is hard to know what to do.

### Use case 2: Online model evaluation

Goal: Compare and find the best performing recommender model

### Use case 3: Personalized re-ranking

Goal: Bring the most relevant option to the top

### Use case 4: Personalized feeds

Goal: Recommend a never-ending feed of items (news, products, images, music)

[https://youtu.be/CgGCbmlRI3o](https://youtu.be/CgGCbmlRI3o)

## References
1. [LinUCB Contextual News Recommendation](https://github.com/kilolgupta/Contextual-Bandits-for-News-Recommendaion)
2. [Experiment with Bandits](http://ethen8181.github.io/machine-learning/bandits/multi_armed_bandits.html)
3. [n-armed Bandit Recommender](https://learning.oreilly.com/library/view/reinforcement-learning-pocket/9781098101527/ch02.html)
4. Bandit Algorithms for Website Optimization [[eBook O’reilly](https://learning.oreilly.com/library/view/bandit-algorithms-for/9781449341565/)] [[GitHub](https://github.com/johnmyleswhite/BanditsBook)] [[Colab](https://nbviewer.jupyter.org/gist/sparsh-ai/b42056d45ca8238fe912baad036597a8)]
5. MAB Ranking [PyPi](https://pypi.org/project/mab-ranking/)
6. RecSim [GitHub](https://github.com/google-research/recsim), [Video](https://youtu.be/T6ZLpi65Bsc), [Medium](https://medium.com/dataseries/googles-recsim-is-an-open-source-simulation-framework-for-recommender-systems-9a802377acc2)
7. [https://vowpalwabbit.org/tutorials/contextual_bandits.html](https://vowpalwabbit.org/tutorials/contextual_bandits.html)
8. [https://github.com/sadighian/recommendation-gym](https://github.com/sadighian/recommendation-gym)
9. [https://learning.oreilly.com/library/view/reinforcement-learning-pocket/9781098101527/ch02.html](https://learning.oreilly.com/library/view/reinforcement-learning-pocket/9781098101527/ch02.html)
10. [https://github.com/awarebayes/RecNN/](https://github.com/awarebayes/RecNN/)
11. [https://vowpalwabbit.org/neurips2019/](https://vowpalwabbit.org/neurips2019/)
12. [https://github.com/criteo-research/reco-gym](https://github.com/criteo-research/reco-gym)
13. [https://pypi.org/project/SMPyBandits/](https://pypi.org/project/SMPyBandits/)
14. [https://github.com/bgalbraith/bandits](https://github.com/bgalbraith/bandits)
15. [https://pypi.org/project/mab-ranking/](https://pypi.org/project/mab-ranking/)
16. [https://www.optimizely.com/optimization-glossary/multi-armed-bandit/](https://www.optimizely.com/optimization-glossary/multi-armed-bandit/)
17. [https://abhishek-maheshwarappa.medium.com/multi-arm-bandits-for-recommendations-and-a-b-testing-on-amazon-ratings-data-set-9f802f2c4073](https://abhishek-maheshwarappa.medium.com/multi-arm-bandits-for-recommendations-and-a-b-testing-on-amazon-ratings-data-set-9f802f2c4073)
