# Explainable Reinforcement Learning through Behavior Cloning.

## Abstract
Reinforcement learning (RL) has emerged as a powerful paradigm for training
agents to make autonomous decisions in complex environments. However, the
lack of explainability in RL agents has been a significant hurdle in many real-
world applications. In this study, we present an approach to address this challenge
by combining Deep Reinforcement Learning (DRL) with Behavior Cloning (BC)
to produce an agent that is both performant and explainable simultaneously.
Initially, a conventional DRL agent is trained to optimize its policy through inter-
action with the environment. This process allows the agent to learn effective
strategies, but the resulting policy may lack interpretability. To enhance explain-
ability, we leverage the knowledge acquired by the non-explainable DRL agent
and employ it to train an interpretable behavior cloning agent.Experimental
results demonstrate that the proposed approach successfully achieves both
high performance and explainability. The BC agent exhibits proficient behav-
ior by inheriting the effective strategies learned by the DRL agent, while the
interpretable classifier ensures the ability to provide human-understandable
explanations for its actions. This combined approach presents a novel contribution
towards addressing the trade-off between performance and explainability in RL,
making it suitable for applications where trust, accountability and interpretability
are crucial.

## Introduction
Reinforcement learning (RL) has witnessed remarkable advancements in recent years,
enabling agents to learn and make decisions in complex environments without explicit programming. RL algorithms, such as deep reinforcement learning (DRL), have
achieved remarkable successes in various domains, including game playing [1], robotics
[2] and recommendation systems [3]. However, one significant challenge in deploying
RL agents in real-world applications lies in their lack of explainability, which hin-
ders their adoption in critical domains where interpretability and transparency are
essential.
Explainability refers to the ability to understand and provide comprehensible jus-
tifications for an agent’s decisions and actions. While DRL agents excel at learning
complex behaviors through trial-and-error interactions with the environment, they
often operate as black boxes, making it challenging to understand why they take spe-
cific actions or behave in a particular manner. Extensive lack of transparency could
lead to concerns regarding the reliability, trustworthiness and accountability of RL
agents in critical applications, such as healthcare, finance and autonomous vehicles.
To address this challenge, a novel approach that combines DRL with behavior
cloning (BC) to create agents that exhibit both high performance and explainability
simultaneously is proposed. Behavior cloning is a supervised learning technique that
involves training a model to imitate expert behavior by learning from their demon-
strations. By leveraging the knowledge acquired by a pre-trained DRL agent, the aim
of this work is to produce an agent that inherits the effective strategies while ensuring
interpretability through an explainable classifier.
Our objective is to demonstrate that the proposed approach can strike a balance
between performance and explainability, overcoming the traditional trade-off between
these two critical aspects in RL. By providing interpretable justifications for their
actions, our agents can address the growing need for transparency and accountability
in RL applications. Furthermore, the ability to understand an agent’s decision-making
process could enhance trust and facilitate collaboration between humans and RL
agents in various domains.
This paper, presents the methodology and experimental results that showcase the
effectiveness of our proposed approach. The performance and explainability of our
agents is evaluated by using standard RL and control benchmarks, highlighting the
advantages of the proposed hybrid approach over traditional DRL methods.


## Related Work
Explainable RL is essential for real-world applications where understanding an agent’s
decision-making process is crucial for trust, safety and deployment. This section
presents a review of related works that focus on explainability in RL and behavior
cloning techniques. Numerous studies have emphasized the importance of inter-
pretability in RL and have proposed methods for explaining RL agents’ behavior.
Many of these works utilize Cascading Decision Trees (CDT) [4] and Soft Decision
Trees (SDT) [5] to generate interpretable policies. CDT involve a sequential structure
of decision trees, where the output of one tree feeds into the next. This cascading
process aims to capture complex decision-making processes while maintaining trans-
parency. On the other hand, SDT utilize probability distributions at internal nodes,
enabling a soft and probabilistic interpretation of the RL agent’s behavior. In contrast,
this paper utilizes normal decision trees as a means of simplifying the interpretability
process, making the explanations more accessible and understandable. Instead of the
cascading or soft approach, the focus is on identifying the most important features
that drive the RL agent’s behavior. This enables the extraction of explicit rules and
conditions that can be easily comprehended by humans, thereby enhancing the overall
transparency and interpretability of the RL agent’s decision-making process.


## Methodology
The training of the DRL agent begins using the Proximal Policy Optimization (PPO)
algorithm, a state-of-the-art RL method known for its stability and sample efficiency.
The DRL agent is trained by interacting with the environment on the standard RL
API Gymnasium[6]. Three popular discrete environments from the RL and control
literature were chosen, namely CartPole[7], MountainCar[8] and Acrobot[9]. After the
DRL agent is trained, its expertise is utilized in order to create a behavior cloning
dataset. The dataset consisting of observation-action pairs is created by letting the
agent interact with the environment for 10000 steps. For each step, the corresponding
observation and the action chosen by the DRL agent are recorded.
Consequently, the BC agent is trained in a supervised manner using the behav-
ior cloning dataset. Furthermore, decision trees and logistic regression models are
employed as the classifiers for behavior cloning, as these models are considered
”white box” and offer substantial interpretability. The decision trees varying depth
is experimented upon, ranging from 1 to 4, in order to explore the trade-off between
performance and explainability.
To assess the performance and explainability of the behavior cloning agents, eval-
uations are conducted on the RL environments. The BC agents are evaluated over 100
episodes in each environment. The performance metrics, such as the average episode
reward, are compared with those of the original DRL agent. This evaluation allows
the understanding of the impact BC has on performance while considering the inter-
pretability of the BC agents at the same time. One of the advantages of utilizing
decision trees and logistic regression models is the availability of feature importances.
By analyzing these feature importances, insight into the decision-making process of the
behavior cloning agents is achieved. The initial training of the DRL agent allows it to
learn effective strategies in complex environments, while the BC agent, trained using
the learned behavior of the DRL agent, provides a transparent and interpretable model
for decision-making. This combination enables for both performance and explainability
in RL agents.
The methodology presented above forms the basis for our experiments, where the
performance and explainability of the proposed approach across multiple RL envi-
ronments is evaluated, by comparing the BC agents to the original DRL agent. The
results of these experiments provide insights into the effectiveness of our approach in
addressing the trade-off between performance and explainability in RL agents.

## Results
he results presented are obtained from our experiments evaluating the performance
and explainability of the proposed approach, which combines DRL with BC to achieve
simultaneous performance and explainability in RL agents. The performance of the
BC agents is compared with that of the original DRL agent and afterwards, the inter-
pretability of the BC agents is examined using decision trees and logistic regression
models.
Specifically, for each environment, we present a barplot of the performance of each
BC agent compared to the original DRL one. Additionally, we present a visualisation
of each Decision Tree and the feature importance for the tree with max depth equals
4, since it proved to be the most performant without sacrificing much in terms of
interpretability.

### CartPole

### MountainCar

### Acrobot

## Conclusion
In this study, the challenge of achieving simultaneous performance and explainability
in RL agents was addressed. By utilizing the data from DRL with BC in a form of
transfer learning, a novel approach that combines the advantages of high-performance
learning from DRL with the interpretability provided by ”white box” models was pro-
posed. The results demonstrate the effectiveness of this approach in generating RL
agents that exhibit both proficiency and transparency in decision-making. The evalu-
ation of the behavior cloning agents on classic RL environments, including CartPole,
MountainCar and Acrobot, demonstrated competitive performance compared to the
original DRL agent. Although there may be a slight decrease in performance due to the
behavior cloning process, the BC agents maintained proficient behavior, indicating the
successful transfer of expertise from the DRL agent. Furthermore, the interpretability
analysis revealed that decision trees and logistic regression models effectively captured
the decision-making process of the BC agents. The feature importance’s and tree visu-
alizations provided by these models allowed for a comprehensive understanding of the
factors influencing the agents’ actions, enhancing their transparency and interpretabil-
ity. The availability of understandable rules and feature importance ranking aims to
encourage trust-building and collaboration between humans and RL agents.
The proposed approach contributes to addressing the challenges associated with
the lack of transparency and interpretability in RL agents. The trade-off between per-
formance and explainability in behavior cloning was also presented, highlighting that
shallow decision trees which prioritized explainability do not always achieve satisfac-
tory performance. On the other hand, decision trees with maximum depth equals 4
strike a reasonable balance between the two aspects, since they could still be considered
sufficiently shallow and explainable. However, the optimal depth may vary depending
on the specific RL environment and the desired levels of explainability in practical
applications. In addition, by providing interpretable justifications for their actions, our
agents facilitate their deployment in critical domains where trust, accountability and
transparency are paramount. For future research, different models should be explored
in an attempt to address possible overfitting issues on the DRL dataset and to be able
to tackle more complex environments. Such approaches could be rule-based models,
distance or similarity based models and different DRL training procedures.
In conclusion, this study demonstrates the feasibility of creating RL agents that
exhibit both performance and explainability by utilizing DRL with BC. The combi-
nation of these approaches opens new avenues for deploying RL agents in real-world
applications, in an attempt at enabling transparency, interpretability and collabora-
tion between humans and RL systems. By bridging the gap between performance and
explainability, the study aims for the adoption of RL agents in critical domains that
demand both proficiency and transparency.

## References
