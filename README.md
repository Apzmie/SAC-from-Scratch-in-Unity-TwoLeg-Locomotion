# Under-Construction



## Environment
### Unity
- Unity Editor: 6000.3.0f1
- ML Agents: 4.0.3
- Sentis: 2.6.1

### Python
- Python 3.10.12

## SAC Diagram
![sac_diagram](images/sac_diagram.png)

### Q-role Difference in DQN vs SAC
In DQN, Q-values are used to select actions via argmax. In SAC, Q-values are used to train both the critic and the actor, while actions are sampled from the stochastic policy (actor). Although DQN can take both the state and action as inputs to estimate a Q-value like SAC, it typically takes only the state as input because this enables a direct argmax operation over the Q-values of all possible actions.

### Alpha Loss
Log probabilities can be positive in continuous action spaces because they are computed from probability density functions rather than discrete probabilities. Since log probabilities are summed over action dimensions, the total log probability can become a large positive value, which may cause alpha to increase. If the target entropy is set with a positive sign, alpha may increase even when it should decrease, leading to excessively high randomness.

### Actor Loss
The Q-value used in the actor loss is computed using a newly sampled action from the current policy, given a state sampled from the replay buffer, rather than the action stored in the replay buffer. This is because the actor should be updated to produce actions that the current critic evaluates highly. The actions stored in the replay buffer are simply records of past transitions and are used during the critic update to learn from past experience.

The objective of the actor loss is to maximize both Q-value and entropy. However, maximizing the Q-value tends to make the policy concentrate on specific actions, which reduces entropy, so the two factors are in conflict. Therefore, the actor learns a policy that finds a balance between maximizing both factors.

Entropy maximization affects not only the actor loss but also all other losses, making the policy distribution soft.

## Training Progress (SAC plot)

The previous stability score method, which saves the model using the mean and variance of rewards, has a problem. Even when the policy improves and the reward increases, the score can still go down because of the variance, so better policies cannot be saved. Because small simulation noise makes agents behave differently with the same neural network, the variance in this case does not correspond to real changes in behavior, so it is not a good criterion for saving the model.

## Conclusion
