# SAC-from-Scratch-in-Unity-TwoLeg-Locomotion
While PPO updates its policy by using samples collected from the current policy to adjust action probabilities, Soft Actor-Critic (SAC) updates its policy by using samples collected from even past policies to select actions with high Q-values. While DQN’s argmax over actions requires computing Q-values for all possible actions, which is infeasible in continuous action spaces, SAC solves this problem by using a critic to estimate Q-values and an actor to directly generate actions, enabling continuous action control. While PPO maximizes an entropy bonus in its total loss to encourage exploration during small policy updates, SAC maximizes entropy both in its loss and its target Q-values, making it more focused on entropy maximization than PPO.

In the previous ES+PPO project, performance failed to improve beyond a certain point despite having room for further improvement. This project aims to see whether SAC can surpass that performance in two-legged locomotion. The code was improved to enable further training across models saved from ES, PPO, and SAC. SAC is based on the paper [*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*](https://arxiv.org/pdf/1801.01290) and OpenAI Spinning Up [Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html).

Because I have no idea how to turn articulation reinforcement learning into a game, and Unity development takes too much time, in this case I will share the visual results only on YouTube instead of itch.io.

<img src="images/twoleg_locomotion.png" alt="twoleg_locomotion.png" width="20%">

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
![sac plot](images/sac_plot.png)

Actor loss is a combination of Q-values and entropy, so it does not have a particular meaning in the plot. Because alpha is set small at the beginning when doing PPO → SAC further training, it is gradually increased to encourage entropy maximization. The critic loss increases slightly as entropy grows. There is a small difference between the two critic losses, and this difference helps reduce overestimation, enabling more stable learning.

The previous stability score method, which saves the model using the mean and variance of rewards, has a problem. Even when the policy improves and the reward increases, the score can still go down because of the variance, so better policies cannot be saved. Because small simulation noise makes agents behave differently with the same neural network, the variance in this case does not correspond to real changes in behavior, so it is not a good criterion for saving the model.

The previous actor freeze method, where the critic is trained first, has a problem. In the previous project, I trained it for a short time, and in this project I tried training it for longer. However, the critic loss does not steadily decrease and instead continuously oscillates up and down because the critic's target is continuously re-estimated by its own prediction (next value), so it is just a waste of time. Despite this instability in the critic, training the actor and critic simultaneously keeps the critic stable because the actor continuously generates new data, so errors on the same data do not accumulate over time, and the critic can improve its approximation through short training on the same data.

## Observation/Action Expansion

I think PPO is more suitable than ES or SAC when further training from a saved model after adding new observations or actions.

For SAC -> SAC, the dimensions of observations or actions stored in the replay buffer do not match those of the newly collected data. For ES/PPO -> SAC, even if a well-pretrained model is loaded, expanding the parameters corresponding to newly added observations or actions forces the agent to explore sufficiently due to entropy maximization, so it may not provide a substantial advantage over training SAC from the beginning.

For ES, because parameter updates are based on random noise rather than real gradients, it is difficult to adjust individual parameters in a targeted and meaningful way, so it may not provide a substantial advantage over training ES from the beginning.

In contrast, PPO performs gradient-based optimization, allowing parameters to be adjusted in meaningful directions. Moreover, the clipping mechanism constrains large policy updates, helping the new policy remain close to the previous one even while exploring.

## Conclusion
ES established balance on two legs, PPO led to forward movement in further training, and SAC enabled faster forward movement in further training. As a result, the limitation of PPO—having room for further improvement but failing to improve beyond that—was solved by SAC, which encourages strong exploration and can discover more optimal actions.

Additionally, directly training with SAC from the beginning could successfully learn forward locomotion in a short time without this step-by-step training process. This is because SAC maintains high action diversity not only in the early stage but also in the mid and late stages, which allows it to keep discovering good samples throughout training, and it continuously reuses samples from past policies, allowing useful samples to be learned from multiple times during training. However, in random environments, PPO may be better because it only uses samples from the current policy, and reusing past samples may not be useful because they come from different policies and do not match the current random environment.
