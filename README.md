# Under-Construction

self.target_entropy = -action_dim

alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

In continuous action spaces, log_prob can be positive value. While increasing and decreasing of alpha in alpha_loss seems to be correct in both -action_dim and action_dim. But, the goal of optimizer is to make gradient zero, not positive or negative value. This is the reason why -action_dim is correct that log_prob should become positive value to get action confidence.
