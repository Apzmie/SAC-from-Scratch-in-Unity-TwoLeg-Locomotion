# Under-Construction

self.target_entropy = -action_dim

alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
