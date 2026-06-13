from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

BASE_DIR = ""


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(x).squeeze(-1)
        return mean, std, value
        
    def sample(self, x):
        mean, std, value = self.forward(x)
        
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, z, log_prob, value
        
    def deterministic(self, x):
        mean, _, _ = self.forward(x)
        return torch.tanh(mean)
                

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.zs = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.returns = []
        self.advantages = []

    def add(self, state, action, z, reward, next_state, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.zs.append(z)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.__init__()
        
        
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.gamma = gamma
        self.lam = lam
        
        ###########################################
        ### Load fc1, fc2, mean ###
        ###########################################
        
        #state_dict = torch.load(f"{BASE_DIR}/best_model.pth")
        #self.actor_critic.fc1.load_state_dict({"weight": state_dict["fc1.weight"], "bias": state_dict["fc1.bias"]})
        #self.actor_critic.fc2.load_state_dict({"weight": state_dict["fc2.weight"], "bias": state_dict["fc2.bias"]})
        #self.actor_critic.mean.load_state_dict({"weight": state_dict["mean.weight"], "bias": state_dict["mean.bias"]})
        
        #==========================================
        
        ###########################################
        ### Add Observation ###
        ###########################################
        
        #old_state_dim = ?     
        #old_model = ActorCritic(old_state_dim, action_dim)
        #old_model.load_state_dict(torch.load(f"{BASE_DIR}/best_model.pth"), strict=False)
        
        #for name, param in old_model.state_dict().items():
        #    if "fc1" not in name:
        #        self.actor_critic.state_dict()[name].copy_(param)
        
        #with torch.no_grad():
        #    old_w = old_model.fc1.weight
        #    old_b = old_model.fc1.bias

        #    new_w = self.actor_critic.fc1.weight
        #    new_w.zero_()
        #    new_w[:, :old_state_dim].copy_(old_w)

        #    self.actor_critic.fc1.bias.copy_(old_b)
        
        #==========================================

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
    def compute_gaes(self, buffer):
        with torch.no_grad():
            rewards = torch.FloatTensor(np.array(buffer.rewards))
            values = torch.FloatTensor(np.array(buffer.values))
            dones = torch.FloatTensor(np.array(buffer.dones))

            if dones[-1] == 1.0:
                next_value = 0
            else:
                last_next_state = torch.FloatTensor(buffer.next_states[-1]).unsqueeze(0)
                _, _, last_value = self.actor_critic(last_next_state)
                next_value = last_value.item()

            gaes = [0] * len(rewards)
            last_gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gaes[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
                next_value = values[t]
                last_gae = gaes[t]

            gaes = torch.FloatTensor(np.array(gaes))
            buffer.advantages = gaes.view(-1)
            buffer.returns = gaes.view(-1) + values.view(-1)

    def update(self, buffers, epochs=5, batch_size=512):
        all_states = []
        all_zs = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        
        for buf in buffers:
            all_states.extend(buf.states)
            all_zs.extend(buf.zs)
            all_old_log_probs.extend(buf.log_probs)
            all_returns.append(buf.returns)
            all_advantages.append(buf.advantages)
            
        states = torch.FloatTensor(np.array(all_states))
        zs = torch.FloatTensor(np.array(all_zs))
        old_log_probs = torch.FloatTensor(np.array(all_old_log_probs))
        returns = torch.cat(all_returns).view(-1)
        advantages = torch.cat(all_advantages).view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                mean, std, values = self.actor_critic(states[idx])
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(zs[idx])
                log_probs = log_probs - torch.log(1 - torch.tanh(zs[idx]).pow(2) + 1e-6)
                log_probs = log_probs.sum(dim=-1)
                entropy = dist.entropy().sum(-1)
                
                ratios = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.view(-1), returns[idx].view(-1))
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + 0.001 * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
            
        return policy_loss.item(), value_loss.item(), entropy_loss.item()


if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    channel2 = EngineConfigurationChannel()
    channel2.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    test_env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel2], no_graphics=True, worker_id=1)
    env.reset()
    test_env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    t_behavior_name = list(test_env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size
    agent = PPOAgent(state_dim, action_dim)
    #agent.actor_critic.load_state_dict(torch.load(f"{BASE_DIR}/best_model.pth"))
    buffer = RolloutBuffer()
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    target_transitions = 3072    # all transitions per one update
    test_interval = 10
    test_max_step = 1000

    update_count = 0
    total_transitions = 0
    save_idx = 0
    best_test_reward = -float('inf')
    agent_buffers = {}
    completed_buffers = []

    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            for agent_id in agent_ids:
                if agent_id not in agent_buffers:
                    agent_buffers[agent_id] = RolloutBuffer()
                    
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)           
            with torch.no_grad():
                actions, zs, log_probs, values = agent.actor_critic.sample(states_tensor)
                  
            actions = actions.cpu().numpy().astype(np.float32)
            env.set_actions(behavior_name, ActionTuple(continuous=actions))           

        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)

        for i, agent_id in enumerate(agent_ids):
            state = states_tensor[i].cpu().numpy()
            action = actions[i]
            log_prob = log_probs[i]
            value = values[i]
            z = zs[i].cpu().numpy()

            if agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                done = 1.0
                next_state = np.zeros_like(state)
            elif agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                done = 0.0
                next_state = next_decision_steps[agent_id].obs[0]
            else:
                continue

            agent_buffers[agent_id].add(state, action, z, reward, next_state, log_prob.item(), value.item(), done)
            total_transitions += 1

            if done == 1.0:
                agent.compute_gaes(agent_buffers[agent_id])
                completed_buffers.append(agent_buffers[agent_id])
                agent_buffers[agent_id] = RolloutBuffer()

        if total_transitions >= target_transitions:
            for aid, buf in agent_buffers.items():
                if len(buf) > 0:
                    agent.compute_gaes(buf)
                    completed_buffers.append(buf)

            policy_loss, value_loss, entropy_loss = agent.update(completed_buffers)
            writer.add_scalar("Train/Policy_Loss", policy_loss, update_count)
            writer.add_scalar("Train/Value_Loss", value_loss, update_count)
            writer.add_scalar("Train/Entropy_Loss", entropy_loss, update_count)

            completed_buffers.clear()
            total_transitions = 0
            agent_buffers = {}
            update_count += 1           
                
            if update_count % test_interval == 0:
                print(f"Update Count {update_count}")
                test_env.reset()
                t_decision_steps, _ = test_env.get_steps(t_behavior_name)
                n_test_agents = len(t_decision_steps.agent_id)
                test_rewards = np.zeros(n_test_agents)
                test_episode_dones = np.zeros(n_test_agents, dtype=bool)
                test_id_to_index = {agent_id: i for i, agent_id in enumerate(t_decision_steps.agent_id)}
                
                test_max_step_count = 0
                while not np.all(test_episode_dones) and test_max_step_count < test_max_step:
                    t_agent_ids = t_decision_steps.agent_id
        
                    if len(t_agent_ids) > 0:
                        t_states_tensor = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32)
                        with torch.no_grad():
                            t_actions = agent.actor_critic.deterministic(t_states_tensor)
                        t_actions = t_actions.cpu().numpy().astype(np.float32)
            
                        for j, agent_id in enumerate(t_agent_ids):
                            idx = test_id_to_index[agent_id]
                            if test_episode_dones[idx]:
                                t_actions[j] = np.zeros(action_dim)
                        
                        test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions))  
                       
                    test_env.step()
                    test_max_step_count += 1
                    t_decision_steps, t_terminal_steps = test_env.get_steps(t_behavior_name)

                    for j, agent_id in enumerate(t_terminal_steps.agent_id):
                        i = test_id_to_index[agent_id]
                        if not test_episode_dones[i]:
                            test_rewards[i] += t_terminal_steps.reward[j]
                            test_episode_dones[i] = True

                    for j, agent_id in enumerate(t_decision_steps.agent_id):
                        i = test_id_to_index[agent_id]
                        if not test_episode_dones[i]:
                            test_rewards[i] += t_decision_steps.reward[j]
                            
                test_average_reward = np.mean(test_rewards)            
                writer.add_scalar("Test/Average_Reward", test_average_reward, update_count)
                print(f"{test_average_reward:.4f}")
                torch.save(agent.actor_critic.state_dict(), f"{BASE_DIR}/period_model.pth")
                
                if test_average_reward > best_test_reward:
                    best_test_reward = test_average_reward
                    save_idx += 1
                    torch.save(agent.actor_critic.state_dict(), f"{BASE_DIR}/#({save_idx})best_{best_test_reward:.4f}.pth")
                    print(f"[Test] Model saved at new best reward {best_test_reward:.4f}")
