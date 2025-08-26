import random
from tqdm import tqdm
from deep_q_network import DQN
from environment import Environment, History, ReplayMemory
import torch
import numpy as np

# Chọn device tự động: CUDA -> MPS (Apple) -> CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
torch.set_float32_matmul_precision("high") if device.type == "cuda" else None


class Agent(object):
    def __init__(self, conf):
        self.env = Environment(name=conf.env, width=conf.width, height=conf.height, history=conf.history)
        self.hist = History(self.env)
        self.mem = ReplayMemory(self.env, capacity=conf.mem_capacity, batch_size=conf.batch_size)
        self._capa = conf.mem_capacity
        self._ep_en = conf.ep_end
        self._ep_st = conf.ep_start
        self._learn_st = conf.learn_start
        self._tr_freq = conf.train_freq
        self._update_freq = conf.update_freq
        self._max_steps = getattr(conf, "max_steps", 50_000_000)

        # Mạng và optimizer
        self.q = DQN(self.hist._history, self.env.action_size).to(device)
        self.target_q = DQN(self.hist._history, self.env.action_size).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optim = torch.optim.RMSprop(self.q.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

    def train(self):
        screen, reward, action, terminal = self.env.new_random_game()
        for _ in range(self.env._history):
            self.hist.add(screen)

        num_game, self.update_count, ep_reward = 0, 0, 0.0
        total_reward, self.total_loss, self.total_q = 0.0, 0.0, 0.0
        ep_rewards, actions = [], []

        for self.step in tqdm(range(self._max_steps), ncols=70, initial=0):
            if self.step == self._learn_st:
                num_game, self.update_count, ep_reward = 0, 0, 0.0
                total_reward, self.total_loss, self.total_q = 0.0, 0.0, 0.0
                ep_rewards, actions = [], []

            action = self._select_action()
            screen, reward, terminal = self.env.act(action)
            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            if self.step >= self._learn_st and self.step % 10_000 == 9_999:
                avg_reward = total_reward / 10_000.0
                avg_loss = (self.total_loss / max(1, self.update_count))
                avg_q = (self.total_q / max(1, self.update_count))
                print(f'# games: {num_game}, reward: {avg_reward}, loss: {avg_loss}, q: {avg_q:.4f}')

                num_game = 0
                total_reward = 0.0
                self.total_loss = 0.0
                self.total_q = 0.0
                self.update_count = 0
                ep_reward = 0.0
                ep_rewards = []
                actions = []

    def observe(self, screen, reward, action, terminal):
        reward = float(np.clip(reward, -1.0, 1.0))
        self.hist.add(screen)
        self.mem.add(screen, reward, action, float(terminal))

        if self.step > self._learn_st:
            if self.step % self._tr_freq == 0:
                self._q_learning()

            if self.step % self._update_freq == self._update_freq - 1:
                self.target_q.load_state_dict(self.q.state_dict())
                if self.step % (self._update_freq * 10) == (self._update_freq * 10) - 1:
                    torch.save(self.target_q.state_dict(), f'models/model_{self.step}.pt')

    def play(self, model_path, num_ep=100):
        # Tải state_dict an toàn; nếu bạn lưu nguyên model thì thay bằng torch.load(model_path)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict):
            self.q.load_state_dict(state)
        else:
            # fallback nếu file là nguyên model
            self.q = state
        self.q.eval()

        best_reward = float("-inf")
        best_screen_hist = []

        for ep in range(num_ep):
            print(f'# episode: {ep}')
            screen, reward, action, terminal = self.env.new_random_game(force=True)
            current_reward = 0.0
            current_screen_hist = []
            act_hist = []

            current_screen_hist.append(self.env.screen)
            for _ in range(self.env._history):
                self.hist.add(screen)

            cnt = 0
            while not terminal:
                cnt += 1
                action = self._select_action(test_mode=True)
                act_hist.append(action)

                # tránh local maxima (lặp 1 action quá lâu)
                if cnt > 200 and len(act_hist) >= 100:
                    if np.mean(act_hist[-100:]) == act_hist[-1]:
                        action = random.randrange(self.env.action_size)

                screen, reward, terminal = self.env.act(action, is_train=False)
                self.hist.add(screen)
                current_reward += reward
                current_screen_hist.append(self.env.screen)

            print(current_reward)
            if current_reward > best_reward:
                best_reward = current_reward
                best_screen_hist = current_screen_hist

        import imageio
        print(f'best reward: {best_reward}')
        # Đảm bảo path tồn tại, có thể đổi '/data/...' thành file cục bộ:
        out_path = f'best_{int(best_reward)}.gif'
        imageio.mimsave(out_path, best_screen_hist, duration=0.0001)
        print(f'Saved GIF to {out_path}')
        return best_reward

    def _q_learning(self):
        sc_t, actions, rewards, sc_t_1, terminals = self.mem.sample()

        # Tensors
        batch_obs_t = self._to_tensor(sc_t, requires_grad=False)                 # [B, H, W] x history -> DQN tự xử lý
        batch_obs_t_1 = self._to_tensor(sc_t_1, requires_grad=False)
        batch_rewards = torch.from_numpy(rewards).to(device).float().unsqueeze(1)
        batch_actions = torch.from_numpy(actions).to(device).long().unsqueeze(1)
        batch_nonterminal = torch.from_numpy(1.0 - terminals).to(device).float().unsqueeze(1)

        # Q(s,a)
        q_values_all = self.q(batch_obs_t)                       # [B, A]
        q_values = q_values_all.gather(1, batch_actions)         # [B, 1]

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_max_q_values = self.target_q(batch_obs_t_1).max(1, keepdim=True)[0]  # [B, 1]
            target_q_values = batch_rewards + 0.99 * batch_nonterminal * next_max_q_values

        # Smooth L1 loss
        cri = torch.nn.SmoothL1Loss()
        loss = cri(q_values, target_q_values)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        # (tùy chọn) gradient clipping cho ổn định
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim.step()

        self.update_count += 1
        # logging
        with torch.no_grad():
            self.total_q += q_values.mean().item()
            self.total_loss += loss.item()

    def _select_action(self, test_mode=False):
        # epsilon-greedy
        if not test_mode:
            ep = self._ep_en + max(
                0.0,
                (self._ep_st - self._ep_en) * (self._capa - max(0.0, self.step - self._learn_st)) / self._capa
            )
        else:
            ep = -1.0

        if random.random() < ep:
            return random.randrange(self.env.action_size)

        with torch.no_grad():
            inputs = self._to_tensor(self.hist.get, requires_grad=False).unsqueeze(0)  # [1, H, H, W] theo DQN định nghĩa
            pred = self.q(inputs)                 # [1, A]
            action = int(pred.argmax(dim=1).item())
        return action

    def _to_tensor(self, ndarray, requires_grad=False):
        """
        ndarray: numpy array (float32), shape thường là:
            - state batch: [B, history, H, W]
            - single state: [history, H, W]
        """
        t = torch.from_numpy(np.array(ndarray, copy=False)).to(device)
        if t.dtype == torch.float64:
            t = t.float()
        t.requires_grad_(requires_grad)
        return t
