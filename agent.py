import random
from tqdm import tqdm
from deep_q_network import DQN
from environment import Environment, History, ReplayMemory
import torch
import numpy as np
from torch.serialization import add_safe_globals
import imageio
import argparse
import os
import re


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")


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
        self._current_step = getattr(conf, "current_step", 0)
        self._ep_decay_steps = 1_000_000  # paper: linear decay over 1M steps

        # Networks
        self.q = DQN(self.hist._history, self.env.action_size).to(device)
        self.target_q = DQN(self.hist._history, self.env.action_size).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optim = torch.optim.RMSprop(self.q.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

    def _reset_and_fill_history(self, target_lives=5):
        # 1. Reset game KHÔNG tự FIRE trong Environment
        screen, reward, action, terminal = self.env.new_random_game(force=True)

        # 2. Reset history ngay lập tức, điền frame tĩnh
        _ = self.hist.reset
        for _ in range(self.env._history):
            self.hist.add(screen)

        # 3. FIRE bóng để bắt đầu game
        if self.env._fire_action is not None:
            screen, reward, terminal = self.env.act(self.env._fire_action)

        # 4. Thêm frame hiện tại vào history sau FIRE
        self.hist.add(screen)

        return screen, reward, self.env._fire_action, terminal


    def train(self):
        # Reset game đầu tiên
        screen, reward, action, terminal = self._reset_and_fill_history()
        init_lives = self.env._env.unwrapped.ale.lives()

        # Biến thống kê
        num_game, self.update_count, ep_reward = 0, 0, 0.0
        total_reward, self.total_loss, self.total_q = 0.0, 0.0, 0.0
        ep_rewards, actions = [], []

        # Mở file log append
        log_path = "training_log.txt"
        log_file = open(log_path, "a")

        # Tạo thư mục lưu replay buffer
        replay_dir = "train_experiences"
        os.makedirs(replay_dir, exist_ok=True)

        # Training loop
        for self.step in tqdm(range(self._current_step, self._max_steps), ncols=70, initial=0):
            if self.step == self._learn_st:
                num_game, self.update_count, ep_reward = 0, 0, 0.0
                total_reward, self.total_loss, self.total_q = 0.0, 0.0, 0.0
                ep_rewards, actions = [], []

            # 1. chọn action
            action = self._select_action()

            # 2. thực hiện action
            screen, reward, terminal = self.env.act(action)
            self.observe(screen, reward, action, terminal)

            # 3. update reward
            ep_reward += reward
            total_reward += reward
            actions.append(action)

            # 4. xử lý nếu terminal
            if terminal:
                lives = self.env._env.unwrapped.ale.lives()

                if lives > 0:
                    # chỉ mất 1 mạng → chỉ reset history stack
                    _ = self.hist.reset
                    for _ in range(self.env._history):
                        self.hist.add(screen)
                else:
                    # Game over thật sự
                    num_game += 1
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.0

                    # Reset env cho episode mới
                    screen, reward, action, terminal = self._reset_and_fill_history()

            # 5. logging mỗi 10k bước
            if self.step >= self._learn_st and self.step % 10_000 == 9_999:
                avg_reward = total_reward / 10_000.0
                avg_loss = (self.total_loss / max(1, self.update_count))
                avg_q = (self.total_q / max(1, self.update_count))
                avg_ep_reward = sum(ep_rewards)/len(ep_rewards) if ep_rewards else 0.0

                log_str = (f'step {self.step}: # games: {num_game}, reward: {avg_reward:.2f}, '
                           f'loss: {avg_loss:.6f}, q: {avg_q:.4f}, avg_ep_reward: {avg_ep_reward:.2f}')

                print(log_str)
                log_file.write(log_str + "\n")
                log_file.flush()  # đảm bảo ghi ra file ngay

                # reset thống kê
                num_game, total_reward, self.total_loss, self.total_q, self.update_count = 0, 0.0, 0.0, 0.0, 0
                ep_reward, ep_rewards, actions = 0.0, [], []

                # 6. Lưu replay buffer mỗi 2M bước
                if self.step >= self._learn_st and self.step % 2_000_000 == 1_999_999:
                    count = self.mem._count
                    save_path = os.path.join(replay_dir, f"replay_{self.step}.npz")
                    np.savez_compressed(
                        save_path,
                        actions=self.mem._actions[:count].astype(self.mem._actions.dtype, copy=False),
                        rewards=self.mem._rewards[:count].astype(self.mem._rewards.dtype, copy=False),
                        screens=self.mem._screens[:count].astype(self.mem._screens.dtype, copy=False),
                        terminals=self.mem._terminals[:count].astype(self.mem._terminals.dtype, copy=False),
                        count=np.int64(count),
                        history=self.env._history,
                        height=self.env._height,
                        width=self.env._width
                    )
                    print(f"[ReplayBuffer] Saved replay buffer to {save_path} (count={count})")

        # Đóng file log sau khi train xong
        log_file.close()


    def observe(self, screen, reward, action, terminal):
        reward = float(np.clip(reward, -1.0, 1.0))
        self.hist.add(screen)
        self.mem.add(screen, reward, action, terminal)

        if self.step > self._learn_st:
            if self.step % self._tr_freq == 0:
                self._q_learning()

            if self.step % self._update_freq == self._update_freq - 1:
                self.target_q.load_state_dict(self.q.state_dict())
                if self.step % (self._update_freq * 10) == (self._update_freq * 10) - 1:
                    torch.save(self.target_q.state_dict(), f'models/model_{self.step}.pt')

    def _load_model_checkpoint(self, model_path):
        """Load checkpoint robust: thử state_dict trước, nếu fail thì load nguyên object."""
        # Cho phép unpickle class DQN nếu là full object
        add_safe_globals([DQN])

        # 1) Thử như state_dict (an toàn nhất)
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(state, dict):
                self.q.load_state_dict(state)
                print("[play] Loaded state_dict checkpoint.")
                return
        except Exception as e:
            print(f"[play] state_dict load failed: {e}")

        # 2) Thử load nguyên object (pickle)
        try:
            print("[play] Loading full model object (pickle, weights_only=False)...")
            loaded_model = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(loaded_model, torch.nn.Module):
                self.q.load_state_dict(loaded_model.state_dict())
                print("[play] Loaded full model object and applied state_dict.")
                return
            else:
                # fallback: nếu file là dict khác
                self.q.load_state_dict(loaded_model)
                print("[play] Loaded dict-like object into state_dict.")
                return
        except Exception as e:
            print(f"[play] full object load failed: {e}")
            raise

    def play(self, model_path, num_ep=10, eval_epsilon=0.05):
        """
        Play với đủ 5 mạng, tránh mất 2 mạng đầu do reset.
        """
        import random, numpy as np, imageio

        # 1) Load model
        self._load_model_checkpoint(model_path)
        self.q.eval()

        best_reward = float("-inf")
        best_screen_hist = []
        rewards = []

        for ep in range(num_ep):
            print(f'# episode: {ep}')
            _ = self.hist.reset   # clear history

            # Reset game hoàn toàn (5 mạng thật sự)
            self.env._env.unwrapped.ale.reset_game()
            screen, reward, action, terminal = self._reset_and_fill_history()

            current_reward = 0.0
            current_screen_hist = [self.env.screen]
            action_counts = {}
            lives = self.env._env.unwrapped.ale.lives()

            # Loop cho tới khi hết mạng
            while not terminal:
                if eval_epsilon > 0.0 and random.random() < eval_epsilon:
                    act = random.randrange(self.env.action_size)
                else:
                    act = self._select_action(test_mode=True)

                action_counts[act] = action_counts.get(act, 0) + 1
                screen, reward, terminal = self.env.act(act, is_train=False)

                self.hist.add(screen)
                current_reward += reward
                current_screen_hist.append(self.env.screen)

                new_lives = self.env._env.unwrapped.ale.lives()
                if new_lives < lives:
                    lives = new_lives

            rewards.append(current_reward)
            print(f"Episode reward: {current_reward} | actions: {action_counts}")

            if current_reward > best_reward:
                best_reward = current_reward
                best_screen_hist = current_screen_hist

        # Summary
        rewards_np = np.array(rewards, dtype=np.float32)
        print(f"\n[Summary over {num_ep} episodes]")
        print(f" Best reward  : {best_reward}")
        print(f" Average      : {float(np.mean(rewards_np)):.2f}")
        print(f" Median       : {float(np.median(rewards_np)):.2f}")
        print(f" Std          : {float(np.std(rewards_np)):.2f}")

        # Save GIF
        if best_screen_hist:
            out_path = f'best_{int(best_reward)}.gif'
            imageio.mimsave(out_path, best_screen_hist, duration=0.0001)
            print(f'Saved GIF to {out_path}')

        return {
            "best": best_reward,
            "average": float(np.mean(rewards_np)),
            "median": float(np.median(rewards_np)),
            "std": float(np.std(rewards_np)),
            "all_rewards": rewards
        }

    def _q_learning(self):
        sc_t, actions, rewards, sc_t_1, terminals = self.mem.sample()

        batch_obs_t = self._to_tensor(sc_t, requires_grad=False)
        batch_obs_t_1 = self._to_tensor(sc_t_1, requires_grad=False)
        batch_rewards = torch.from_numpy(rewards).to(device).float().unsqueeze(1)
        batch_actions = torch.from_numpy(actions).to(device).long().unsqueeze(1)
        batch_nonterminal = torch.from_numpy((~terminals).astype(np.float32)).to(device).unsqueeze(1)

        q_values_all = self.q(batch_obs_t)
        q_values = q_values_all.gather(1, batch_actions)

        with torch.no_grad():
            next_max_q_values = self.target_q(batch_obs_t_1).max(1, keepdim=True)[0]
            target_q_values = batch_rewards + 0.99 * batch_nonterminal * next_max_q_values

        cri = torch.nn.SmoothL1Loss()
        loss = cri(q_values, target_q_values)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim.step()

        self.update_count += 1
        with torch.no_grad():
            self.total_q += q_values.mean().item()
            self.total_loss += loss.item()

    def _select_action(self, test_mode=False):
        if not test_mode:
            if self.step < self._learn_st:
                ep = self._ep_st
            else:
                decay = min(1.0, (self.step - self._learn_st) / self._ep_decay_steps)
                ep = self._ep_st - decay * (self._ep_st - self._ep_en)
        else:
            ep = 0.0  # greedy policy

        if random.random() < ep:
            return random.randrange(self.env.action_size)

        with torch.no_grad():
            inputs = self._to_tensor(self.hist.get, requires_grad=False).unsqueeze(0)
            pred = self.q(inputs)
            return int(pred.argmax(dim=1).item())

    def _to_tensor(self, ndarray, requires_grad=False):
        t = torch.from_numpy(np.array(ndarray, copy=False)).to(device)
        if t.dtype == torch.float64:
            t = t.float()
        t.requires_grad_(requires_grad)
        return t

    def play_save_mode(self, model_path, num_ep=10, eval_epsilon=0.05, fire_warmup=8,
                       save_mem_capacity=100000, save_dir="play_experiences", save_name=None):
        """
        Giống play(), nhưng đồng thời lưu lại trải nghiệm vào 1 ReplayMemory tạm
        và xuất ra file .npz (giữ nguyên dtype, không convert sang uint8).
        Trả về dict giống play().
        - save_mem_capacity: kích thước replay buffer để lưu trải nghiệm (mặc định 100k)
        - save_dir: thư mục lưu file trải nghiệm
        - save_name: nếu truyền, dùng tên này (không có ext), ngược lại sẽ tạo từ model_path
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 1) Load model
        self._load_model_checkpoint(model_path)
        self.q.eval()

        # replay memory tạm để lưu trải nghiệm chơi (giữ dtype/format giống ReplayMemory)
        play_mem = ReplayMemory(self.env, capacity=save_mem_capacity, batch_size=self.mem._batch_size)

        # 2) Debug action meanings (nếu ALE expose)
        try:
            meanings = self.env._env.unwrapped.get_action_meanings()
            print("[play_save_mode] Action meanings:", meanings)
        except Exception:
            meanings = None

        best_reward = float("-inf")
        best_screen_hist = []
        rewards = []

        for ep in range(num_ep):
            print(f'# episode: {ep}')
            _ = self.hist.reset   # reset history

            # start new game
            screen, reward, action, terminal = self._reset_and_fill_history()
                # LƯU initial frames vào play_mem (giữ nguyên format)
                # Lưu initial frames as transitions with action 0 and reward 0 is optional;
                # Nhưng để đầy đủ, ta không ghi action/reward ở frame khởi tạo.
            # Warmup FIRE nếu cần
            fire_action = None
            if meanings is not None:
                for i, m in enumerate(meanings):
                    if "FIRE" in m:
                        fire_action = i
                        break

            if fire_action is not None and fire_warmup > 0:
                for _ in range(fire_warmup):
                    screen, r, terminal = self.env.act(fire_action, is_train=False)
                    self.hist.add(screen)
                    # lưu transition (fire warmup) vào play_mem
                    try:
                        play_mem.add(screen, float(np.clip(r, -1.0, 1.0)), int(fire_action), bool(terminal))
                    except Exception:
                        # nếu play_mem đầy thì ngắt lưu (play_mem giới hạn)
                        pass
                    if terminal:
                        screen, reward, action, terminal = self._reset_and_fill_history()
                        break

            current_reward = 0.0
            current_screen_hist = [self.env.screen]
            action_counts = {}

            while not terminal:
                if eval_epsilon > 0.0 and random.random() < eval_epsilon:
                    act = random.randrange(self.env.action_size)
                else:
                    act = self._select_action(test_mode=True)

                action_counts[act] = action_counts.get(act, 0) + 1

                # thực hiện action
                screen, reward, terminal = self.env.act(act, is_train=False)
                self.hist.add(screen)

                # LƯU transition vào play_mem
                try:
                    play_mem.add(screen, float(np.clip(reward, -1.0, 1.0)), int(act), bool(terminal))
                except Exception:
                    # Nếu buffer đã đầy (capacity nhỏ), ta chỉ bỏ qua thêm transition
                    pass

                current_reward += reward
                current_screen_hist.append(self.env.screen)

            rewards.append(current_reward)
            print(f"Episode reward: {current_reward} | actions: {action_counts}")

            if current_reward > best_reward:
                best_reward = current_reward
                best_screen_hist = current_screen_hist

        # thống kê giống play()
        rewards_np = np.array(rewards, dtype=np.float32)
        avg_reward = float(np.mean(rewards_np))
        median_reward = float(np.median(rewards_np))
        std_reward = float(np.std(rewards_np))

        print(f"\n[Summary over {num_ep} episodes]")
        print(f" Best reward  : {best_reward}")
        print(f" Average      : {avg_reward:.2f}")
        print(f" Median       : {median_reward:.2f}")
        print(f" Std          : {std_reward:.2f}")

        # save GIF như cũ
        out_gif = f'best_{int(best_reward)}.gif'
        if best_screen_hist:
            imageio.mimsave(out_gif, best_screen_hist, duration=0.0001)
            print(f'Saved GIF to {out_gif}')
        else:
            print("[play_save_mode] No frames captured; GIF not saved.")

        # Lưu toàn bộ trải nghiệm đã thu vào file .npz (giữ nguyên dtype)
        # tạo tên file
        if save_name:
            fname = save_name
        else:
            # lấy tên model (không path, không ext) làm base
            base = os.path.splitext(os.path.basename(model_path))[0]
            fname = f"experiences_{base}"

        save_path = os.path.join(save_dir, f"{fname}.npz")
        # cắt theo count hiện tại
        count = play_mem._count
        # LƯU: giữ nguyên dtype/mảng như ReplayMemory
        np.savez_compressed(
            save_path,
            actions=play_mem._actions[:count].astype(play_mem._actions.dtype, copy=False),
            rewards=play_mem._rewards[:count].astype(play_mem._rewards.dtype, copy=False),
            screens=play_mem._screens[:count].astype(play_mem._screens.dtype, copy=False),
            terminals=play_mem._terminals[:count].astype(play_mem._terminals.dtype, copy=False),
            count=np.int64(count),
            history=self.env._history,
            height=self.env._height,
            width=self.env._width
        )
        print(f"[play_save_mode] Saved experiences to {save_path} (count={count})")

        # trả về kết quả giống play()
        return {
            "best": best_reward,
            "average": avg_reward,
            "median": median_reward,
            "std": std_reward,
            "all_rewards": rewards,
            "experience_file": save_path
        }

    def resume_train(self, resume_model_path=None, max_steps=None, buffer_dir=None):
        # Load lại model
        if resume_model_path is not None:
            print(f"Loading model from {resume_model_path}")
            self.q.load_state_dict(torch.load(resume_model_path))
            self.target_q.load_state_dict(self.q.state_dict())

        # Hỏi người dùng nhập step hiện tại của checkpoint
        try:
            current_step = int(input("Nhập step hiện tại của checkpoint: "))
            self._current_step = current_step
            print(f"[resume_train] Đặt self._max_step = {self._max_steps}")
        except ValueError:
            print("[resume_train] Input không hợp lệ, giữ nguyên self.step.")

        # Khôi phục replay buffer
        if buffer_dir is not None:
            print(f"Loading replay buffer from directory: {buffer_dir}")
            files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(".npz")])

            if not files:
                print("Không tìm thấy file .npz nào trong thư mục!")
            else:
                for f in files:
                    path = os.path.join(buffer_dir, f)
                    print(f"Loading buffer chunk: {path}")
                    data = np.load(path, allow_pickle=True)
                    # Giả sử bạn đã lưu các mảng screen, reward, action, terminal
                    screens = data["screens"]
                    rewards = data["rewards"]
                    actions = data["actions"]
                    terminals = data["terminals"]

                    # Add vào replay buffer
                    for s, r, a, t in zip(screens, rewards, actions, terminals):
                        self.mem.add(s, r, a, t)

        # Resume training loop
        print("Start/resume training...")
        self.train()