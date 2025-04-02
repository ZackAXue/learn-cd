import os
import re
import json
import numpy as np
import torch

from typing import Dict, Any, List
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply, bw_create_tokenizer, text_to_bits
# 假设你项目里已有 logger, bw_create_tokenizer, text_to_bits, create_tokenizer 这些函数
# 如果没有，请自行替换或实现。


def load_blocks_data(data_path: str):
    """加载 BlocksWorld 的 JSON 数据，返回 episodes 列表。"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data["episodes"]


def extract_block_name(token_str):
    """
    e.g. 'clearA' / 'onTableB' / 'holdC' -> 返回 'A', 'B', 'C'.
    若找不到，则返回 None。
    """
    match = re.search(r'([A-Z])$', token_str)
    if match:
        return match.group(1)
    return None


class BlocksWorldDataset(BaseDataset):
    """
    BlocksWorldDataset:
    参考 D4RLKitchenDataset, 用于在 Blocks World 环境中读取离散(谓词) + 连续(motion) 信息。

    - 在 __init__ 中:
      1) 加载数据 & episodes
      2) (可选) 创建/加载 tokenizer
      3) 逐条解析episodes，构造 self.data

    - 在 __getitem__ 中:
      返回包含下列字段的字典(或你需要的子集):
        batch["obs"]["init_discrete"]:  (11,6)
        batch["obs"]["goal_discrete"]:  (11,6)
        batch["obs"]["init_coords_block"]: (5,2)
        batch["obs"]["goal_coords_block"]: (5,2)
        batch["obs"]["init_coords_ee"]: (T_init,2)
        batch["obs"]["goal_coords_ee"]: (T_goal,2)
        batch["hl_discrete_action_seq"]: (8,6)
        batch["ll_traj"]: (48,3)
        batch["segment_idx"]: e.g. [(0,6), (6,12), ...]
        batch["act"], batch["rew"], batch["val"] (如果需要)
    """
    def __init__(
            self,
            data_path: str,
            horizon: int = 8,
            steps_per_action: int = 6,
            max_predicates: int = 11,
            max_blocks: int = 5,
            bit_dim: int = 6,
            tokenizer_save_path: str = './blocksworld_tokenizer',
            set_tokenizer: bool = False,
            do_normalize: bool = True,
            discount: float = 0.99,
            **kwargs
    ):
        super().__init__()  # 如果 BaseDataset 有默认构造则可省
        self.data_path = data_path
        self.horizon = horizon              # HL动作数量
        self.steps_per_action = steps_per_action  # LL每动作5帧
        self.bit_dim = bit_dim
        self.max_predicates = max_predicates
        self.max_blocks = max_blocks
        self.discount = discount
        
        # 1. 加载 BlocksWorld JSON
        self.episodes = load_blocks_data(data_path)

        # 2. 准备离散文本(做tokenizer)
        all_discrete_texts = []
        for ep in self.episodes:
            init_text = " ".join(ep.get("initial_symbolic_state", []))   # e.g. ["clearA", "onTableB", ...]
            goal_text = " ".join(ep.get("goal_symbolic_state", []))
            action_names = [a["action_name"] for a in ep.get("action_sequence", [])]
            action_text = " ".join(action_names)
            combined = init_text + " " + goal_text + " " + action_text
            all_discrete_texts.append(combined)
        big_text = " ".join(all_discrete_texts)

        # 3. 创建 or 加载 tokenizer
        if not os.path.exists(tokenizer_save_path):
            print(f"[BlocksWorldDataset] tokenizer path not found, creating: {tokenizer_save_path}")
            bw_create_tokenizer(big_text, tokenizer_save_path)
        elif set_tokenizer:
            print(f"[BlocksWorldDataset] forcing re-generate tokenizer at: {tokenizer_save_path}")
            bw_create_tokenizer(big_text, tokenizer_save_path)
        # load tokenizer
        tokenizer_json_path = os.path.join(tokenizer_save_path, 'tokenizer.json')
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_json_path,
            unk_token="[UNK]",
            pad_token="[PAD]"
        )
        # 这里可添加自定义特殊 token
        special_tokens_dict = {'additional_special_tokens': ['[MOTION]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"[BlocksWorldDataset] tokenizer loaded, vocab_size={self.tokenizer.vocab_size}")

        # 准备 [PAD] bits
        self.pad_bits = text_to_bits('[PAD]', self.tokenizer, self.bit_dim)
        assert(self.pad_bits.ndim == 1)
            

        # 4. 解析 episodes => self.data
        #    每个 episode -> convert to sample dict
        self.data = []
        for ep in self.episodes:
            sample_dict = self.convert_episode_to_sample(ep)
            if sample_dict is not None:
                self.data.append(sample_dict)
        # (可选) 5. Normalizer (比如对 coords 做GaussianNormalizer)
        self.normalizers = {}
        if do_normalize:
            coords_list = []
            traj_list = []
            for d in self.data:
                # 例如 d["obs"]["init_coords_block"] shape (5,2)
                coords_list.append(d["obs"]["init_coords_block"])
                coords_list.append(d["obs"]["goal_coords_block"])
                # 也可能想把 ll_traj 的 (40,3) 一并统计
                traj_list.append(d["ll_traj"])
            if len(coords_list)>0:
                all_arr = np.concatenate(coords_list, axis=0)
                self.normalizers["blocks"] = GaussianNormalizer(all_arr)
            else:
                self.normalizers["blocks"] = None
            if len(traj_list) > 0:
                all_traj_arr = np.concatenate(traj_list, axis=0)   # shape (M,3)
                self.normalizers["traj"] = GaussianNormalizer(all_traj_arr)
            else:
                self.normalizers["traj"] = None
        else:
            self.normalizers["blocks"] = None
            self.normalizers["traj"] = None
        print(f"[BlocksWorldDataset] dataset created, total {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def convert_episode_to_sample(self, ep: Dict[str,Any]) -> Dict[str,Any]:
        """
        将单个 episode 转成一个 sample dict:
          {
            "obs": {
              "init_discrete": (11,6),
              "goal_discrete": (11,6),
              "init_coords_block": (5,2),
              "goal_coords_block": (5,2),
              "init_coords_ee": ...,
              "goal_coords_ee": ...
            },
            "hl_discrete_action_seq": (8,6),
            "ll_traj": (40,3),
            "segment_idx": [...],
            "act": ...,
            "rew": ...,
            "val": ...
          }
        """
        sample_data = {}

        # 解析离散 init/goal
        init_list = ep.get("initial_symbolic_state", [])
        goal_list = ep.get("goal_symbolic_state", [])
        init_discrete = self.parse_predicate_list(init_list, self.max_predicates, self.bit_dim)
        goal_discrete = self.parse_predicate_list(goal_list, self.max_predicates, self.bit_dim)

        # 解析 block coords
        init_coords_block = np.zeros((self.max_blocks,2), dtype=np.float32)
        goal_coords_block = np.zeros((self.max_blocks,2), dtype=np.float32)
        init_coords_block = self.parse_block_coords(ep["state_sequence"][0].get("positions", {}))
        goal_coords_block = self.parse_block_coords(ep["state_sequence"][-1].get("positions", {}))

        # ee coords
        init_coords_ee = np.zeros((1,2), dtype=np.float32)
        goal_coords_ee = np.zeros((1,2), dtype=np.float32)
        init_coords_ee = np.array(ep["state_sequence"][0].get("ee_pos"), dtype=np.float32)
        goal_coords_ee = np.array(ep["state_sequence"][-1].get("ee_pos"), dtype=np.float32)

        # HL 动作 (8, bit_dim)
        action_names = [a["action_name"] for a in ep.get("action_sequence",[])]
        hl_actions = self.parse_action_sequence(action_names, self.horizon, self.bit_dim)

        # LL 轨迹 (horizon*steps_per_action, 3)
        # segment_idx: e.g. [(0,5),(5,10),...]
        ll_traj, seg_idx = self.parse_ll_motion(ep.get("motion_data", []),
                                                self.horizon,
                                                self.steps_per_action)

        # obs
        obs_dict = {
            "init_discrete": init_discrete,       # (11,6)
            "goal_discrete": goal_discrete,       # (11,6)
            "init_coords_block": init_coords_block,
            "goal_coords_block": goal_coords_block,
            "init_coords_ee": init_coords_ee,
            "goal_coords_ee": goal_coords_ee
        }
        # 也可加更多

        # act/rew/val 占位
        act = np.zeros((1,1), dtype=np.float32)
        rew = np.zeros((1,1), dtype=np.float32)
        val = 0.0

        sample_data = {
            "obs": obs_dict,
            "hl_discrete_action_seq": hl_actions,  # (8,6)
            "ll_traj": ll_traj,                    # (40,3)
            "segment_idx": seg_idx,                # list of (start,end)
            "act": act,
            "rew": rew,
            "val": val
        }
        return sample_data

    def parse_predicate_list(self, pred_list:List[str], max_items:int, bit_dim:int):
        """
        把离散谓词列表 -> (max_items, bit_dim)。若长度不足，补 [PAD]；超出则截断。
        """
        arr = np.zeros((max_items, bit_dim), dtype=np.float32)
        assert(len(pred_list) <= max_items)
        n = min(len(pred_list), max_items)
        for i in range(n):
            # text_to_bits() 会返回 shape (n_tokens, bit_dim)，这里示例只取第一个token行
            bits_2d = text_to_bits(pred_list[i], self.tokenizer, bit_dim)
            assert(bits_2d.ndim==1)
            arr[i,:] = bits_2d
        # 剩余用pad_bits填充 (可选)
        for i in range(n, max_items):
            arr[i,:] = self.pad_bits
        return arr

    def parse_block_coords(self, coords_data):
        """
        coords_data 是dict，key是block name，value是坐标列表
        """
        arr = np.zeros((self.max_blocks,2), dtype=np.float32)
        assert(len(coords_data) <= self.max_blocks)
        if isinstance(coords_data, list):
            for i, c in enumerate(coords_data[:self.max_blocks]):
                arr[i,0] = c[0]
                arr[i,1] = c[1]
        elif isinstance(coords_data, dict):
            i=0
            for _, v in coords_data.items():
                arr[i,0] = v[0]
                arr[i,1] = v[1]
                i+=1
                if i>=self.max_blocks:
                    break
        return arr

    def parse_action_sequence(self, action_list:List[str], horizon:int, bit_dim:int):
        """
        HL动作列表 -> shape (horizon, bit_dim)
        """
        arr = np.zeros((horizon, bit_dim), dtype=np.float32)
        assert(len(action_list) <= horizon)
        n = min(len(action_list), horizon)
        for i in range(n):
            bits_2d = text_to_bits(action_list[i], self.tokenizer, bit_dim)
            if bits_2d.ndim>1:
                arr[i,:] = bits_2d[0]
            else:
                arr[i,:] = bits_2d
        # padding
        for i in range(n, horizon):
            arr[i,:] = self.pad_bits
        return arr

    def parse_ll_motion(self, motion_data, horizon:int, steps_per_action:int):
        """
        生成 (horizon*steps_per_action,3) + segment_idx
        motion_data: list of shape -> motion_data[i]["time_series"] ~ (<=5 steps), each [t, x, z]
        returns:
            arr: (horizon*steps_per_action, 3)
            seg_idx: list of tuples (start, end)
        """
        total_len = horizon*steps_per_action
        arr = np.zeros((total_len,3), dtype=np.float32)
        seg_idx = np.zeros((horizon,2), dtype=np.int32)
        # pad motion_data到horizon
        assert(len(motion_data) <= horizon)
        while len(motion_data)< horizon:
            motion_data.append({"time_series":[]})

        for i in range(horizon):
            start = i*steps_per_action
            end = start+steps_per_action
            seg_idx[i, 0] = start
            seg_idx[i, 1] = end
            series = motion_data[i].get("time_series", [])
            # 截断或pad
            assert(len(series) <= steps_per_action)
            steps = min(len(series), steps_per_action)
            for s_idx in range(steps):
                arr[start+s_idx,:] = series[s_idx]
        return arr, seg_idx

    def __getitem__(self, idx: int):
        sample_data = self.data[idx]
        
        # 如果要对 coords 做normalize
        if self.normalizers.get("blocks", None) is not None:
            c_norm = self.normalizers["blocks"]
            # init, goal block coords
            sample_data["obs"]["init_coords_block"] = c_norm.normalize(sample_data["obs"]["init_coords_block"])
            sample_data["obs"]["goal_coords_block"] = c_norm.normalize(sample_data["obs"]["goal_coords_block"])
            # 也可对 ll_traj normalize: sample_data["ll_traj"] = c_norm.normalize(sample_data["ll_traj"])
        if self.normalizers.get("traj", None) is not None:
            t_norm = self.normalizers["traj"]
            sample_data["ll_traj"] = t_norm.normalize(sample_data["ll_traj"])
            # TODO: check here
        # 转成 torch tensor
        batch = {}
        batch["obs"] = {}
        for k in ["init_discrete","goal_discrete","init_coords_block","goal_coords_block",
                  "init_coords_ee","goal_coords_ee"]:
            batch["obs"][k] = torch.tensor(sample_data["obs"][k], dtype=torch.float32)

        batch["hl_discrete_action_seq"] = torch.tensor(sample_data["hl_discrete_action_seq"], dtype=torch.float32)
        batch["ll_traj"] = torch.tensor(sample_data["ll_traj"], dtype=torch.float32)
        batch["segment_idx"] = torch.tensor(sample_data["segment_idx"], dtype=torch.long)

        batch["act"] = torch.tensor(sample_data["act"], dtype=torch.float32)
        batch["rew"] = torch.tensor(sample_data["rew"], dtype=torch.float32)
        batch["val"] = torch.tensor(sample_data["val"], dtype=torch.float32)

        return batch

    def get_normalizer(self):
        return self.normalizers


def main():
    from torch.utils.data import DataLoader
    
    data_path = "dev/blocksworld/tamp_test_dataset.json"      # 你的JSON文件路径
    tokenizer_save_path = "dev/blocksworld/blocks_tokenizer"  # tokenizer保存或读取的路径
    
    dataset = BlocksWorldDataset(
        data_path=data_path,
        horizon=8,
        steps_per_action=6,
        max_predicates=11,
        max_blocks=5,
        bit_dim=6,
        tokenizer_save_path=tokenizer_save_path,
        set_tokenizer=False,     # 是否强制重建 tokenizer
        do_normalize=True,       # 是否做GaussianNormalizer
        discount=0.99,
    )
    
    print(f"[main] Dataset created! Length = {len(dataset)} episodes.")

    # 2. 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. 取一个 batch 测试
    batch = next(iter(dataloader))
    
    # 4. 打印 batch 里的一些信息
    print(f"[main] batch['obs']['init_discrete'].shape = {batch['obs']['init_discrete'].shape}")       # (B, 11, 6)
    print(f"[main] batch['obs']['goal_discrete'].shape = {batch['obs']['goal_discrete'].shape}")       # (B, 11, 6)
    print(f"[main] batch['hl_discrete_action_seq'].shape = {batch['hl_discrete_action_seq'].shape}")   # (B, 8, 6)
    print(f"[main] batch['ll_traj'].shape = {batch['ll_traj'].shape}")  
    print(f"[main] segment_idx shape = {batch['segment_idx'].shape}")
    print(f"[main] segment_idx = {batch['segment_idx']}")                                              # list of tuples
    
    # 5. 若要查看 Normalizer, 也可
    normalizers = dataset.get_normalizer()
    if normalizers["blocks"] is not None:
        print("[main] blocks normalizer mean:", normalizers["blocks"].mean)
        print("[main] blocks normalizer std:", normalizers["blocks"].std)
    if normalizers.get("traj", None) is not None:
        print("[main] traj normalizer mean:", normalizers["traj"].mean)
        print("[main] traj normalizer std:", normalizers["traj"].std)

    print("[main] Done testing dataset & dataloader!")

if __name__ == "__main__":
    main()