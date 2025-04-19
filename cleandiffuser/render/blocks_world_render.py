import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast

def bits_to_text_customize(bit_matrix, tokenizer, n_bits):
    """
    Convert bit matrix to token texts.
    
    Args:
        bit_matrix: numpy array of shape (batch_size, n_bits)
        tokenizer: PreTrainedTokenizerFast object
        n_bits: number of bits per token
        
    Returns:
        list of strings: decoded tokens
    """
    # Convert bits to ids
    bits = bit_matrix.astype(np.int64)
    # Example implementation (you'll need to adjust based on your actual tokenizer implementation)
    bit_ids = np.packbits(bits, axis=1)
    token_ids = [int.from_bytes(byte_arr.tobytes(), byteorder='big') for byte_arr in bit_ids]
    # Decode token ids to text
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokens

class BlocksWorldRender:
    """
    Renderer for BlocksWorld data that directly accepts separate components.
    """
    def __init__(self, tokenizer_save_path, n_bits):
        """
        Initialize the renderer.
        
        Args:
            tokenizer_save_path: path to saved tokenizer
            n_bits: number of bits per token (e.g., 6)
        """
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_save_path)
        self.n_bits = n_bits
        
        # Normalization parameters for motion data
        self.pred_mean = np.array([0.19594476, 4.657679, 1.6395435])
        self.pred_std = np.array([0.1829183, 2.736968, 1.2567762])

    def decode_discrete_token(self, bit_row):
        """
        Decode a single row of bits into a token string.
        
        Args:
            bit_row: numpy array of shape (n_bits,)
            
        Returns:
            str: decoded token
        """
        discrete_part_2d = np.expand_dims(bit_row, axis=0)
        tokens = bits_to_text_customize(discrete_part_2d, self.tokenizer, self.n_bits)
        
        if len(tokens) > 0:
            return tokens[0]
        else:
            return "[UNK]"

    def visualize_sample(self, savepath, init_discrete, goal_discrete, 
                        actions_discrete=None, motion_data=None,
                        pred_actions_discrete=None, pred_motion_data=None):
        """
        Visualize and validate a sample with its separate components.
        
        Args:
            savepath: path to save visualization files
            init_discrete: numpy array of init state tokens (shape: n_init, n_bits)
            goal_discrete: numpy array of goal state tokens (shape: n_goal, n_bits)
            actions_discrete: numpy array of action tokens (shape: n_actions, n_bits) or None
            motion_data: numpy array of motion data (shape: n_motion, 3) or None
            pred_actions_discrete: numpy array of predicted action tokens (shape: n_actions, n_bits) or None
            pred_motion_data: numpy array of predicted motion data (shape: n_motion, 3) or None
        """
        # Create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))

        txt_path = os.path.splitext(savepath)[0] + ".txt"
        mode = 'w'  # Overwrite existing file

        with open(txt_path, mode) as f:
            f.write("==== Sample Visualization ====\n\n")
            
            # Process init states
            f.write("=== [Init States] ===\n")
            init_positions = {}
            for i, row in enumerate(init_discrete):
                token_str = self.decode_discrete_token(row)
                f.write(f"[Init {i}] token='{token_str}'\n")
                # Store block positions for later visualization if needed
                block_name = self._extract_block_name(token_str)
                if block_name:
                    # Use motion_data to get block position if available
                    init_positions[block_name] = (i, token_str)

            # Process goal states
            f.write("\n=== [Goal States] ===\n")
            for i, row in enumerate(goal_discrete):
                token_str = self.decode_discrete_token(row)
                f.write(f"[Goal {i}] token='{token_str}'\n")

            # Process ground truth actions if available
            if actions_discrete is not None:
                f.write("\n=== [Ground Truth Actions] ===\n")
                for i, row in enumerate(actions_discrete):
                    token_str = self.decode_discrete_token(row)
                    f.write(f"[Action {i}] token='{token_str}'\n")

            # Process predicted actions if available
            if pred_actions_discrete is not None:
                f.write("\n=== [Predicted Actions] ===\n")
                for i, row in enumerate(pred_actions_discrete):
                    token_str = self.decode_discrete_token(row)
                    f.write(f"[Action {i}] token='{token_str}'\n")

            # Process ground truth motion data if available
            if motion_data is not None:
                f.write("\n=== [Ground Truth Motion] ===\n")
                for i, xyz in enumerate(motion_data):
                    # Apply denormalization
                    denorm_xyz = (xyz * self.pred_std) + self.pred_mean
                    f.write(f"[Motion {i}] (time,x,z)={denorm_xyz.tolist()}\n")

            # Process predicted motion data if available
            if pred_motion_data is not None:
                f.write("\n=== [Predicted Motion] ===\n")
                for i, xyz in enumerate(pred_motion_data):
                    # Apply denormalization
                    denorm_xyz = (xyz * self.pred_std) + self.pred_mean
                    f.write(f"[Motion {i}] (time,x,z)={denorm_xyz.tolist()}\n")

            # Add comparison if both ground truth and prediction are available
            if actions_discrete is not None and pred_actions_discrete is not None:
                f.write("\n=== [Action Comparison] ===\n")
                for i in range(min(len(actions_discrete), len(pred_actions_discrete))):
                    gt_token = self.decode_discrete_token(actions_discrete[i])
                    pred_token = self.decode_discrete_token(pred_actions_discrete[i])
                    match = "✓" if gt_token == pred_token else "✗"
                    f.write(f"[Action {i}] GT: '{gt_token}' | Pred: '{pred_token}' | Match: {match}\n")

        print(f"[BlocksWorldRenderer] Wrote text visualization to {txt_path}")

        # Create motion visualization
        if motion_data is not None or pred_motion_data is not None:
            self.visualize_motion(
                os.path.join(os.path.dirname(savepath), "motion_visualization.png"), 
                init_discrete, motion_data, pred_motion_data
            )

    def visualize_motion(self, save_path, init_discrete=None, motion_data=None, pred_motion_data=None):
        """
        Visualize motion trajectory with optional comparison.
        
        Args:
            save_path: path to save the visualization
            init_discrete: numpy array of init state tokens for extracting block positions
            motion_data: numpy array of ground truth motion data (shape: n_motion, 3) or None
            pred_motion_data: numpy array of predicted motion data (shape: n_motion, 3) or None
        """
        plt.figure(figsize=(12, 8))
        
        # Plot ground truth trajectory if available
        if motion_data is not None and len(motion_data) > 0:
            # Denormalize motion data
            denorm_motion = (motion_data * self.pred_std) + self.pred_mean
            
            # Plot trajectory
            time_vals = denorm_motion[:, 0]
            x_vals = denorm_motion[:, 1]
            z_vals = denorm_motion[:, 2]
            
            plt.plot(x_vals, z_vals, 'b-', linewidth=2, label='Ground Truth Trajectory')
            plt.scatter(x_vals, z_vals, c=time_vals, cmap='viridis', s=50, alpha=0.7)
            
            # Plot start and end points
            plt.scatter(x_vals[0], z_vals[0], color='green', s=100, label='GT Start')
            plt.scatter(x_vals[-1], z_vals[-1], color='red', s=100, label='GT End')
        
        # Plot predicted trajectory if available
        if pred_motion_data is not None and len(pred_motion_data) > 0:
            # Denormalize motion data
            denorm_pred_motion = (pred_motion_data * self.pred_std) + self.pred_mean
            
            # Plot trajectory
            pred_time_vals = denorm_pred_motion[:, 0]
            pred_x_vals = denorm_pred_motion[:, 1]
            pred_z_vals = denorm_pred_motion[:, 2]
            
            plt.plot(pred_x_vals, pred_z_vals, 'r--', linewidth=2, label='Predicted Trajectory')
            plt.scatter(pred_x_vals, pred_z_vals, c=pred_time_vals, cmap='plasma', s=30, alpha=0.7)
            
            # Plot start and end points
            plt.scatter(pred_x_vals[0], pred_z_vals[0], color='lime', s=100, label='Pred Start')
            plt.scatter(pred_x_vals[-1], pred_z_vals[-1], color='orange', s=100, label='Pred End')
        
        # Extract and plot block positions from init states if available
        if init_discrete is not None:
            block_coords = self._extract_block_coordinates(init_discrete)
            if block_coords:
                for block_name, pos in block_coords.items():
                    # Use the block position directly from blocks_coords
                    # Assuming pos is already denormalized
                    plt.scatter(pos[0], pos[1], color='brown', s=100, marker='s')
                    plt.annotate(block_name, (pos[0], pos[1]), xytext=(5, 5), 
                                textcoords='offset points', fontsize=12)
        
        # Add colorbar for time if trajectories are plotted
        if (motion_data is not None and len(motion_data) > 0) or \
           (pred_motion_data is not None and len(pred_motion_data) > 0):
            cbar = plt.colorbar()
            cbar.set_label('Time')
        
        # Set labels and title
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Z Coordinate', fontsize=12)
        plt.title('End Effector Trajectory Visualization', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"[BlocksWorldRenderer] Saved motion visualization to {save_path}")

    def visualize_batch(self, save_dir, batch_data, predictions=None):
        """
        Visualize all samples in a batch.
        
        Args:
            save_dir: directory to save visualizations
            batch_data: dictionary containing batch data components
            predictions: dictionary containing prediction components (optional)
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        batch_size = batch_data["init_discrete"].shape[0]
        
        for i in range(batch_size):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Extract sample data
            init_discrete = batch_data["init_discrete"][i]
            goal_discrete = batch_data["goal_discrete"][i]
            actions_discrete = batch_data.get("actions_discrete", None)
            motion_data = batch_data.get("motion_data", None)
            
            # Extract sample predictions if available
            pred_actions_discrete = None
            pred_motion_data = None
            if predictions is not None:
                pred_actions_discrete = predictions.get("actions_discrete", None)
                if pred_actions_discrete is not None:
                    pred_actions_discrete = pred_actions_discrete[i]
                
                pred_motion_data = predictions.get("motion_data", None)
                if pred_motion_data is not None:
                    pred_motion_data = pred_motion_data[i]
            
            # Visualize this sample
            self.visualize_sample(
                os.path.join(sample_dir, "visualization"),
                init_discrete,
                goal_discrete,
                None if actions_discrete is None else actions_discrete[i],
                None if motion_data is None else motion_data[i],
                pred_actions_discrete,
                pred_motion_data
            )
            
            # Save raw data for this sample
            np.save(os.path.join(sample_dir, "init_discrete.npy"), init_discrete)
            np.save(os.path.join(sample_dir, "goal_discrete.npy"), goal_discrete)
            
            if actions_discrete is not None:
                np.save(os.path.join(sample_dir, "actions_discrete.npy"), actions_discrete[i])
            
            if motion_data is not None:
                np.save(os.path.join(sample_dir, "motion_data.npy"), motion_data[i])
            
            if pred_actions_discrete is not None:
                np.save(os.path.join(sample_dir, "pred_actions_discrete.npy"), pred_actions_discrete)
            
            if pred_motion_data is not None:
                np.save(os.path.join(sample_dir, "pred_motion_data.npy"), pred_motion_data)

    def _extract_block_name(self, token):
        """
        Extract block name from a token string.
        
        Args:
            token: token string like 'onTableA', 'clearB', etc.
            
        Returns:
            str: block name (e.g., 'A', 'B') or None if not found
        """
        if not token or token == '[PAD]' or token == '[UNK]':
            return None
            
        # Extract last character if it's likely a block name
        if token[-1].isalpha() and token[-1].isupper():
            return token[-1]
            
        return None

    def _extract_block_coordinates(self, init_discrete):
        """
        Extract block coordinates from init_discrete.
        This is a placeholder that should be replaced with actual logic based on your data.
        
        Args:
            init_discrete: numpy array of init state tokens
            
        Returns:
            dict: mapping from block names to (x, z) coordinates
        """
        # This is a placeholder - you'll need to adapt this to your actual data structure
        block_coords = {}
        
        # Assuming block coordinates might be encoded in certain tokens
        # or in associated motion data that's not passed here
        # This is just an example implementation
        for i, row in enumerate(init_discrete):
            token = self.decode_discrete_token(row)
            block_name = self._extract_block_name(token)
            
            if block_name:
                # In a real implementation, you'd extract the actual coordinates
                # For now, just placing blocks in a grid
                block_coords[block_name] = (3 + i, 2 + i)
        
        return block_coords