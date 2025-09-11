import torch
import torch.nn.functional as F
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_from_paths
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import random
import glob
import lietorch as lt
import numpy as np
from scipy.spatial.transform import Rotation
from torch.amp import GradScaler

def cosine_similarity_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity loss between two tensors of shape [B, T, D].
    The loss encourages corresponding vectors to point in the same direction.

    Args:
        x (torch.Tensor): Tensor of shape [B, T, D]
        y (torch.Tensor): Tensor of shape [B, T, D]

    Returns:
        torch.Tensor: Scalar tensor representing the cosine similarity loss.
    """
    # Normalize feature vectors to unit length
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    # Compute cosine similarity for each vector pair (shape [B, T])
    cos_sim = torch.sum(x_norm * y_norm, dim=-1)

    # Loss is 1 - mean cosine similarity across all vectors
    loss = 1 - cos_sim.mean()

    return loss

# --- Dataset ---
class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir, train_fp8=False, min_chunk_size=3, max_chunk_size=13, num_chunks_per_sequence=50, extensions=(".png", ".jpg", ".jpeg")):
        """
        Initialize by collecting all subdirectories under root_dir as sequences.
        Each sequence directory is scanned recursively for image files.
        """
        self.root_dir = root_dir
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.extensions = tuple([e.lower() for e in extensions])
        # repeat the sequences 10 times
        self.sequences = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.sequences = self.sequences * num_chunks_per_sequence
        self.train_fp8 = train_fp8  

    def __len__(self):
        """
        Returns the total number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Load a random chunk of a sequence by scanning for images (.png/.jpg/.jpeg).
        Grayscale images are converted to RGB by load_images_from_paths.
        """
        sequence_path = self.sequences[idx]

        # Find all images recursively in the sequence directory
        image_paths = []
        for ext in self.extensions:
            image_paths.extend(glob.glob(os.path.join(sequence_path, f"**/*{ext}"), recursive=True))

        image_paths = sorted(image_paths)

        num_images = len(image_paths)
        if num_images < self.min_chunk_size:
            return torch.empty(0)

        # Determine chunk size (must be divisible by 8 for FP8 training)
        max_cap = min(self.max_chunk_size, num_images)
        if self.train_fp8:
            max_aligned = max_cap - (max_cap % 8)
            if max_aligned < 8:
                return torch.empty(0)

            min_cap = max(self.min_chunk_size, 8)
            min_aligned = ((min_cap + 7) // 8) * 8
            if min_aligned > max_aligned:
                return torch.empty(0)

            options = ((max_aligned - min_aligned) // 8) + 1
            chunk_size = min_aligned + 8 * random.randint(0, options - 1)
        else:
            chunk_size = random.randint(self.min_chunk_size, min(self.max_chunk_size, num_images))

        # Start index for this chunk
        start_index = random.randint(0, num_images - chunk_size)

        # Select the chunk of image paths
        chunk_paths = image_paths[start_index : start_index + chunk_size]

        # Load images as a tensor (handles grayscale->RGB and uniform resizing)
        imgs = load_images_from_paths(chunk_paths)
        return imgs

def collate_fn(batch):
    # Filter out any empty tensors from sequences that were too short
    batch = [b for b in batch if b.nelement() > 0]
    if not batch:
        return None
    # Since the batch size is 1 sequence, we just return the first item
    return batch[0]

# --- Helper Functions ---
def pose_matrix_to_vec(pose_mat_torch):
    """Converts a batch of 4x4 pose matrices (torch tensor) to a 7D vector 
    (translation + quaternion) representation."""
    # Ensure tensor is on CPU for numpy conversion
    pose_mat_np = pose_mat_torch.detach().cpu().numpy()
    
    # Extract rotation and translation
    rot_mat = pose_mat_np[:, :3, :3]
    trans = pose_mat_np[:, :3, 3]
    
    # Convert rotation matrix to quaternion (x, y, z, w)
    quat = Rotation.from_matrix(rot_mat).as_quat()
    
    # Concatenate translation and quaternion
    # lietorch expects [t, q] where t is (tx, ty, tz) and q is (qx, qy, qz, qw)
    pose_data_np = np.concatenate((trans, quat), axis=-1)
    
    # Convert back to a torch tensor on the original device
    return torch.tensor(pose_data_np, device=pose_mat_torch.device, dtype=pose_mat_torch.dtype)

def normalize_depth_for_tb(depth):
    """Normalizes a depth map for TensorBoard visualization."""
    # Add a channel dimension
    depth = depth.unsqueeze(0)
    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

# --- Loss Functions ---
def _pose_matrix_to_vec(pose_matrices):
    """Helper to convert a batch of 4x4 pose matrices (torch tensor) to 7D vectors (trans + quat)."""
    poses_np = pose_matrices.detach().cpu().numpy()
    trans = poses_np[:, :3, 3]
    # Scipy's as_quat returns (x, y, z, w)
    quats = Rotation.from_matrix(poses_np[:, :3, :3]).as_quat()
    # Concatenate translation and quaternion
    pose_vec_np = np.concatenate((trans, quats), axis=-1)
    return torch.tensor(pose_vec_np, dtype=pose_matrices.dtype, device=pose_matrices.device)

def camera_pose_loss_lietorch(student_poses, teacher_poses):
    """
    Calculates the camera pose loss using lietorch for SE(3) manifolds.
    Uses InitFromVec for robust initialization from 7D vectors (t, q).
    """
    # Reshape to (B*N, 4, 4)
    B, N, _, _ = student_poses.shape
    student_poses_flat = student_poses.view(B * N, 4, 4)
    teacher_poses_flat = teacher_poses.view(B * N, 4, 4)

    # Convert 4x4 matrices to 7D vectors (translation + quaternion)
    student_vec = _pose_matrix_to_vec(student_poses_flat)
    teacher_vec = _pose_matrix_to_vec(teacher_poses_flat)

    # Convert to lietorch SE3 objects using the 7D vector representation
    se3_student = lt.SE3.InitFromVec(student_vec)
    se3_teacher = lt.SE3.InitFromVec(teacher_vec)

    # Calculate the residual error in the tangent space (the Lie algebra)
    # This gives a 6D vector representing the error in rotation and translation
    error_vector = (se3_teacher.inv() * se3_student).log()

    # Penalize the deviation from zero in the tangent space using L1 loss
    return F.l1_loss(error_vector, torch.zeros_like(error_vector))

# --- Model Initialization ---
def initialize_student_from_teacher(teacher, student):
    """
    Initializes the weights of the student model's heads from the teacher model.
    Handles size mismatches by copying only the overlapping parts of the weights.
    """
    print("Initializing student model heads from teacher weights...")

    heads_to_initialize = ['point_head', 'conf_head', 'camera_head', 'point_decoder', 'conf_decoder', 'camera_decoder']

    for head_name in heads_to_initialize:
        teacher_head = getattr(teacher, head_name)
        student_head = getattr(student, head_name)

        teacher_dict = teacher_head.state_dict()
        student_dict = student_head.state_dict()

        # Create a new state dict for the student with potentially sliced weights
        new_student_dict = {}
        for name, teacher_param in teacher_dict.items():
            if name in student_dict:
                student_param = student_dict[name]
                if teacher_param.shape != student_param.shape:
                    print(f"  - Mismatch in '{head_name}.{name}': Teacher {teacher_param.shape}, Student {student_param.shape}")
                    
                    # Handle the mismatch by slicing the teacher's parameter
                    # This assumes the mismatch is in the input dimension (dim=1 for weights)
                    if teacher_param.dim() > 1: # For weight matrices
                        slice_dims = [slice(0, s) for s in student_param.shape]
                        new_student_dict[name] = teacher_param[tuple(slice_dims)]
                        print(f"    -> Copied slice {new_student_dict[name].shape} from teacher.")
                    else: # For bias vectors
                        new_student_dict[name] = teacher_param[:student_param.shape[0]]
                        print(f"    -> Copied slice {new_student_dict[name].shape} from teacher.")

                else:
                    # Shapes match, copy directly
                    print("Copying weights of", name)
                    new_student_dict[name] = teacher_param
            else:
                 print(f"  - Warning: Parameter '{name}' not found in student's '{head_name}'.")


        student_head.load_state_dict(new_student_dict)

    print("Student model heads initialized successfully.")
    return student

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_teacher = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
model_student = Pi3(decoder_size="small", global_merging=False, use_fp8_attention=False).to(device).train() # Set student to train mode
model_student.is_distillation = True
# --- Initialize Student from Teacher ---
model_student = initialize_student_from_teacher(model_teacher, model_student)

# --- TensorBoard Setup ---
writer = SummaryWriter('runs/tartanair_distillation')

# --- Hyperparameters ---
learning_rate = 5e-5
encoder_learning_rate = 5e-6
epochs = 200
batch_size = 1 # Each batch is one sequence
weight_decay = 5e-2
grad_clip_norm = 1.0
loss_weights = {
    "local_points": 1.0,
    "camera_poses": 1.0,
    "conf": 1.0,
    "hidden": 100.0
}


# --- Load Data ---
# Set this to your dataset root that contains multiple subfolders (one per sequence)
dataset = ImageSequenceDataset(root_dir='/media/steffen/Data/Pi3_distll_dataset/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# --- Optimizer ---
# print("Freezing all layers except the main decoder blocks for training.")
# for name, param in model_student.named_parameters():
#     #if "camera_decoder" in name or "conf_decoder" in name or "point_decoder" in name or "camera_head" in name or "conf_head" in name or "point_head" in name or "encoder" in name:
#     if "encoder" in name:
#         print(f"Freezing parameter: {name}")
#         param.requires_grad = False
#     else:
#         param.requires_grad = True
#         print(f"Training parameter: {name}")

# Pass only the trainable parameters to the optimizer
# set different learning rates for the encoder and the decoder
# get all parameters from the encoder
encoder_params = [p for p in model_student.encoder.parameters() if p.requires_grad]
# get all other parameters (avoid tensor equality by comparing identities)
encoder_param_ids = set(id(p) for p in encoder_params)
decoder_params = [p for p in model_student.parameters() if p.requires_grad and id(p) not in encoder_param_ids]
optimizer = optim.AdamW([{"params": encoder_params, "lr": encoder_learning_rate, "weight_decay": weight_decay}, 
                        {"params": decoder_params, "lr": learning_rate, "weight_decay": weight_decay}], 
                        weight_decay=weight_decay)


# --- Training Loop ---
print("Starting distillation training...")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
global_step = 0
scaler = GradScaler()
start_epoch = 0

# # --- Resume from checkpoint if available ---
# # Set resume_ckpt_path to a specific file to force resume, or leave None to auto-pick latest from 'checkpoints/'
# resume_ckpt_path = None
# if resume_ckpt_path is None and os.path.isdir('checkpoints'):
#     candidates = sorted(glob.glob(os.path.join('checkpoints', 'student_epoch_*.pth')))
#     if len(candidates) > 0:
#         resume_ckpt_path = candidates[-1]

# if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
#     print(f"Resuming from checkpoint: {resume_ckpt_path}")
#     ckpt = torch.load(resume_ckpt_path, map_location=device)
#     model_student.load_state_dict(ckpt['model_student'])
#     optimizer.load_state_dict(ckpt['optimizer'])
#     scaler.load_state_dict(ckpt['scaler'])
#     start_epoch = ckpt.get('epoch', 0)
#     global_step = ckpt.get('global_step', 0)
#     # Move optimizer state to the correct device
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if torch.is_tensor(v):
#                 state[k] = v.to(device)
#     print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

for epoch in range(start_epoch, epochs):
    for i, imgs in enumerate(dataloader):
        if imgs is None: # Skip batch if collate_fn returned None
            continue
        imgs = imgs.to(device)
        
        # --- Generate Teacher Targets ---
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                # Add a batch dimension -> (1, N, 3, H, W)
                teacher_results = model_teacher(imgs[None])
        torch.cuda.empty_cache()

        # --- Student Forward Pass ---
        with torch.amp.autocast('cuda', dtype=dtype):
            # Add a batch dimension -> (1, N, 3, H, W)
            student_results = model_student(imgs[None])

        # --- Calculate Losses ---
        # L1 loss for points
        loss_points = F.l1_loss(student_results['local_points'], teacher_results['local_points'])
        # Use lietorch for a geometrically correct camera pose loss
        loss_camera_poses = camera_pose_loss_lietorch(student_results['camera_poses'], teacher_results['camera_poses'])

        hidden_loss = cosine_similarity_loss(student_results['hidden'], teacher_results['hidden'])
        # L1 loss for confidence
        loss_conf = F.l1_loss(student_results['conf'], teacher_results['conf'])

        total_loss = (
            loss_weights["local_points"] * loss_points +
            loss_weights["camera_poses"] * loss_camera_poses +
            loss_weights["hidden"] * hidden_loss +
            loss_weights["conf"] * loss_conf
        )

        # --- Backpropagation (with Mixed Precision) ---
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(encoder_params, grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(decoder_params, grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        # --- TensorBoard Logging ---
        writer.add_scalar('Loss/Total', total_loss.item(), global_step)
        writer.add_scalar('Loss/Points', loss_points.item(), global_step)
        writer.add_scalar('Loss/Camera_Poses', loss_camera_poses.item(), global_step)
        writer.add_scalar('Loss/Confidence', loss_conf.item(), global_step)
        writer.add_scalar('Loss/Hidden', hidden_loss.item(), global_step)


        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {total_loss.item():.4f}")
            print(f"  - Points Loss: {loss_points.item():.4f}, Camera Pose Loss: {loss_camera_poses.item():.4f}, Conf Loss: {loss_conf.item():.4f}, Hidden Loss: {hidden_loss.item():.4f}")

            # Log depth images to TensorBoard
            with torch.no_grad():
                # Extract depth (z-coordinate, which is index 2)
                teacher_depth_first = teacher_results['local_points'][0, 0, :, :, 2]
                teacher_depth_last = teacher_results['local_points'][0, -1, :, :, 2]
                student_depth_first = student_results['local_points'][0, 0, :, :, 2]
                student_depth_last = student_results['local_points'][0, -1, :, :, 2]

                # Normalize for visualization
                teacher_depth_first_norm = normalize_depth_for_tb(teacher_depth_first)
                teacher_depth_last_norm = normalize_depth_for_tb(teacher_depth_last)
                student_depth_first_norm = normalize_depth_for_tb(student_depth_first)
                student_depth_last_norm = normalize_depth_for_tb(student_depth_last)

                # Create grids for comparison
                grid_first = torchvision.utils.make_grid([teacher_depth_first_norm, student_depth_first_norm])
                grid_last = torchvision.utils.make_grid([teacher_depth_last_norm, student_depth_last_norm])

                writer.add_image('Depth/First_Frame (Teacher vs Student)', grid_first, global_step)
                writer.add_image('Depth/Last_Frame (Teacher vs Student)', grid_last, global_step)
        
        global_step += 1

    # --- Checkpoint at end of each epoch ---
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'model_student': model_student.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }
    ckpt_path = os.path.join('checkpoints', f'student_epoch_{epoch+1:04d}.pth')
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


print("Distillation training complete!")
writer.close()
# You can save the student model here
torch.save(model_student.state_dict(), "student_model.pth")