"""
Reconstruction alignment utilities for finding common 3D points between subsequent reconstructions.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import torch

try:
    import pytheia as pt
    PYTHEIA_AVAILABLE = True
except ImportError:
    PYTHEIA_AVAILABLE = False
    print("⚠️  Warning: PyTheia not available, reconstruction alignment will be disabled")


def create_view_graph_matches(chunk_size: int, overlap_size: int) -> List[Tuple[int, int]]:
    """
    Create view graph matches for overlapping chunks.
    
    Args:
        chunk_size: Number of frames per chunk
        overlap_size: Number of overlapping frames between chunks
    
    Returns:
        List of (ref_view_id, qry_view_id) pairs for matching views
    """
    view_graph_matches = []
    
    # Create matches for the overlapping region
    # ref views: chunk_size - overlap_size to chunk_size - 1
    # qry views: 0 to overlap_size - 1
    for i in range(overlap_size):
        idx_ref = chunk_size - overlap_size + i
        idx_qry = i
        view_graph_matches.append((idx_ref, idx_qry))
    
    return view_graph_matches


def align_reconstructions_sim3(points_ref: np.ndarray, points_qry: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Align two reconstructions using Sim3 transformation (7DOF: scale, rotation, translation).
    
    Args:
        points_ref: Reference 3D points (N, 3)
        points_qry: Query 3D points to be aligned (N, 3)
    
    Returns:
        transformation_matrix: 4x4 transformation matrix to align query to reference
        alignment_info: Dictionary with alignment statistics
    """
    if len(points_ref) < 3 or len(points_qry) < 3:
        print("⚠️  Warning: Need at least 3 point correspondences for Sim3 alignment")
        return np.eye(4), {"error": "insufficient_points", "num_points": len(points_ref)}
    
    print(f"🔄 Aligning reconstructions using {len(points_ref)} common points...")
    
    try:
        import pytheia as pt
        
        # Use PyTheia's Sim3 alignment - POINT_TO_POINT for final alignment
        options = pt.sfm.Sim3AlignmentOptions()
        options.alignment_type = pt.sfm.Sim3AlignmentType.POINT_TO_POINT
        options.max_iterations = 100
        options.verbose = False
        
        # Convert points to the format expected by PyTheia
        source_points = points_qry.astype(np.float64)
        target_points = points_ref.astype(np.float64)
        
        # Run Sim3 optimization
        summary = pt.sfm.OptimizeAlignmentSim3(source_points, target_points, options)
        
        # Get transformation matrix
        transformation_matrix = pt.math.Sim3d.exp(summary.sim3_params).matrix()
        
        alignment_info = {
            "num_points": len(points_ref),
            "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost,
            "num_iterations": summary.num_iterations,
            "converged": summary.converged,
            "success": summary.success,
            "alignment_error": summary.alignment_error,
            "sim3_params": summary.sim3_params
        }
        
        print(f"✅ Sim3 alignment completed:")
        print(f"   Success: {summary.success}")
        print(f"   Final cost: {summary.final_cost:.6f}")
        print(f"   Iterations: {summary.num_iterations}")
        print(f"   Alignment error: {summary.alignment_error:.6f}")
        
        return transformation_matrix, alignment_info
        
    except Exception as e:
        print(f"❌ PyTheia Sim3 alignment failed: {e}")
        
        # Fallback to simple least squares alignment
        return _align_points_least_squares(points_ref, points_qry)


def _align_points_least_squares(points_ref: np.ndarray, points_qry: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Fallback alignment using least squares (similarity transformation).
    
    Args:
        points_ref: Reference 3D points (N, 3)
        points_qry: Query 3D points to be aligned (N, 3)
    
    Returns:
        transformation_matrix: 4x4 transformation matrix
        alignment_info: Dictionary with alignment statistics
    """
    print("🔄 Using fallback least squares alignment...")
    
    # Center the point sets
    centroid_ref = np.mean(points_ref, axis=0)
    centroid_qry = np.mean(points_qry, axis=0)
    
    centered_ref = points_ref - centroid_ref
    centered_qry = points_qry - centroid_qry
    
    # Compute cross-covariance matrix
    H = centered_qry.T @ centered_ref
    
    # SVD for rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation matrix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    scale = np.trace(S) / np.sum(centered_qry**2)
    
    # Compute translation
    t = centroid_ref - scale * R @ centroid_qry
    
    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = scale * R
    transformation_matrix[:3, 3] = t
    
    # Compute alignment error
    transformed_qry = (scale * R @ points_qry.T).T + t
    error = np.mean(np.linalg.norm(transformed_qry - points_ref, axis=1))
    
    alignment_info = {
        "num_points": len(points_ref),
        "scale": scale,
        "rotation_det": np.linalg.det(R),
        "alignment_error": error,
        "method": "least_squares"
    }
    
    print(f"✅ Least squares alignment completed: scale={scale:.3f}, error={error:.6f}")
    
    return transformation_matrix, alignment_info


def align_and_refine_reconstructions(recon_ref: pt.sfm.Reconstruction, 
                                    recon_qry: pt.sfm.Reconstruction,
                                    view_graph_matches: List[Tuple[int, int]],
                                    feature_match_threshold: float = 5.0,
                                    descriptor_match_threshold: float = 0.8,
                                    save_transformed_ply: str = None,
                                    save_debug_files: bool = True,
                                    output_dir: str = None,
                                    chunk_idx: int = 0) -> Tuple[bool, Dict]:
    """
    Complete reconstruction alignment with transformation and bundle adjustment refinement.
    
    This function replicates the C++ workflow:
    1. Find common tracks between reconstructions
    2. Perform Sim3 alignment 
    3. Transform the query reconstruction
    4. Set position and orientation priors for matching views
    5. Bundle adjust the query reconstruction
    
    Args:
        recon_ref: Reference reconstruction 
        recon_qry: Query reconstruction (will be modified in-place)
        view_graph_matches: List of (ref_view_id, qry_view_id) pairs for overlapping views
        feature_match_threshold: Distance threshold for feature matching (in pixels)
        descriptor_match_threshold: Distance threshold for descriptor matching
        save_transformed_ply: Optional path to save transformed reconstruction as PLY
        save_debug_files: Whether to save debug files (SFM and PLY) at each stage
        output_dir: Output directory for debug files
        chunk_idx: Chunk index for filename generation
    
    Returns:
        success: Whether the alignment was successful
        alignment_info: Dictionary with alignment statistics and results
    """
    print(f"🔄 Starting complete reconstruction alignment and refinement...")
    
    try:
        import pytheia as pt
        import os
        
        # Step 1: Find common tracks
        print("🔍 Step 1: Finding common tracks...")
        points_ref, points_qry, track_pairs = pt.sfm.FindCommonTracksByFeatureInReconstructions(
            recon_ref, recon_qry, view_graph_matches
        )
                
        if len(track_pairs) < 3:
            print(f"❌ Insufficient common tracks found: {len(track_pairs)} < 3")
            return False, {"error": "insufficient_tracks", "num_tracks": len(track_pairs)}
        
        # Save original reconstructions for debugging (before any transformation)
        if save_debug_files and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save original reference reconstruction
                ref_sfm_path = os.path.join(output_dir, f"chunk_{chunk_idx-1:06d}_01_original_ref.sfm")
                ref_ply_path = os.path.join(output_dir, f"chunk_{chunk_idx-1:06d}_01_original_ref.ply")
                
                pt.io.WriteReconstruction(recon_ref, ref_sfm_path)
                color_ref = [255, 0, 0]  # Red color for reference
                pt.io.WritePlyFile(ref_ply_path, recon_ref, color_ref, 0)
                
                # Save original query reconstruction  
                qry_sfm_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_01_original_qry.sfm")
                qry_ply_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_01_original_qry.ply")
                
                pt.io.WriteReconstruction(recon_qry, qry_sfm_path)
                color_qry = [0, 0, 255]  # Blue color for query
                pt.io.WritePlyFile(qry_ply_path, recon_qry, color_qry, 0)
                
                print(f"💾 Saved original reference reconstruction: {ref_sfm_path}")
                print(f"💾 Saved original query reconstruction: {qry_sfm_path}")
            except Exception as e:
                print(f"⚠️  Failed to save original reconstructions: {e}")
        
        # Step 2: Perform Sim3 alignment
        print("🔧 Step 2: Performing Sim3 alignment...")
        options = pt.sfm.Sim3AlignmentOptions()
        options.alignment_type = pt.sfm.Sim3AlignmentType.POINT_TO_POINT
        options.max_iterations = 10
        options.verbose = False

        # Run Sim3 optimization
        summary = pt.sfm.OptimizeAlignmentSim3(points_qry, points_ref, options)
        
        print(f"   Sim3 Summary: Success={summary.success}, Cost={summary.final_cost:.6f}, "
              f"Iterations={summary.num_iterations}, Error={summary.alignment_error:.6f}")
        
        if not summary.success:
            print("❌ Sim3 alignment failed")
            return False, {"error": "sim3_failed", "summary": summary}
        
        # Step 3: Transform the query reconstruction
        print("🔧 Step 3: Transforming query reconstruction...")
        sim3_transform = pt.math.Sim3d.exp(summary.sim3_params)
        pt.sfm.TransformReconstruction4(recon_qry, sim3_transform.matrix())
        
        # Save transformed reconstruction (before bundle adjustment) for debugging
        if save_debug_files and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save transformed reconstruction as SFM and PLY
                transformed_sfm_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_02_transformed.sfm")
                transformed_ply_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_02_transformed.ply")
                
                pt.io.WriteReconstruction(recon_qry, transformed_sfm_path)
                color = [0, 255, 255]  # Cyan color for transformed
                pt.io.WritePlyFile(transformed_ply_path, recon_qry, color, 0)
                
                print(f"💾 Saved transformed reconstruction: {transformed_sfm_path}")
                print(f"💾 Saved transformed point cloud: {transformed_ply_path}")
            except Exception as e:
                print(f"⚠️  Failed to save transformed reconstruction: {e}")
        
        # Save transformed reconstruction if explicitly requested
        if save_transformed_ply:
            print(f"💾 Saving transformed reconstruction to: {save_transformed_ply}")
            color = [0, 255, 255]  # Cyan color
            pt.io.WritePlyFile(save_transformed_ply, recon_qry, color, 0)
        
        # Step 4: Set position and orientation priors for matching views
        print("🔧 Step 4: Setting position and orientation priors...")
        
        # Get view IDs from both reconstructions
        ref_view_ids = recon_ref.ViewIds()
        qry_view_ids = recon_qry.ViewIds()
        
        # Now generate priors before we bundle adjust the query reconstruction
        priors_set = 0
        for ref_view_idx, qry_view_idx in view_graph_matches:
            # Check if view indices are valid
            if ref_view_idx >= len(ref_view_ids) or qry_view_idx >= len(qry_view_ids):
                continue
            
            ref_view_id = ref_view_ids[ref_view_idx]
            qry_view_id = qry_view_ids[qry_view_idx]
            
            # Get views
            ref_view = recon_ref.View(ref_view_id)
            qry_view = recon_qry.MutableView(qry_view_id)
            
            if not ref_view.IsEstimated() or not qry_view.IsEstimated():
                continue
            
            # Get reference camera pose
            ref_camera = ref_view.Camera()
            ref_orientation = ref_camera.GetOrientationAsAngleAxis()
            ref_position = ref_camera.GetPosition()
            
            # Set orientation prior (5 * Identity matrix for covariance)
            orientation_covariance = 2.0 * np.eye(3)
            qry_view.SetOrientationPrior(ref_orientation, orientation_covariance)
            
            # Set position prior (100 * Identity matrix for covariance) 
            position_covariance = 10.0 * np.eye(3)
            qry_view.SetPositionPrior(ref_position, position_covariance)
            
            priors_set += 1
        
        print(f"   Set priors for {priors_set} matching views")
        
        # Step 5: Bundle adjust the query reconstruction
        print("🔧 Step 5: Bundle adjusting query reconstruction...")
        
        ba_options = pt.sfm.BundleAdjustmentOptions()
        ba_options.use_orientation_priors = True
        ba_options.use_position_priors = True
        ba_options.max_num_iterations = 25
        ba_options.verbose = True
        
        ba_summary = pt.sfm.BundleAdjustReconstruction(ba_options, recon_qry)
        
        print(f"   Bundle adjustment completed: Success={ba_summary.success}, "
              f"Final cost={ba_summary.final_cost:.6f}")
        
        # Save final refined reconstruction for debugging
        print("output_dir", output_dir)
        if save_debug_files and output_dir:
            try:
                # Save final refined reconstruction as SFM and PLY
                final_sfm_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_03_refined.sfm")
                final_ply_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_03_refined.ply")
                
                pt.io.WriteReconstruction(recon_qry, final_sfm_path)
                color = [0, 255, 0]  # Green color for final refined
                pt.io.WritePlyFile(final_ply_path, recon_qry, color, 0)
                
                print(f"💾 Saved refined reconstruction: {final_sfm_path}")
                print(f"💾 Saved refined point cloud: {final_ply_path}")
            except Exception as e:
                print(f"⚠️  Failed to save refined reconstruction: {e}")
        
        # Compile results
        alignment_info = {
            "success": True,
            "num_common_tracks": len(track_pairs),
            "sim3_summary": {
                "success": summary.success,
                "final_cost": summary.final_cost,
                "alignment_error": summary.alignment_error,
                "sim3_params": summary.sim3_params
            },
            "priors_set": priors_set,
            "bundle_adjustment": {
                "success": ba_summary.success,
                "final_cost": ba_summary.final_cost,
                "initial_cost": ba_summary.initial_cost
            }
        }
        
        print(f"✅ Complete reconstruction alignment successful!")
        print(f"   Common tracks: {len(track_pairs)}")
        print(f"   Sim3 error: {summary.alignment_error:.6f}")
        print(f"   BA final cost: {ba_summary.final_cost:.6f}")
        
        return True, alignment_info
        
    except Exception as e:
        print(f"❌ Complete reconstruction alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": "exception", "message": str(e)}