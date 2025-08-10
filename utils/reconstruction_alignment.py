"""
Reconstruction alignment utilities for finding common 3D points between subsequent reconstructions.
"""

import numpy as np
from typing import List, Tuple, Dict

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


def align_and_refine_reconstructions(recon_ref: pt.sfm.Reconstruction, 
                                    recon_qry: pt.sfm.Reconstruction,
                                    view_graph_matches: List[Tuple[int, int]]) -> Tuple[bool, Dict]:
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
    
    Returns:
        success: Whether the alignment was successful
        alignment_info: Dictionary with alignment statistics and results
    """
    print(f"🔄 Starting complete reconstruction alignment and refinement...")
    
    try:
        import os
        
        # Step 1: Find common tracks
        
        # Debug: show view names in both reconstructions
        ref_view_ids = sorted(recon_ref.ViewIds())
        qry_view_ids = sorted(recon_qry.ViewIds())
                
        points_ref, points_qry, track_pairs = pt.sfm.FindCommonTracksByFeatureInReconstructions(
            recon_ref, recon_qry, view_graph_matches
        )

        # select points with average distance from last camera position less than 10 meters
        last_camera_position = recon_ref.View(sorted(recon_ref.ViewIds())[-1]).Camera().GetPosition()
        med_distance = np.median(np.linalg.norm(points_ref - last_camera_position, axis=1))

        # select points with distance less than 10 meters
        valid_indices = np.linalg.norm(points_ref - last_camera_position, axis=1) < med_distance
        points_ref = np.array(points_ref)[valid_indices]
        points_qry = np.array(points_qry)[valid_indices]
        track_pairs = [track_pair for track_pair, valid in zip(track_pairs, valid_indices) if valid]
        
        # Step 2: Perform Sim3 alignment
        options = pt.sfm.Sim3AlignmentOptions()
        options.alignment_type = pt.sfm.Sim3AlignmentType.POINT_TO_POINT
        options.max_iterations = 5
        options.huber_threshold = 1.0
        options.perform_optimization = False
        options.verbose = False

        # Run Sim3 optimization
        summary = pt.sfm.OptimizeAlignmentSim3(points_qry, points_ref, options)

        if not summary.success:
            print("❌ Sim3 alignment failed")
            return False, {"error": "sim3_failed", "summary": summary}
        
        # Step 3: Transform the query reconstruction
        sim3_transform = pt.math.Sim3d.exp(summary.sim3_params)
        pt.sfm.TransformReconstruction4(recon_qry, sim3_transform.matrix())

        # Now generate priors before we bundle adjust the query reconstruction
        priors_set = 0
        
        for ref_view_idx, qry_view_idx in view_graph_matches:

            ref_view_id = ref_view_ids[ref_view_idx]
            qry_view_id = qry_view_ids[qry_view_idx]
            
            # Get views
            ref_view = recon_ref.View(ref_view_id)
            qry_view = recon_qry.MutableView(qry_view_id)

            # Get reference camera pose
            ref_camera = ref_view.Camera()
            ref_orientation = ref_camera.GetOrientationAsAngleAxis()
            ref_position = ref_camera.GetPosition()
            
            # Set orientation prior (5 * Identity matrix for covariance)
            orientation_covariance = 2.0 * np.eye(3)
            qry_view.SetOrientationPrior(ref_orientation, orientation_covariance)
            
            # Set position prior (100 * Identity matrix for covariance) 
            position_covariance = 25.0 * np.eye(3)
            qry_view.SetPositionPrior(ref_position, position_covariance)
            
            priors_set += 1
        
        
        # Step 5: Bundle adjust the query reconstruction
       
        ba_options = pt.sfm.BundleAdjustmentOptions()
        ba_options.use_orientation_priors = True
        ba_options.use_position_priors = True
        ba_options.max_num_iterations = 50
        ba_options.use_inner_iterations = False
        ba_options.use_mixed_precision_solves = True
        ba_options.max_num_refinement_iterations = 1
        ba_options.linear_solver_type = pt.sfm.LinearSolverType.DENSE_SCHUR
        ba_options.preconditioner_type = pt.sfm.PreconditionerType.IDENTITY
        ba_options.visibility_clustering_type = pt.sfm.VisibilityClusteringType.CANONICAL_VIEWS
        ba_options.use_homogeneous_point_parametrization = True
        ba_options.use_inverse_depth_parametrization = False
        ba_options.verbose = True

        pt.io.WriteReconstruction(recon_qry, "recon_qry_before_ba.sfm")
        
        ba_summary = pt.sfm.BundleAdjustReconstruction(ba_options, recon_qry)
        
        print(f"   Bundle adjustment completed: Success={ba_summary.success}, "
              f"Final cost={ba_summary.final_cost:.6f}")
        
        removed_tracks = pt.sfm.SetOutlierTracksToUnestimated(set(recon_qry.TrackIds()), 3, 0.25, recon_qry)
        print(f"   Removed {removed_tracks} tracks")


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
        
        return True, alignment_info
        
    except Exception as e:
        print(f"❌ Complete reconstruction alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": "exception", "message": str(e)}