"""Diffusion-PPO (group-normalized PPO) post-training for ViT-H G1 motion generation."""

POLICY_SINGLE_WINDOW = "ddpo_single_window"
POLICY_STITCHED_WINDOW = "ddpo_stitched_window"

REWARD_COMBINED_TRACK = "combined_track"
REWARD_DECOMPOSED_VEL = "decomposed_vel"
REWARD_STITCHED_SEAM = "stitched_seam"
