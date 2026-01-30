
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
)
from Robot_Parkour.tasks.manager_based.robot_parkour.utils.rl_cfg import MyRslRlPpoActorCriticRecurrentCfg


@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    load_run="2026-01-29_05-45-53"
    load_checkpoint="model_2550.pt"
    num_steps_per_env = 120
    max_iterations = 1000
    obs_groups = {
            "teacher": ["policy", "physics", "visual"],
            "policy": ["policy", "depth"],
        }
    save_interval = 50
    experiment_name = "parkour_gru"
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std = 1.0,
        student_obs_normalization = True,
        teacher_obs_normalization = True,
        student_hidden_dims = [512, 256, 128],
        teacher_hidden_dims = [512, 256, 128],
        activation = "elu",
        rnn_type = "gru",
        rnn_hidden_dim = 256,
        rnn_num_layers = 1,
        teacher_recurrent = True
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name = "Distillation",
        num_learning_epochs = 5,
        learning_rate = 3e-4,
        gradient_length = 12,
        optimizer = "adamw",
        loss_type = "mse"
    )