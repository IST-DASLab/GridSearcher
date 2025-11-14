import os
from string import Template
from datetime import datetime
from gridsearcher import GridSearcher, TorchRunConfig, SchedulingConfig

def main():
    gs = GridSearcher(
        script='myscript.py',
        defaults=dict( # will be interpreted as a standalone parameter
            batch_size=128,
            epochs=100,
            lr_decay_at=[82, 123],
        ))

    gs.add_param('wandb_project', 'cifar10-training')

    # epochs and batch_size already exist in GridSearcher (added via defaults param in the constructor) and will be added to the template
    gs.add_param('wandb_group', Template('cifar10_rn18_adamw_E=${epochs}_bs=${batch_size}'))

    # The values for parameters lr, wd, beta1, beta2 and eps are not yet defined, but the Template won't fail
    # They MUST be defined in scheduling['params_values'] in the run method
    gs.add_param('wandb_job_type', Template('lr=${lr}_wd=${wd}_beta1=${beta1}_beta2=${beta2}_eps=${eps}'))
    gs.add_param('wandb_name', Template('seed=${seed}' + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")))

    commands = gs.run(
        # check the value for the parameter root_folder (--root_folder) in commands
        param_name_for_exp_root_folder='root_folder',
        exp_folder=Template(os.path.join('./results',
                                         '${wandb_project}',
                                         '${wandb_group}',
                                         '${wandb_job_type}',
                                         '${wandb_name}')),
        cfg_sched=SchedulingConfig(
            distributed_training=False,
            gpus=[0, 1, 2, 3, 4, 5, 6, 7],
            max_jobs_per_gpu=1,
            params_values=dict(
                seed=[1, 2, 3],
                lr=['1e-3', '1e-2'],
                wd=['1e-3', '1e-2'],

                # fixed parameters
                beta1=['0.9'],
                beta2=['0.999'],
                eps=['1e-8'],
            )
        ),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=True,
            master_addr='127.0.0.1',
            master_port=29500,
            rdzv_backend='static', # or c10d to enable address discovery via network
        ),
        debug=True, # only output the commands
    )
    print()
    print(f'Printing commands')
    print(commands)


if __name__ == '__main__':
    main()

