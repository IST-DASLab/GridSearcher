# GridSearcher 𖣯🔍

---

GridSearcher is a pure Python project designed to simplify the process of running grid searches for Machine Learning 
projects. It serves as a robust alternative to traditional bash scripts, providing a more flexible and user-friendly 
way to manage and execute multiple programs in parallel. 

⚠️ **It is designed for systems where users have direct SSH access 
to machines and can run their python scripts right away.**

# Features ✨󠁇󠁇󠁇
- **Grid Search Made Easy:** Define parameter grids effortlessly and the cartesian product of your hyper-parameters 
will be computed automatically and an instance of your script will be run for all possible combinations.
- **Parallel Execution:** Run multiple programs concurrently, maximizing your computational resources.
- **GPU Scheduling:** Built-in GPU allocation ensures efficient use of available GPUs. Specify the number of GPUs and 
jobs per GPU, and **GridSearcher** will handle the rest
- **Flexible Configuration:** Easily control the number of parallel jobs and GPU assignments through a scheduling 
dictionary.
- **Pure Python:** No more dealing with complex bash scripts. **GridSearcher** is written entirely in Python, making it 
easy to integrate into your existing Python workflows.

# Why GridSearcher? 🤔
- **User-Friendly:** Simplifies the setup and execution of grid searches, allowing you to focus on your Machine 
Learning models.
- **Efficient Resource Management:** Optimize the use of your GPUs and computational resources.
- **Pythonic Approach:** Seamlessly integrates with your Python projects and leverages Python's rich ecosystem.
- **Direct SSH Access:** Ideal for systems where users have direct SSH access to machines, providing a straightforward 
setup and execution process without the need for SLURM or other workload managers, ensuring a smooth and efficient operation.
 
# Installation 🛠️
Install **GridSearcher** via pip:

```shell
pip install gridsearcher
```

# How to use GridSearcher?

---

We provide a minimal working example in the file [example.py](https://github.com/IST-DASLab/GridSearcher/blob/main/example.py).
Just set `debug=True` with `debug=False` in the `run` method call to run on GPUs. The output of `example.py` is the following:

```shell 
GridSearcher PID: 8940
command 1: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-2_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=1_2024-06-19_23-04-23 --seed 1 --lr 1e-2 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-2_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8/seed=1_2024-06-19_23-04-23
command 2: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-2_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=1_2024-06-19_23-04-23 --seed 1 --lr 1e-2 --wd 1e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-2_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8/seed=1_2024-06-19_23-04-23
command 3: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-3_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=1_2024-06-19_23-04-23 --seed 1 --lr 1e-3 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-3_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8/seed=1_2024-06-19_23-04-23
command 4: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-3_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=1_2024-06-19_23-04-23 --seed 1 --lr 1e-3 --wd 1e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-3_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8/seed=1_2024-06-19_23-04-23
command 5: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-2_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=2_2024-06-19_23-04-23 --seed 2 --lr 1e-2 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-2_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8/seed=2_2024-06-19_23-04-23
command 6: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-2_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=2_2024-06-19_23-04-23 --seed 2 --lr 1e-2 --wd 1e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-2_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8/seed=2_2024-06-19_23-04-23
command 7: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-3_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=2_2024-06-19_23-04-23 --seed 2 --lr 1e-3 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-3_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8/seed=2_2024-06-19_23-04-23
command 8: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-3_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=2_2024-06-19_23-04-23 --seed 2 --lr 1e-3 --wd 1e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-3_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8/seed=2_2024-06-19_23-04-23
command 9: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-2_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=3_2024-06-19_23-04-23 --seed 3 --lr 1e-2 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-2_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8/seed=3_2024-06-19_23-04-23
command 10: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-2_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=3_2024-06-19_23-04-23 --seed 3 --lr 1e-2 --wd 1e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-2_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8/seed=3_2024-06-19_23-04-23
command 11: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-3_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=3_2024-06-19_23-04-23 --seed 3 --lr 1e-3 --wd 1e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-3_wd=1e-2_beta1=0.9_beta2=0.999_eps=1e-8/seed=3_2024-06-19_23-04-23
command 12: python3 myscript.py --batch_size 128 --epochs 100 --lr_decay_at 82 123 --wandb_project cifar10-training --wandb_group cifar10_rn18_adamw_E=100_bs=128 --wandb_job_type lr=1e-3_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8 --wandb_name seed=3_2024-06-19_23-04-23 --seed 3 --lr 1e-3 --wd 1e-3 --beta1 0.9 --beta2 0.999 --eps 1e-8 --root_folder ./results/cifar10-training/cifar10_rn18_adamw_E=100_bs=128/lr=1e-3_wd=1e-3_beta1=0.9_beta2=0.999_eps=1e-8/seed=3_2024-06-19_23-04-23
```

# SBATCH wrapper for SLURM (NEW in version 1.0.4)
We also added a wrapper for SBATCH that allows running SLURM jobs directly from Python!

```python
SBATCH(
    script='script.sh', # the shell script that will be launched using sbatch
    env_vars=dict( # these variables will be set in the --export argument of sbatch and will be available in the h100-eval.sh script
        root=ROOT,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_job_type=wandb_job_type,
        wandb_name=wandb_name,
        task='imagenet',
        batch_size=128,
    ),
    sbatch_args={ # the variables below will be added to the sbatch command (e.g. --job_name)
        'exclude': 'machine-1,machine-2',
        'job-name': f'job-name',
        'error': 'slurm_output/%j-%x.err',
        'output': 'slurm_output/%j-%x.out',
        'ntasks': 1,
        'cpus-per-task': 20,
        'time': '1-00:00:00',
        'mem': '200G',
        'partition': 'gpu',
        'gres': 'gpu:H100:1',
    }
).run()
```

# Contribute 🤝

---

We welcome contributions! If you have suggestions for new features or improvements, feel free to open an issue or submit a 
pull request.

# Versions history:
- **1.1.2** @ 2025-01-23:
  - added `create_state_finished` parameter to control whether the process writes the file `state.finished` when returns with error code 0 or not
- **1.1.1**: fixed import issues
- **1.1.0**: removed specific arguments and replaced them with dictionaries to offer flexibility to use any SBATCH params
- **1.0.4**: added SBATCH class, which can be used in a completely separated manner from GridSearcher, allowing running slurm jobs from python
- **1.0.3**: do not check whether the script ends with `.py` extension anymore
- **1.0.2**: checking the return code of `os.system` and create file `state.finished` only if `code == 0`
- **1.0.1**: added assert statement to make sure that all values in the `scheduling["params_values"]` are of type list
- **1.0.0**: added initial project
