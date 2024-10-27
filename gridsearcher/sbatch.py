import os
from typing import Dict

class SBATCH:
    def __init__(self,
                 script: str,
                 env_vars: Dict[str, str],
                 sbatch_args: Dict[str, str]):
        self.script = script
        self.env_vars = env_vars
        self.sbatch_args = sbatch_args

    def run(self, verbose=True):
        if len(self.env_vars) > 0:
            self.sbatch_args['export'] = ','.join(f'{k}={v}' for k, v in self.env_vars.items())
        args = ' '.join([
            f'--{key}={value}'
            for key, value in self.sbatch_args.items()
        ])

        cmd = f'sbatch {args} {self.script}'

        if verbose:
            print(cmd)

        os.system(cmd)


# class SBATCH_old:
#     def __init__(self,
#                  script,
#                  env_vars,
#                  job_name,
#                  nodelist=None,
#                  out_err_folder='slurm_output',
#                  ntasks=1,
#                  cpus_per_task=32,
#                  time='10-00:00:00',
#                  mem='100G',
#                  partition='gpu100',
#                  gres='gpu:H100:1'):
#
#         self.script = script
#         self.nodelist = nodelist
#         self.job_name = job_name
#         self.ntasks = ntasks
#         self.cpus_per_task = cpus_per_task
#         self.time = time
#         self.mem = mem
#         self.partition = partition
#         self.gres = gres
#         self.out_err_folder = out_err_folder
#         self.export = ','.join(f'{k}={v}' for k, v in env_vars.items())
#
#     def run(self, verbose=True):
#         args_list = [
#             f'--export={self.export}',
#             f'--job-name={self.job_name}',
#             f'--error={self.out_err_folder}/%j-%x.err',
#             f'--output={self.out_err_folder}/%j-%x.out',
#             f'--ntasks={self.ntasks}',
#             f'--cpus-per-task={self.cpus_per_task}',
#             f'--time={self.time}',
#             f'--mem={self.mem}',
#             f'--partition={self.partition}',
#             f'--gres={self.gres}'
#         ]
#
#         if self.nodelist is not None:
#             args_list.append(f'--nodelist={self.nodelist}')
#
#         args = ' '.join(args_list)
#         cmd = f'sbatch {args} {self.script}'
#
#         if verbose:
#             print(cmd)
#
#         os.system(cmd)
