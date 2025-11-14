import multiprocessing as mp
from string import Template
from itertools import product
from copy import deepcopy
from .tools import *
from .configs import SchedulingConfig, TorchRunConfig

class GridSearcher:
    def __init__(self,
                 script,
                 defaults=None,
                 exe: GSExe = GSExe.PYTHON,
                 key_value_separator: GSKeyValSep = GSKeyValSep.SPACE,
                 use_dashes=True):
        """
        Description:
            Creates a GridSearcher object to run experiments using a grid.
            :param script: absolute path to the python script to run
            :param exe: instance of GridSearcherExe
            :param key_value_separator: separator between keys and values when launching the program, can be space or equal sign
            :param use_dashes: whether to use dashes for keys or not when creating the command
            :param defaults: default cmd arguments that are usually unchanged, for example batch size
        """
        validate_constructor_params(script, exe, key_value_separator)
        self.script = script
        self.exe = exe.value
        self.key_value_separator = key_value_separator.value
        self.use_dashes = use_dashes
        self.exp_folder_template = None

        if defaults is not None:
            for k, v in defaults.items():
                self.add_param(k, v)

    def add_from_yaml(self, yaml_file):
        """
        Description:
            This method allows adding parameters to a GridSearcher object from a YAML file.

        Args:
            :param yaml_file: absolute path to the YAML file
        """
        if os.path.isfile(yaml_file):
            data = read_yaml(yaml_file)
            for k, v in data.items():
                self.add_param(key=k, value=v)

    def add_param(self, key, value):
        """
        Description:
            Adds a parameter to the command line args list in __dict__ in the form "--name value"

        Args:
            :param key: The name of the parameter, which will be preceded by two dashes ("--") if use_dashes=True
            :param value: The value for the parameter. If it's a Template, it will be filled with the values of already existing parameters
        """

        if value is not None:
            key = forward_key_replace(key)
            if isinstance(value, list):
                value = ' '.join(map(str, value))
                setattr(self, f'_{key}', value)
            elif isinstance(value, Template):
                setattr(self, f'template_{key}', deepcopy(value))
                setattr(self, f'_{key}', None)
                # setattr(self, f'_{name}', self._fill_template(value)) # to avoid dict-changed-size error
            else:
                setattr(self, f'_{key}', value)

    def run(self,
            param_name_for_exp_root_folder: str,
            exp_folder: Template,
            cfg_sched: SchedulingConfig,
            cfg_torchrun: TorchRunConfig,
            debug: bool = False,
            create_state_finished: bool = True):
        """
        Description:
            Runs the GridSearcher using the provided configuration.

        Args:
            :param param_name_for_exp_root_folder: set this parameter to the name of the cmd
                argument for the output directory of your script. For example, if your script
                uses "output_dir" directory for writing checkpoints, then set this parameter
                to "output_dir" and it will automatically be filled with the value of `exp_folder`,
                being equivalent to "--output_dir=exp_folder". Note that `exp_folder` parameter can
                be a template, which might make it easy for you to embed some hyper-parameters to this folder.
            :param exp_folder: absolute path of the root folder where you want your experiments to be
            :param debug: print commands if True, run commands if False
            :param create_state_finished: whether to create the file "state.finished" or not
            :param cfg_sched:an object of type SchedulingConfig
            :param cfg_torchrun: an object of type TorchRunConfig
        """
        n_gpus = len(cfg_sched.gpus)
        if cfg_sched.distributed_training: # use all GPUs for a single run (distributed training)
            n_workers = cfg_sched.max_jobs_per_gpu
        else: # use GPUs to run one experiment per GPU
            n_workers = n_gpus * cfg_sched.max_jobs_per_gpu

        self.exp_folder_template = deepcopy(exp_folder)
        os.system('cls' if on_windows() else 'clear')
        print(f'GridSearcher PID: {os.getpid()}')

        cmds = [] # will store all commands to be run (as strings)
        cmds_dict = [] # will store dictionaries containing key:value pairs of hyper-parameters
        root_folders = [] # each command will have a separate value for the output folder specified by param_name_for_exp_root_folder

        params = list(cfg_sched.params_values.keys()) # if we do grid search for lr and wd, then params will contain "lr" and "wd"

        cart_prod = list(product(*list(cfg_sched.params_values.values()))) # perform the cartesian product of all hyper-parameters
        for i, values in enumerate(cart_prod):
            # for each element of cartesian product (contained in `values`), we have to (follow the steps given by numbers):

            # step 1: add the values for hyper-parameter optimization (HPO)to GridSearcher object
            for k, v in zip(params, values):
                self.add_param(k, v)

            # step 2: after filling in the values for HPO, go through all templated fields and fill them with the new values
            for k, v in self.__dict__.items():
                if k.startswith('template_'): # template parameters have "template_" prefix
                    tmpl_filled = self._fill_template(v) # this returns string or the same template if there are no matching values
                    self.__dict__[k.replace('template', '')] = tmpl_filled # only replace "template" prefix and keep "_" prefix

            # step 3: if the cartesian product element `values` contains some values templated in param_name_for_exp_root_folder, fill them
            root_folder = self._create_root_arg(
                param_name_for_exp_root_folder,
                self.exp_folder_template)

            # create
            p = {k: v for k, v in self.__dict__.items() if k.startswith('_')}

            # step 4: add metadata
            cmds.append(self._build_command()) # add the string commands
            cmds_dict.append(p) # add current parameters from the GridSearch object's internal dictionary, as key:value dictionary
            root_folders.append(root_folder) # add root folders (e.g., output_dir based on the example for param_name_for_exp_root_folder)
        # end cartesian product loop

        if debug: # only print commands to check for correctness, do not run anything
            for index, cmd in enumerate(cmds):
                print(f'command {index+1}: {self.exe}', cmd.replace('\\', '/'))
            return cmds
        else: # actually run the processes for hyper-parameter optimizations
            manager = mp.Manager()
            gpu_processes_count = manager.dict() # shared dict, where key=gpu_id and value=how many processes were run that GPU id
            for gpu in cfg_sched.gpus:
                gpu_processes_count[gpu] = 0

            """
                We will write the file `state.finished` to the folder specified by param_name_for_exp_root_folder when the experiment ends.
                If some experiments were already run and have a file state.finished, they will not be run again and the experiment will be 
            skipped.
            """

            cmds_total = 0 # how many program instances (commands to run) were generated by the cartesian product of hyper-parameters grid
            cmds_runnable = 0 # some runs might have already been run

            params_list = [] # will hold the parameters to be send to the MultiProcessing worker to run experiments
            for cmd, root, cmd_dict in zip(cmds, root_folders, cmds_dict):
                cmds_total += 1
                if os.path.isfile(os.path.join(root, 'state.finished')):
                    continue
                cmds_runnable += 1
                params_list.append([cmd, root, cmd_dict])

            console_info = f'Commands:\tRunnable: {cmds_runnable}\tFinished: {cmds_total - cmds_runnable}\tTotal: {cmds_total}'
            print(console_info)

            pause_process(seconds=5, message=f'Waiting 5 seconds before running GridSearcher...')

            if cmds_runnable > 0:
                # transform the params list (previously initialized)
                params_list = [
                    (
                        self.exe, # python or composer
                        index, # run index, zero based
                        cmd,
                        root,
                        cmd_dict,
                        gpu_processes_count, # shared dict where key=gpu id and value=processes cound per gpu id
                        cfg_sched,
                        cfg_torchrun,
                        create_state_finished
                    )
                    for index, (cmd, root, cmd_dict) in enumerate(params_list)
                ]

                with mp.Pool(processes=n_workers) as pool:
                    lock_release() # make sure there are no lock files on disk before starting pool
                    pool.map(func=waiting_worker, iterable=params_list)

            print('GridSearcher ended. Summary:')
            print(console_info)

    def _create_root_arg(self, param_name_for_exp_root_folder, exp_folder):
        """
        Description:
            This method fills in the exp_folder template and adds it to the __dict__ to be used as output directory.
            For example, if your python script requires setting the parameter "output_dir" to specify where to save checkpoints,
            then you can specify param_name_for_exp_root_folder="output_dir" and the value for this parameter will be given by
            the value of `exp_folder` (which can be a template).

        Args:
            :param param_name_for_exp_root_folder: a string containing the argument name for the experiment root folder
        """
        exp_root_folder = self._fill_template(exp_folder) # fill in the template
        self.add_param(param_name_for_exp_root_folder, exp_root_folder) # add to the __dict__
        return exp_root_folder

    def _fill_template(self, template):
        """
        Description:
            This method fills in the `template` given as parameter with values stored in `self.__dict__`.
            If the template uses a variable which is not in `self.__dict__` yet, the method returns the template again because that parameter
            is expected to be set in the next steps when iterating through the elements of the cartesian product.

        Args:
            :param template: Template or string. If Template, we try to fill it. If string, then it's immediately returned.
            :return: a string containing the template with substitutions or the same template if no substitutions could be made
        """
        if isinstance(template, str):
            return template

        try:
            d = {}
            for key, val in self.__dict__.items():
                if key.startswith('_'): # only look at the keys in __dict__ that start with underscore (those were added by GridSearcher)
                    d[key[1:]] = val # erase the underscore at position 0

            """
                Call the substitute method for the template:
                - if we have the template "lr=${lr}" and we have the dictionary d = { "lr": 1e-3 }, then the template will be fully filled
                and the returned value will be "lr=1e-3"
                
                All parameters in the template should be present in the dictionary d. If the substituted string still contains the 
                delimiter (e.g. the dollar sign character), it means that the dictionary d did not contain all required values to fill in
                the template and in that case we raise a RuntimeError
                
                We can also use safe_substitute
            """
            substituted = template.substitute(**d)
            # dlmtr = template.delimiter
            # if dlmtr in substituted: # after substitution, the delimiter should disappear. If it's still present, it's an issue
            #     raise RuntimeError(f'Could not replace the variables marked by {dlmtr}: {substituted}. '
            #                        f'Please specify them in the scheduling["params_values"] dictionary')
            return substituted
        except KeyError as e:
            print(f'[TemplateError] {str(e)}, {e.__cause__}')
            return template

    def _build_command(self):
        """
        Description:
            This method actually builds the list of parameters of form "--key=value" or "--key value".
            It can handle boolean parameters: if key=True, then only "--key" is added, for example "--bf16" sets bf16=True in the script
            We also support parameters that have dot or dash characters, such as "--train.lr" or "--optimizer-name". In these cases,
            the parameters should be specified as "trainDOTlr" or "optimizerDASHname" in the keys of the scheduling["params_values"]
        """
        params = []
        dash_or_not = '--' if self.use_dashes else '@'

        for k, v in self.__dict__.items(): # iterate through __dict__
            if k.startswith('_'): # process parameters with underscore prefix because these are the ones that we added to GridSearcher
                if isinstance(v, bool): # we have a boolean parameter without a value, but its presence or absence means True or False
                    if v:
                        params.append(f'{dash_or_not}{backward_key_replace(k)}') # replace
                elif isinstance(v, Template):
                    # elif isinstance(v, str) and '${' in v:
                    params.append(f'{dash_or_not}{backward_key_replace(k)}{self.key_value_separator}{self._fill_template(v)}')
                else:
                    params.append(f'{dash_or_not}{backward_key_replace(k)}{self.key_value_separator}{str(v)}')
        params = ' '.join(params).replace(f'{dash_or_not}_', f'{dash_or_not}').replace('@', '')
        return f'{self.script} {params}'

    def __getattr__(self, item):
        return self.__dict__[f'_{item}']
