import math

import yaml
import argparse
import subprocess
from pathlib import Path
from copy import deepcopy
from datetime import date

qsub_submit_script = '''#!/bin/bash

#PBS -l walltime={walltime}:00:00,select=1:ncpus={ncpus}:ngpus={ngpus}{gpu_mem_arg}:mem={mem}
#PBS -N {job_index}.{name}
#PBS -A {allocation_code}
#PBS -m abe
#PBS -M {user_mail}
#PBS -o {run_dir}/output_{job_index}.txt
#PBS -e {run_dir}/error_{job_index}.txt

################################################################################

source /home/lemohama/.bashrc
module restore 2021_gpu

{cuda_visible_devices_expr}

conda activate old_jax
conda deactivate
conda activate old_jax

export JAX_ENABLE_X64=True

module load http_proxy

cd $PBS_O_WORKDIR

python run.py --config_path={config_path} --save_dir={run_dir}/run_{job_index}/ --specific_tag={specific_tag}
'''

# #SBATCH --gres=gpu:{ngpus}              # Number of GPU(s) per node
# #SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node

sbatch_submit_script = '''#!/bin/sh

#SBATCH --account={allocation_code}
#SBATCH --gres=gpu:{ngpus}              # Number of GPU(s) per node
#SBATCH --cpus-per-task={ncpus}         # CPU cores/threads
#SBATCH --mem={mem}               # memory per node
#SBATCH --job-name={job_index}.{name}
#SBATCH --time={walldays}-{wallhours}:59            # time (DD-HH:MM)
#SBATCH --mail-type=ALL
#SBATCH --output={run_dir}/slurm_output_{job_index}.txt

module restore new_gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.4.2/targets/x86_64-linux/lib/:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.4/cudnn/8.2.0/lib:/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.4.2/extras/CUPTI/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.4.2/lib64

{cuda_visible_devices_expr}

source /scratch/leamin/venvs/jax_new/bin/activate

export JAX_ENABLE_X64=True

wandb login a2ec141c7427b0c7830f3493d822be104c0cd6e3

WANDB__SERVICE_WAIT=300 python run.py --config_path={config_path} --save_dir={run_dir}/run_{job_index}/ --specific_tag={specific_tag}
'''


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_config(config_dict, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=None)


def add_explanation(explanation, key, value):
    return explanation + ('' if explanation == '' else '-__-') + '{}={}'.format(key, value)


def make_and_get(parent, child):
    if child not in parent:
        parent[child] = {}
    return parent[child]


def update_attribute_in_config(base_config, attribute, new_value, inplace=False):
    hierarchy = attribute.split('_--_')
    config = deepcopy(base_config)
    attribute = config
    for item in hierarchy[:-1]:
        attribute = make_and_get(attribute, item)
    attribute[hierarchy[-1]] = new_value
    return config


def generate_configs(base_config, hyperparam_config):
    def extract_configs_from_attribute_tree(tree, key, base_config, explanation):
        for branch in tree.keys():
            yield from extract_configs_from_branch(tree[branch], key, base_config, explanation=explanation)

    def extract_configs_from_branch(tree, parent_key, base_config, explanation):
        for i, value in enumerate(tree['__value']):

            new_config = update_attribute_in_config(base_config, parent_key, value)
            new_explanation = add_explanation(explanation, parent_key, value)
            correspondings = tree.get('__correspondings', None)
            if correspondings is not None:
                corr_keys = list(correspondings.keys())
                for attr in corr_keys:
                    val = correspondings[attr]['__value'][i]
                    new_config = update_attribute_in_config(new_config, attr, val)
                    new_explanation = add_explanation(new_explanation, attr, val)

            has_children = False
            for key in filter(lambda k: not k.startswith('__'), tree.keys()):
                has_children = True
                yield from extract_configs_from_attribute_tree(
                    tree[key], key, new_config, explanation=new_explanation)

            if not has_children:
                yield new_config, new_explanation

    for key in hyperparam_config.keys():
        yield from extract_configs_from_attribute_tree(hyperparam_config[key], key, base_config, explanation='')


def construct_task_name(explanation):
    name = ''

    def cond_query(label, test_query, true_val, false_val):
        if (label + "=" + test_query) in explanation:
            return true_val
        else:
            return false_val

    def extract_query(label, addition='', postval='_'):
        l = label + '='
        if l in explanation:
            index = explanation.index(l) + len(l)
            exp_begin = explanation[index:]
            if '-__-' in exp_begin:
                exp_end = exp_begin.index('-__-')
                val = exp_begin[:exp_end]
            else:
                val = exp_begin
            return addition + val + postval
        return addition + '???' + postval

    if 'data_--_seed' in explanation:
        name += extract_query('data_--_seed', addition='seed')  # seed

    if 'data_--_frac_train' in explanation:
        name += extract_query('data_--_frac_train', addition='ft')  # frac train

    if 'data_--_lookup_type' in explanation:
        name += extract_query('data_--_lookup_type', addition='lt')  # lookup type

    if 'data_--_embed_dim' in explanation:
        name += extract_query('data_--_embed_dim', addition='ed')  # embed dim

    if 'model_--_name' in explanation:
        name += extract_query('model_--_name', addition='model')  # name

    if 'model_--_width' in explanation:
        name += extract_query('model_--_width', addition='w')  # width

    if 'model_--_depth' in explanation:
        name += extract_query('model_--_depth', addition='d')  # depth

    if 'model_--_activation' in explanation:
        name += extract_query('model_--_activation', addition='ac')  # activation

    if 'train_--_lr' in explanation:
        name += extract_query('train_--_lr', addition='lr')

    if 'train_--_l2_reg' in explanation:
        name += extract_query('train_--_l2_reg', addition='reg')

    if 'train_--_criterion' in explanation:
        name += extract_query('train_--_criterion', 'cr')

    return name


if __name__ == '__main__':

    print('Warning! Should be run inside *root* directory of the project.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config_path', default='', help="Path to the base config file")
    parser.add_argument('--hyperparams_config_path', default='', help='Hyperparameters to choose from')
    parser.add_argument('--allocation_code', default='', help='Script allocation code')
    parser.add_argument('--user_mail', default='', help='Script user mail')
    parser.add_argument('--repetition_num', default=5, type=int, help='Script user mail')
    parser.add_argument('--cycle_time_in_hours', default=2, type=int, help='Script user mail')
    parser.add_argument('--pbs', dest='pbs', action='store_true')
    parser.add_argument('--narval', dest='narval', action='store_true')
    parser.add_argument('--sbatch', dest='pbs', action='store_false')
    parser.add_argument('--directory_modifier', default='', help='Storage directory of results modification')
    parser.add_argument('--force_random', dest='force_random', action='store_true')
    parser.add_argument('--random_mem', default='', help='Random memory requirement')
    parser.add_argument('--random_cpu', default=-1, type=int, help='Random cpu# requirement')
    parser.add_argument('--random_large_gpu', dest='random_large_gpu', help='Random needs large gpu?')
    parser.set_defaults(pbs=True, force_random=False, narval=False, random_large_gpu=False)
    args = parser.parse_args()

    base_config = load_config(args.base_config_path)
    hyperparams_config = load_config(args.hyperparams_config_path)
    new_configs = list(generate_configs(base_config, hyperparams_config))

    today = date.today().strftime('%Y_%m_%d')
    root_path = 'jobs/' + today + args.directory_modifier
    path = Path(root_path)
    path.mkdir(parents=True)

    for config, explanation in new_configs:

        name = construct_task_name(explanation)  # todo: should not result in same thing for different configs
        job_path = path.joinpath(name)
        job_path.mkdir()
        config_path = str(job_path.joinpath('config.yml').absolute())

        # is_random = 'acquisition=random' in explanation
        if args.force_random:
            is_random = True
        else:
            is_random = False
        save_config(config, config_path)
        for i in range(args.repetition_num):
            script_path = str(job_path.joinpath('script_{}.sh'.format(i)).absolute())
            script = qsub_submit_script if args.pbs else sbatch_submit_script
            time_hrs = args.cycle_time_in_hours
            # time_hrs = 8
            script = script.format(
                walltime=time_hrs,
                walldays=int(math.floor(time_hrs / 24)),
                wallhours=time_hrs,
                ncpus=args.random_cpu if args.random_cpu > 0 else
                ((12 if args.narval else 6) if is_random else (48 if args.narval else 24)),
                # ncpus=(24 if args.pbs else 48),
                ngpus=1 if is_random else 4,
                # ngpus=4,
                gpu_mem_arg='' if (is_random and not args.random_large_gpu) else ':gpu_mem=32gb',
                mem=args.random_mem if args.random_mem != '' else
                (('40g' if args.pbs else '122G' if args.narval else '40G') if is_random
                 else ('187gb' if args.pbs else '490G' if args.narval else '187G')),
                cuda_visible_devices_expr='' if is_random else 'export CUDA_VISIBLE_DEVICES=0,1,2,3',
                # cuda_visible_devices_expr='export CUDA_VISIBLE_DEVICES=0,1,2,3',
                allocation_code=args.allocation_code,
                user_mail=args.user_mail,
                config_path=config_path,
                run_dir=str(job_path.absolute()),
                name=name,
                job_index=i,
                specific_tag=args.directory_modifier[1:]
            )
            with open(script_path, 'w') as f:
                f.write(script)

    inp = input('Created all the files! Press any key to submit them...\n')

    for config, explanation in new_configs:
        name = construct_task_name(explanation)
        job_path = path.joinpath(name)
        config_path = str(job_path.joinpath('config.yml').absolute())
        for i in range(args.repetition_num):
            script_path = str(job_path.joinpath('script_{}.sh'.format(i)).absolute())
            if args.pbs:
                subprocess.check_call(['qsub', script_path])
            else:
                subprocess.check_call(['sbatch', script_path])
