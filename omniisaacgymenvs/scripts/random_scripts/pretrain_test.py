# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

import hydra
from omegaconf import DictConfig

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import onnx
import onnxruntime as ort

import datetime
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import math

import pdb


class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        
        
    def forward(self,input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        #print(input_dict)
        #output_dict = self._model.a2c_network(input_dict)
        #input_dict['is_train'] = False
        #return output_dict['logits'], output_dict['values']
        return self._model.a2c_network(input_dict)

class RLGTrainer():
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        self.cfg_dict["task"]["test"] = self.cfg.test

        # register the rl-games adapter to use inside the runner
        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: env
        })

        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self):
        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())
        runner.load(self.rlg_config_dict)
        runner.reset()

        # dump config dict
        experiment_dir = os.path.join('runs', self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        agent = runner.create_agent()
        #agent.restore(self.cfg.checkpoint)
        model = ModelWrapper(agent.model)

        mse = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 51
        batch_size = 16
        ranges = torch.ones(batch_size, 36).to(agent.device)

        if self.cfg.checkpoint == '':
            # Training loop
            for epoch in range(num_epochs):
                des_velocity = torch.rand(batch_size, 2).to(agent.device) * 2 - 1
                base_vel = torch.rand(batch_size, 3).to(agent.device) * 2 - 1
                old_action = torch.rand(batch_size, 3).to(agent.device) * 2 - 1
                old_old_action = torch.rand(batch_size, 3).to(agent.device) * 2 - 1

                headings = torch.atan2(des_velocity[:,1], des_velocity[:,0]).to(agent.device).unsqueeze(1)
                headings = torch.where(headings > math.pi, headings - 2 * math.pi, headings)
                headings = torch.where(headings < -math.pi, headings + 2 * math.pi, headings)

                #obs = torch.cat((ranges, des_velocity, base_vel), 1).to(agent.device)
                obs = torch.cat((ranges, headings, des_velocity, base_vel, old_action, old_old_action), 1).to(agent.device)

                target = torch.cat((des_velocity, torch.zeros_like(base_vel[:,2:])), 1).to(agent.device)
                # Forward pass
                outputs = model({'obs': obs})
                loss = mse(outputs[0], target) #+ 1 / (outputs[1])**2 + torch.log(outputs[1])
                value = -0.2 * abs(headings) - 0.002 * torch.norm(outputs[0] - old_action, dim=1)**2
                loss += mse(outputs[2], value)
                loss = torch.mean(loss)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{num_epochs}')
                    print(f'Loss: {loss.item()}')
                    print(f'mu: {outputs[0][:1,:]}')
                    print(f'sigma: {outputs[1][:1,:]}')
                    print(f'value: {outputs[2][:1,:]}')

            # Save the trained model
            agent.save('smooth_action')

        else:
            agent.restore(self.cfg.checkpoint)
            # Test loop
            for epoch in range(num_epochs):
                des_velocity = torch.rand(batch_size, 2).to(agent.device) * 2 - 1
                base_vel = torch.rand(batch_size, 3).to(agent.device) * 2 - 1
                obs = torch.cat((ranges, des_velocity, base_vel), 1).to(agent.device)

                target = torch.cat((des_velocity, base_vel[:,2:]), 1).to(agent.device)
                # Forward pass
                outputs = model({'obs': obs})
                loss = mse(outputs[0], target) + 1 / (outputs[1])**2 + torch.log(outputs[1])
                loss += mse(outputs[2], torch.zeros_like(outputs[2]))
                loss = torch.mean(loss)

                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{num_epochs}')
                    print(f'Loss: {loss.item()}')
                    print(f'Output: {outputs[0][:1,:]}')
                    print(f'Desired: {target[:1,]}')
            


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    if cfg_dict["test"]:
        cfg_dict["task"]["env"]["numEnvs"] = 1
        cfg_dict["train"]["params"]["config"]["minibatch_size"] = cfg_dict["train"]["params"]["config"]["horizon_length"]
        #cfg_dict["task"]["domain_randomization"]["randomize"] = False

    task = initialize_task(cfg_dict, env)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    #cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.seed = set_seed(-1, torch_deterministic=cfg.torch_deterministic)


    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()


if __name__ == '__main__':
    parse_hydra_configs()