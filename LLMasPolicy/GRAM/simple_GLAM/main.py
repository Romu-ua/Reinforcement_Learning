"""
train_language_agents.pyのシンプル化した実装
"""

import hydra

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    
