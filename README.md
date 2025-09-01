# Deep Reinforcement Learning for World3 Model Control

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A sophisticated implementation of Deep Q-Networks (DQN) for autonomous control of the World3 system dynamics model, demonstrating the application of reinforcement learning to complex environmental and socio-economic modeling challenges.

## Overview

This project combines the seminal **World3 global sustainability model** with cutting-edge **Deep Reinforcement Learning** techniques to autonomously discover optimal control policies for managing global resources, population, and environmental health. The system learns to manipulate key policy levers within the World3 model to achieve specific sustainability objectives, such as stabilizing life expectancy and managing resource depletion.

### Key Achievements
- **Custom World3 Implementation**: Developed a modular Python implementation of the World3 system dynamics model with granular control capabilities
- **Advanced DQN Architecture**: Engineered a Deep Q-Network agent with experience replay, target networks, and dynamic state normalization
- **Intelligent Control System**: Designed reward functions and state representations to guide the agent toward sustainable global development scenarios
- **Reproducible Research**: Implemented comprehensive seed management and modular architecture for scientific reproducibility

## Architecture

### System Components

**World3 Environment** (`pyworld3/`)
- Modular implementation of the classic World3 model across 5 sectors:
  - Population dynamics
  - Industrial capital
  - Agriculture and food production  
  - Pollution and environmental degradation
  - Non-renewable resource consumption
- Support for continuous simulation with state persistence
- Granular control over 18 policy parameters

**DQN Agent** (`dqn.py`)
- Multi-layer neural network for Q-function approximation
- Experience replay buffer for stable learning
- Target network with periodic updates
- Epsilon-greedy exploration strategy
- TensorFlow/Keras implementation with GPU support

**Training Framework** (`deep_q_learning_training.py`)
- **State Space**: 6-dimensional continuous vector (life expectancy, population, industrial output per capita, agricultural investment, pollution generation rate, natural resource fraction remaining)
- **Action Space**: Discrete combinations of scaling factors applied to World3 control signals
- **Reward Engineering**: Custom reward function optimizing for life expectancy stabilization and sustainability metrics

## Quick Start

### Prerequisites
```bash
pip install tensorflow numpy matplotlib collections
```

### Basic Usage

1. **Run Standard World3 Simulation**:
```bash
python example_world3_standard.py
```

2. **Train DQN Agent**:
```bash
python deep_q_learning_training.py
```

3. **Evaluate Trained Policy**:
```bash
python deep_q_run.py
```

4. **Compare with Tabular Q-Learning**:
```bash
python Q_learning.py
```

### Example: Custom Training Configuration
```python
from dqn import DQNAgent
from pyworld3.world3 import World3

# Initialize World3 environment
world3 = World3(year_min=1900, year_max=2100, dt=1)
world3.set_world3_control()
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()

# Create DQN agent
state_size = 6
action_size = 64  # Combinations of control actions
agent = DQNAgent(state_size, action_size)

# Training loop with custom reward function
for episode in range(episodes):
    state = get_normalized_state(world3)
    action = agent.act(state)
    next_state, reward, done = step_world3(world3, action)
    agent.remember(state, action, reward, next_state, done)
    agent.replay(batch_size)
```

## Technical Specifications

### Neural Network Architecture
- **Input Layer**: 6 neurons (normalized World3 state variables)
- **Hidden Layers**: 2 × 128 neurons with ReLU activation
- **Output Layer**: 64 neurons (Q-values for each action)
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error

### State Representation
| Variable | Description | Normalization |
|----------|-------------|---------------|
| `le` | Life Expectancy | Dynamic min-max scaling |
| `pop` | Total Population | Dynamic min-max scaling |
| `iopc` | Industrial Output Per Capita | Dynamic min-max scaling |
| `ai` | Agricultural Investment | Dynamic min-max scaling |
| `ppgr` | Pollution Generation Rate | Dynamic min-max scaling |
| `nrfr` | Natural Resource Fraction Remaining | Dynamic min-max scaling |

### Action Space
- **Control Signals**: `lmhs` (Health Services), `nruf` (Resource Usage)
- **Scaling Factors**: [0.5, 1.0, 2.0] per signal
- **Total Actions**: 3² = 9 discrete action combinations

## Results & Performance

The DQN agent demonstrates successful learning of control policies that:
- Stabilize life expectancy around target values (e.g., 50-70 years)
- Manage resource depletion more effectively than uncontrolled scenarios
- Discover non-intuitive policy combinations for sustainable development
- Show convergence in training loss and episode rewards over iterations

*Detailed results and performance metrics are documented in the accompanying Bachelor's Thesis.*

## Research Applications

This project demonstrates practical applications of:
- **Reinforcement Learning in Complex Systems**: Applying DQN to multi-variable, non-linear system dynamics
- **Policy Discovery**: Automated discovery of effective governance strategies for global challenges
- **AI for Sustainability**: Using machine learning to address environmental and socio-economic modeling
- **Scientific Computing**: Bridging theoretical models with practical AI implementations

## Development

### Project Structure
```
World3-DQN/
├── pyworld3/              # World3 model implementation
│   ├── world3.py         # Main World3 class
│   ├── population.py     # Population dynamics
│   ├── capital.py        # Industrial capital
│   ├── agriculture.py    # Food production
│   ├── pollution.py      # Environmental impact
│   └── resource.py       # Resource management
├── dqn.py                # DQN agent implementation
├── deep_q_learning_training.py  # Main training script
├── state_reward.py       # Reward function and state normalization
├── Q_learning.py         # Tabular Q-learning comparison
└── results.py            # Analysis and visualization tools
```

### Key Features
- **Modular Architecture**: Clean separation between World3 environment and RL agent
- **Reproducible Research**: Comprehensive seed management for consistent results
- **Extensible Design**: Easy integration of new control signals and reward functions
- **Performance Monitoring**: Built-in logging and visualization tools

## Academic Context

This work was completed as a Bachelor's Thesis, exploring the intersection of:
- **System Dynamics**: Classical World3 model from "Limits to Growth"
- **Deep Reinforcement Learning**: Modern DQN algorithms for policy optimization  
- **Sustainability Science**: Application of AI to global environmental challenges
- **Scientific Computing**: Robust implementation of complex mathematical models

The project contributes to understanding how AI can be applied to discover effective policies for managing complex, interconnected global systems.

## Future Enhancements

- **Multi-Agent Systems**: Explore cooperative/competitive scenarios with multiple DQN agents
- **Advanced RL Algorithms**: Implementation of A3C, PPO, or SAC for improved performance
- **Extended Action Spaces**: Include more World3 control parameters for richer policy discovery
- **Uncertainty Quantification**: Incorporate model uncertainty and robustness analysis
- **Real-World Validation**: Compare learned policies with historical data and expert recommendations

## Citation

If you use this work in your research, please cite:
```
@thesis{world3_dqn_2024,
  title={Deep Reinforcement Learning for World3 Model Control},
  author={[Your Name]},
  school={KTH Royal Institute of Technology},
  year={2024},
  type={Bachelor's Thesis}
}
```

## Contributing

This project serves as a research demonstration and portfolio piece. For questions or collaboration opportunities, please reach out via [your contact information].

---

*This project demonstrates the application of modern AI techniques to classical sustainability modeling, showcasing skills in deep learning, system dynamics, and scientific computing relevant to AI/ML engineering roles.*