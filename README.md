# 🤖 Deep RL Trading

> Deep Reinforcement Learning for Algorithmic Stock Trading using LSTM and Transformer architectures

## 📋 Overview

This project implements a **Deep Direct Reinforcement Learning** framework for financial signal representation and automated trading. The system uses neural network policies (LSTM, Transformer) to learn optimal trading strategies directly from price data, maximizing cumulative profits while considering transaction costs.

## ✨ Key Features

- 🧠 **Multiple Policy Architectures**
  - LSTM-based trading policy
  - Transformer-based trading policy
  - Fuzzy layer for feature representation
- 📈 **Real-time Trading Decisions** - Make buy/sell decisions at each time point
- 💰 **Profit Maximization** - Optimize cumulative returns with transaction cost consideration
- 📊 **Financial Data Integration** - Works with real stock market data via yfinance
- 🎯 **Direct RL Framework** - Learn trading policies without explicit reward shaping

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning Framework** | PyTorch |
| **Data Source** | yfinance (Yahoo Finance API) |
| **Policy Architectures** | LSTM, Transformer |
| **Feature Engineering** | Fuzzy K-means clustering |
| **Optimization** | Direct Reinforcement Learning |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |

## 📁 Project Structure

```
deep-rl-trading/
├── MachineLearning_Test.ipynb    # Main implementation notebook
├── Amazon.csv                     # Sample stock data
└── README.md                      # Project documentation
```

## 🧠 Model Architecture

### 1. Policy Network μ_θ

The policy network makes trading decisions based on market conditions:

```
δ_t = μ_θ(f_t, δ_{t-1}) ∈ [-1, 1]
```

Where:
- `δ_t`: Trading decision at time t (-1: sell, 0: hold, +1: buy)
- `f_t`: Feature vector of current market condition
- `θ`: Neural network parameters

### 2. Feature Representation

**Fuzzy Layer** - Uses K-means clustering for feature representation:
- Groups similar market conditions
- Creates fuzzy membership functions
- Provides robust feature encoding

### 3. Policy Architectures

#### LSTM Policy
```python
- LSTM encoder (2 layers, 256 hidden dims)
- Captures temporal dependencies
- Output: Trading decision δ_t
```

#### Transformer Policy
```python
- Transformer encoder (2 layers, 2 heads)
- Positional encoding for sequence modeling
- Self-attention mechanism
- Output: Trading decision δ_t
```

### 4. Reward Function

```
R_t = δ_{t-1} * z_t - c * |δ_t - δ_{t-1}|
```

Where:
- `z_t`: Return at time t (p_t - p_{t-1})
- `c`: Transaction cost
- `δ_t`: Current trading position
- `δ_{t-1}`: Previous trading position

### 5. Optimization Objective

**Maximize cumulative reward:**
```
U_T = Σ R_t = Σ [δ_{t-1} * z_t - c * |δ_t - δ_{t-1}|]
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd deep-rl-trading
```

2. **Install dependencies**
```bash
pip install torch torchvision
pip install yfinance pandas numpy matplotlib
pip install tqdm pickle
```

3. **Download stock data**
```python
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
```

### Usage

1. **Open Jupyter Notebook**
```bash
jupyter notebook MachineLearning_Test.ipynb
```

2. **Run cells sequentially**
- Import libraries
- Load and preprocess data
- Define policy networks
- Train the trading agent
- Evaluate performance

### Training Example

```python
# Initialize policy
policy = DRT_LSTM_Policy(in_features=m, hidden_dims=256, num_layers=2)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Calculate cumulative reward
    cumulative_reward = calc_cumulative_reward(
        returns, features, policy, c=transaction_cost
    )
    
    # Maximize reward (minimize negative reward)
    loss = -cumulative_reward
    loss.backward()
    optimizer.step()
```

## 📊 Key Components

### Fuzzy Layer
- Implements fuzzy K-means clustering
- Creates fuzzy membership functions
- Provides robust feature representation

### LSTM Policy
- Processes sequential market data
- Captures long-term dependencies
- Outputs trading decisions

### Transformer Policy
- Self-attention mechanism
- Positional encoding
- Parallel processing of sequences

### Reward Calculation
- Considers actual returns
- Penalizes excessive trading (transaction costs)
- Balances profit and trading frequency

## 📈 Performance Metrics

- **Cumulative Return**: Total profit/loss over evaluation period
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## 🎯 Mathematical Framework

### Terminology
- `p_t`: Price at time t
- `z_t = p_t - p_{t-1}`: Return at time t
- `f_t`: Feature vector at time t
- `δ_t ∈ [-1, 1]`: Trading decision at time t
- `R_t`: Reward at time t
- `U_T`: Cumulative reward

### Direct RL Objective
Find optimal policy parameters θ* that maximize:
```
θ* = argmax_θ E[U_T(θ)]
```

## 🔧 Configuration

### Hyperparameters
```python
# Model
hidden_dims = 256
num_layers = 2
embed_dims = 256
nhead = 2

# Training
learning_rate = 0.001
transaction_cost = 0.001
window_size = 20
batch_size = 32
```

## 📝 License

This project is developed for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📚 References

- Deep Direct Reinforcement Learning for Financial Signal Representation and Trading (IEEE)
- LSTM Networks for Stock Market Prediction
- Attention Is All You Need (Transformer Architecture)
- Reinforcement Learning: An Introduction (Sutton & Barto)

---
⭐ If you find this project useful, please give it a star!## Overview

This project explores the application of **Deep Direct Reinforcement Learning (DRL)** for automated financial signal representation and trading. It is a research project conducted at the **Faculty of Information Technology, Industrial University of Ho Chi Minh City**.

The core objective is to develop a computational agent capable of **outperforming experienced human traders** in financial markets.  This is achieved by training a system that can:

* **Represent complex financial signals effectively.**
* **Make real-time trading decisions autonomously.**

The project introduces a novel architecture based on a **Recurrent Deep Neural Network (RDNN)** combined with **Fuzzy Logic** to address the challenges of financial trading, such as noisy data, market volatility, and the need for robust decision-making under uncertainty.

## Methodology

This project leverages a combination of advanced techniques:

* **Deep Direct Reinforcement Learning (DRL):**  A reinforcement learning approach where the agent directly learns a policy (trading actions) from raw financial data, without relying on traditional value function approximation methods for complex, continuous action spaces.
* **Recurrent Deep Neural Network (RDNN):**  A deep neural network architecture with recurrent connections. The RDNN is used to:
    * **Capture temporal dependencies** in financial time series data.
    * **Represent market conditions** by learning powerful feature representations directly from the data.
    * **Incorporate memory** of past trading actions and market states to make informed decisions.
* **Fuzzy Logic:** Integrated to address the inherent **uncertainty and noise** in financial markets. Fuzzy logic is used in the input representation layer to:
    * **Reduce data uncertainty** by mapping raw numerical data to fuzzy sets (representing concepts like "increase," "decrease," "no trend").
    * **Improve robustness** and stability of the trading agent by handling imprecise and ambiguous financial signals.
* **Task-Aware Backpropagation Through Time (Task-Aware BPTT):** A modified BPTT algorithm tailored for the RDNN architecture to address the vanishing gradient problem and improve training efficiency in deep and recurrent networks. This method propagates gradients directly from the task objective (profit maximization) to deeper layers, enhancing learning in complex sequential decision-making tasks.

## Key Features & Findings

* **Novel RDNN-Fuzzy Architecture:**  Introduces a unique deep learning architecture combining RDNNs and Fuzzy Logic for financial trading.
* **Robust Feature Learning:** The RDNN component enables automatic and effective feature extraction from complex financial time series, surpassing traditional handcrafted technical indicators.
* **Enhanced Decision Making:**  The DRL framework allows the agent to learn optimal trading policies directly from market interactions, adapting to dynamic market conditions.
* **Uncertainty Handling:** Fuzzy logic integration improves the system's ability to handle noisy financial data and make more stable and reliable trading decisions.
* **Empirical Validation:** The system was tested on real-world financial market data for futures contracts, demonstrating its potential to generate consistent profits across various market conditions and outperform other trading strategies.

## Installation 
* Required Libraries (Example - adjust based on your actual project):

Python (>= 3.7 recommended)

TensorFlow/Keras (or PyTorch for Deep Learning)

NumPy

Pandas

...

## Results
* The thesis provides detailed experimental results and performance evaluations. Key findings are visualized and summarized within the thesis document, demonstrating the effectiveness of the proposed DRNN-Fuzzy architecture.

* Refer to the figures within the thesis document (like "Cumumlative profit through time step" and "Avg300 training steps") for visual representations of the results.

**Authors**
* Đặng Quốc Toàn - 20051051

* Nguyễn Trường Chinh - 20045391

* Đỗ Thu Đông - 20043131

* Faculty of Information Technology, Industrial University of Ho Chi Minh City
* Year: 2024
