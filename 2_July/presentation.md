---
marp: true
theme: default
size: 16:9
paginate: true
backgroundColor: var(--bg-primary)
color: var(--text-primary)
math: katex
style: |
  @import url('../style.css');
header: 'Advanced Machine Learning Research | Weekly Progress'
footer: 'Aniket Dixit | Computer Science PhD | Coventry University | 2025'
---

<!-- _class: title-slide -->

# Dollhouse Experiment: RL-Based HVAC Control

<div class="author-info">

**Aniket Dixit**  
PhD Candidate, Computer Science  
Coventry University  
Weekly Progress Meeting

**Supervisor:** Prof. James Brusey  
Date: 2 July 2025

</div>

---

# Today's Agenda

- **Reward Function Design**: Binary comfort + energy efficiency approach
- **Training Methodology**: Fixed vs. variable episode length strategies
- **Experimental Setup**: PPO implementation with different weight configurations
- **Results**: Baseline vs. PPO comparison across comfort and energy metrics
- **Next Steps**: Future directions and improvements

<div class="key-insight">
<strong>Focus:</strong> Developing an RL agent that maintains comfort while optimizing energy efficiency in a simulated dollhouse environment
</div>

---

<!-- _class: section-divider -->

# Reward Function Design

---

# Reward Function Architecture

## Base Reward (Binary + Energy Bonus)

<div class="math-block">

$$R_{base}(s,a) = \begin{cases} 
1.0 + 0.5 \times (1 - \frac{\text{lights\_on}}{2}) & \text{if both floors comfortable} \\
0.0 & \text{otherwise}
\end{cases}$$

</div>

- **Comfort Check**: Both ground and top floor temperatures within setpoint range
- **Energy Bonus**: 0.0 to 0.5 based on light usage (0 lights = max bonus, 2 lights = no bonus)
- **Per-step Range**: 0.0 to 1.5

---

## Potential-Based Reward Shaping

<div class="two-column">

<div class="column">

**Comfort Potential:**
$$\Phi_{comfort}(T) = \begin{cases} 
1.0 & \text{if } T_{heat} \leq T \leq T_{cool} \\
e^{-\alpha \cdot d} & \text{otherwise}
\end{cases}$$

**Energy Potential:**
$$\Phi_{energy}(a) = \begin{cases} 
1.0 - 0.3 \times \frac{\text{lights}}{2} & \text{if comfortable} \\
0.2 - 0.1 \times \frac{\text{lights}}{2} & \text{otherwise}
\end{cases}$$

</div>

<div class="column">

**Total Shaped Reward:**
$$R'(s,a,s') = R_{base} + \lambda \times F(s,a,s')$$

Where: $F(s,a,s') = \gamma \times \Phi(s') - \Phi(s)$

**Parameters:**
- $\lambda = 0.4$ (shaping weight)
- $\gamma = 0.99$ (discount factor)
- $\alpha = 0.2$ (comfort decay rate)

</div>

</div>

---

<!-- _class: section-divider -->

# Training Methodology

---

<div class="one-column">

<div class="column">

## Approach 1: Variable Episode Length

```python
min_steps = 240    # 2 hours
max_steps = 20160  # 7 days

if (self.current_step >= min_steps and 
    np.random.rand() < 0.01) or \
   self.current_step >= max_steps:
    truncated = True
else:
    truncated = False
```

**Disadvantages:**
- Difficult reward convergence interpretation
- Inconsistent training signals
- Variable return magnitudes

</div>

---

<div class="column">

## Approach 2: Fixed Length + Random Start ✅

```python
episode_length = 2880  # Fixed
start_time = np.random.randint(0, 
    total_simulation_days * 24 * 60)

# Initialize environment at random time
env.reset(start_timestamp=start_time)
```

**Advantages:**
- Consistent episode structure
- Better reward convergence
- Easier hyperparameter tuning
- **Selected approach**

</div>

</div>

---
<!-- _class: section-divider -->

# Experimental Results

---
## Weight Configurations Tested

| Configuration | Energy Weight | Comfort Weight | Objective |
|---------------|---------------|----------------|-----------|
| **Baseline** | - | - | Rule-based heuristic |
| **PPO-Comfort** | 0.0 | 1.0 | Pure comfort optimization |
| **PPO-Energy** | 1.0 | 0.0 | Pure energy optimization |
| **PPO-Balanced** | 1.0 | 1.0 | Comfort + energy balance |

## Evaluation Metrics

- **Comfort Level**: Percentage of time both floors within setpoint range
- **Light Usage Hours**: Total hours lights were on (energy proxy)
- **Episode Return**: Cumulative reward over 2880 steps

---

# Results: Comfort vs Energy Trade-offs

<div class="two-column">

<div class="column">

## Comfort Performance
| Configuration | Ground Floor | Top Floor |
|---------------|--------------|-----------|
| **Baseline** | **97.0%** | **95.0%** |
| **PPO-Comfort** | 94.7% | 94.1% |
| **PPO-Balanced** | 94.7% | 94.6% |

**Observation**: Baseline achieves highest comfort but with poor energy efficiency

</div>

<div class="column">

## Energy Performance
| Configuration | Light Hours | Energy Savings |
|---------------|-------------|----------------|
| **Baseline** | 9.95 hrs | - |
| **PPO-Comfort** | 6.09 hrs | **+38.8%** |
| **PPO-Balanced** | 6.15 hrs | **+38.2%** |

**Observation**: Both PPO agents achieve similar ~38% energy savings

</div>

</div>

---
<!-- _class: title-slide -->

# Thank You

## Questions & Discussion

<div class="author-info">

**Progress Summary:**  
✅ Implemented balanced reward function  
✅ Established fixed-length training methodology  
✅ Achieved 38% energy savings with minimal comfort loss (~2%)  
❓ Investigating why baseline outperforms on comfort metrics  

</div>



