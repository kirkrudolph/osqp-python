# MPC Quadcopter Simulation

- Default code from [OSQP python demo](https://osqp.org/docs/examples/mpc.html#python).
- Additional code to plot the simulated states and actuation.
- Additional code to generate c embedded code into the `osqp_gen` folder (disabled by default).

## Install OSQP

```
pip3 install osqp
```

## Run Simulation

```
python3 demo.py
```

## Generate Embedded MPC Code
1. Uncomment last line.
2. Run script
3. See [kirkrudolph/esp-osqp](https://github.com/kirkrudolph/esp-osqp-demo)

## Simulation Results #1

|![state-vector](images/State_Vector.png)|
|:--:|
| Figure 1: State Vector |
|![actuator-vector](images/Actuator_Vector.png)|
| Figure 2: Actuator Vector |

## Simulation Results #2 (Heavily Actuator Saturated)

|![state-vector](images/State_Vector_2.png)|
|:--:|
| Figure 1: State Vector |
|![actuator-vector](images/Actuator_Vector_2.png)|
| Figure 2: Actuator Vector: |