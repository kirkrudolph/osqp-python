import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt


# Discrete time model of a quadcopter
# 
# lin. pos. 3
# ang. pos. 3
# lin. vel. 3
# ang. vel. 3
Ad = sparse.csc_matrix([
  [1.,      0.,     0., 0.1,     0.,     0.,  0.,     0.,     0.    ], # 1    p_x[k+1] = p_x + .1*v_x
  [0.,      1.,     0., 0.,      0.1,    0.,  0.,     0.,     0.    ], # 2    p_y[k+1] = p_y + .1*v_y
  [0.,      0.,     1., 0.,      0.,     0.1, 0.,     0.,     0.    ], # 3    p_z[k+1] = p_z + .1*v_z
  [0.,      0.,     0., 1.,      0.,     0.,  0.,     0.,     0.    ], # 7    v_x[k+1] = v_x
  [0.,      0.,     0., 0.,      1.,     0.,  0.,     0.,     0.    ], # 8    v_y[k+1] = v_y
  [0.,      0.,     0., 0.,      0.,     1.,  0.,     0.,     0.    ], # 9    v_z[k+1] = v_z
  [0.9734,  0.,     0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ], # 10   w_x[k+1] =  .9734*p_x + 0.0488*v_x .9846*w_x
  [0.,     -0.9734, 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ], # 11   w_x[k+1] = -.9734*p_x - 0.0488*v_x .9846*w_x
  [0.,      0.,     0., 0.,      0.,     0.,  0.,     0.,     0.9846]  # 12   w_z[k+1] = 0.9846*w_z
])

Bd = sparse.csc_matrix([
  [0.,      -0.0726,  0.,    ],   # 1    -0.0726*u2 + 0.0726*u4
  [-0.0726,  0.,      0.0726,],   # 2    -0.0726*u1 + 0.0726*u3
  [-0.0152,  0.0152, -0.0152,],   # 3    -0.0152*u1 + 0.0152*u2 - 0.0152*u3 + 0.0152*u4
  [0,       -1.4512,  0.,    ],   # 7    -1.4512*u1 + 1.4512*u2
  [-1.4512,  0.,      1.4512,],   # 8    -1.4512*u1 + 1.4512*u3
  [-0.3049,  0.3049, -0.3049,],   # 9    -0.3049*u1 + 0.3049*u2 - 0.3049*u3 + 0.3049*u4
  [-0.,     -0.0236,  0.,    ],   # 10   -0.0236*u1 + 0.0236*u4
  [0.0236,   0.,     -0.0236,],   # 11    0.0236*u1 - 0.0236*u3
  [0.2107,   0.2107,  0.2107,]])  # 12    0.2107*u1 + 0.2107*u2 + 0.2107*u3 + 0.2107*u4

[nx, nu] = Bd.shape
print(nx)
print(nu)
# Constraints
u0 = 10.5916
umin = np.array([9.6, 9.6, 9.6]) - u0
umax = np.array([13., 13., 13.]) - u0
xmin = np.array([ -np.pi/6,
                  -np.pi/6,
                  -np.inf,
                  -np.inf,
                  -np.inf,
                  -np.inf,
                  -np.inf,
                  -np.inf,
                  -np.inf])
xmax = np.array([ 
                  np.pi/6, 
                  np.pi/6, 
                  np.inf, 
                  np.inf, 
                  np.inf, 
                  np.inf, 
                  np.inf, 
                  np.inf, 
                  np.inf])

# Objective function
Q = sparse.diags([0., 0., 10., 0., 0., 0., 5., 5., 5.])
QN = Q
R = 0.1*sparse.eye(3)

# Initial and reference states
x0 = np.zeros(9)
xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.])

# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
nsim = 15
x = x0                      # initial state vector
u_out = np.zeros((nu,1))     # initial control vector
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad.dot(x0) + Bd.dot(ctrl)

    # Store States for plotting
    x = np.append(x,x0)

    # Store Control for plotting
    u_out = np.append(u_out,ctrl)

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)
# 
# num_states = np.size(x0)    # 12
# num_actuators = np.size(np.zeros((nu,1)))
# num_steps = nsim+1          # 16
# 
# # Reshape output data
# x = np.reshape(x,[num_steps,num_states])
# u_out = np.reshape(u_out,[num_steps,num_actuators])
# print(x)
# print(np.shape(x))
# print(np.shape(u_out))
# 
# # Plot all states over all time steps
# plt.figure(1)
# plt.suptitle('State Vector')
# # Plot Linear positions
# plt.subplot(2,2,1)
# plt.plot(x[0:num_steps,0:3])
# plt.legend(['$p_x$', '$p_y$', '$p_z$'])
# plt.xlabel('Time')
# plt.ylabel('Linear Positions [m]')
# # Plot Linear Velocities
# plt.subplot(2,2,3)
# plt.plot(x[0:num_steps,3:6])
# plt.legend(['$v_x$','$v_y$','$v_z$'])
# plt.xlabel('Time')
# plt.ylabel('Linear Velocities [m/s]')
# # Plot Angular Velocities
# plt.subplot(2,2,4)
# plt.plot(x[0:num_steps,6:9])
# plt.legend(['$\Phi_x$','$\Phi_y$','$\Phi_z$'])
# plt.xlabel('Time')
# plt.ylabel('Angular Velocities [rad/s]')
# # Disaply Plot
# plt.show()
# 
# # Plot all actuation over all time steps
# plt.figure(2)
# plt.title('Actuator Vector')
# plt.plot(u_out)
# plt.legend(['$u_1$', '$u_2$', '$u_3$','$u_4$'])
# plt.xlabel('Time')
# plt.ylabel('Actuator Effort')
# # Disaply Plot
# plt.show()
# 
# Generate c-code for embedded
# prob.codegen('osqp_gen', parameters='matrices')