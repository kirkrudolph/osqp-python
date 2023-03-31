import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt


# 2 state system
Ad = sparse.csc_matrix([
  [0.998366193285897, 0.001775489291212],
  [0.000037801523782, 0.999406363929909]
])

# 3 actuator system
Bd = sparse.csc_matrix([
  [0.000427434645214, -0.001856823229817, 0.001707194327017],
  [0.000116008676630,  0.000893590824269, 0.000463344178685]
])

[nx, nu] = Bd.shape

# Constraints

umin = np.array([   0, -500, -500])
umax = np.array([1000,  500,  500])
xmin = np.array([   900,
                  -4000])*np.pi/30
xmax = np.array([  2000,
                   4000])*np.pi/30

# Objective function
Q = sparse.diags([1000., 1000.])
QN = Q
R = sparse.diags([1., 0.1, 4.])

# Initial and reference states
x0 = np.array([900.,   400.])*np.pi/30
xr = np.array([1500., 1000.])*np.pi/30

# Prediction horizon
N = 100

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
nsim = 400 # 2s
x = x0                      # initial state vector
u_out = np.zeros((3,1))     # initial control vector
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

num_states = np.size(x0)
num_actuators = np.size(np.zeros((3,1)))
num_steps = nsim + 1          

# Reshape output data
x = np.reshape(x,[num_steps,num_states])
u_out = np.reshape(u_out,[num_steps,num_actuators])
print(x)
print(np.shape(x))
print(np.shape(u_out))

# Plot all states over all time steps
plt.figure(1)
plt.suptitle('Closed Loop Simulation')
# Plot Linear positions
plt.subplot(2,1,1)
plt.plot(x[0:num_steps,0:2]*30./np.pi)
plt.axhline(xr[0]*30./np.pi,color='k', linestyle='--')
plt.axhline(xr[1]*30./np.pi,color='k', linestyle='--')
plt.legend(['$w_1[k]$', '$w_2[k]$'])
plt.ylabel('Angular Velocity [RPM]')
# Plot all actuation over all time steps
plt.subplot(2,1,2)
plt.plot(u_out)
plt.legend(['$u_1[k]$', '$u_2[k]$', '$u_3[k]$'])
plt.xlabel('Samples')
plt.ylabel('Torque [N*m]')
# Disaply Plot
plt.show()

# Generate c-code for embedded
# prob.codegen('osqp_gen', parameters='matrices')