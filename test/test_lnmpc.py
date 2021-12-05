#!/usr/bin/env python3
import numpy as np
import casadi as ca
import math
import time

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped

# Odometry callback
odom = Odometry()
def odom_callback(data):
    global odom
    odom = data

def quaternion2Yaw(orientation):
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w

    yaw = math.atan2(2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1*q1 + q2*q2))
    return yaw

def shift(T, t0, x0, u, x_n, f):
    f_value = f(x0, u[0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return t, st, u_end, x_n

def desired_command_and_trajectory(t, T, x0_:np.array, N_):
    # initial state / last state
    x_ = np.zeros((N_+1, 6))
    x_[0] = x0_
    u_ = np.zeros((N_, 3))
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + T*i
        x_ref_ = 4*math.cos(2*math.pi/12*t_predict)
        y_ref_ = 4*math.sin(2*math.pi/12*t_predict)
        theta_ref_ = 2*math.pi/12*t_predict + math.pi/2
        
        dotx_ref_ = -2*math.pi/12*y_ref_
        doty_ref_ =  2*math.pi/12*x_ref_
        dotq_ref_ =  2*math.pi/12

        # vx_ref_ = dotx_ref_*math.cos(theta_ref_) + doty_ref_*math.sin(theta_ref_)
        # vy_ref_ = -dotx_ref_*math.sin(theta_ref_) + doty_ref_*math.cos(theta_ref_)
        vx_ref_ = 4*2*math.pi/12
        vy_ref_ = 0
        omega_ref_ = dotq_ref_

        x_[i+1] = np.array([x_ref_, y_ref_, theta_ref_, vx_ref_, vy_ref_, omega_ref_])
    # return pose and command
    return x_

def correct_state(states, tracjectories):
    error = tracjectories - states
    error[:,2] = error[:,2] - np.floor((error[:,2] + np.pi)/(2*np.pi))*2*np.pi
    tracjectories = states + error
    return tracjectories

# The main node
def nmpc_node():
    rospy.init_node("rl_node", anonymous=True)

    # Subscriber
    rospy.Subscriber("/odom", Odometry, odom_callback)

    # Publisher
    pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=50)
    pub_pre_path = rospy.Publisher('/predict_path', Path, queue_size=50)
    
    rate = 50
    r = rospy.Rate(rate)

    print("[INFO] Init Node...")
    while(odom.header.frame_id == ""):
        r.sleep()
        continue
    print("[INFO] NMPC Node is ready!!!")

    # Robot configuration
    d = 0.2
    M = 10; J = 2
    Bx = 0.05; By = 0.05; Bw = 0.06
    Cx = 0.5; Cy = 0.5; Cw = 0.6

    T = 1/rate              # time step
    N = 25                  # horizon length
    v_max = 3.0             # linear velocity max
    omega_max = np.pi/3.0   # angular velocity max
    u_max = 10              # force max of each direction

    opti = ca.Opti()
    # control variables, toruqe of each wheel
    opt_controls = opti.variable(N, 3)
    u1 = opt_controls[:, 0]
    u2 = opt_controls[:, 1]
    u3 = opt_controls[:, 2]
    
    # state variable: position and velocity
    opt_states = opti.variable(N+1, 6)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    vx = opt_states[:, 3]
    vy = opt_states[:, 4]
    omega = opt_states[:, 5]

    # create model
    f = lambda x_, u_: ca.vertcat(*[x_[3]*ca.cos(x_[2]) - x_[4]*ca.sin(x_[2]), 
                                    x_[3]*ca.sin(x_[2]) + x_[4]*ca.cos(x_[2]),
                                    x_[5],
                                    (0*u_[0] +  ca.sqrt(3)/2*u_[1] - ca.sqrt(3)/2*u_[2] - Bx*x_[3])/M,
                                    (-1*u_[0] + 1/2*u_[1] + 1/2*u_[2] - By*x_[4])/M,
                                    (d*u_[0] +  d*u_[1] + d*u_[2] - Bw*x_[5])/J])
    f_np = lambda x_, u_: np.array([x_[3]*ca.cos(x_[2]) - x_[4]*ca.sin(x_[2]), 
                                    x_[3]*ca.sin(x_[2]) + x_[4]*ca.cos(x_[2]),
                                    x_[5],
                                    (0*u_[0] +  ca.sqrt(3)/2*u_[1] - ca.sqrt(3)/2*u_[2] - Bx*x_[3])/M,
                                    (-1*u_[0] + 1/2*u_[1] + 1/2*u_[2] - By*x_[4])/M,
                                    (d*u_[0] +  d*u_[1] + d*u_[2] - Bw*x_[5])/J])
    # parameters, these parameters are the reference trajectories of the pose and inputs
    opt_u_ref = opti.parameter(N, 3)
    opt_x_ref = opti.parameter(N+1, 6)

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x_ref[0, :])
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # weight matrix
    Q = np.diag([50.0, 50.0, 10.0, 5.0, 5.0, 2.0])
    R = np.diag([1.0, 1.0, 1.0])

    # cost function
    obj = 0
    for i in range(N):
        state_error_ = opt_states[i, :] - opt_x_ref[i+1, :]
        # control_error_ = opt_controls[i, :] - opt_u_ref[i, :]
        obj = obj + ca.mtimes([state_error_, Q, state_error_.T]) \
                    + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
    opti.minimize(obj)

    # Lyapunov NMPC - SMC constraints
    lamda = np.diag([2,2,2])
    H = ca.vertcat(ca.horzcat(ca.cos(opt_states[0,2]), -ca.sin(opt_states[0,2]),   0),
                   ca.horzcat(ca.sin(opt_states[0,2]), ca.cos(opt_states[0,2]),    0),
                   ca.horzcat(0,                       0,                          1),)
    H_inv = ca.transpose(H)
    H_dot = ca.vertcat(ca.horzcat(-ca.sin(opt_states[0,2]), -ca.cos(opt_states[0,2]),   0),
                       ca.horzcat(ca.cos(opt_states[0,2]), -ca.sin(opt_states[0,2]),    0),
                       ca.horzcat(0,                       0,                           1),)
    
    M_mat = np.diag([1/M,1/M,1/J])
    B_mat = np.diag([Bx, By, Bw])
    C_mat = np.diag([Cx, Cy, Cw])
    G_mat = ca.DM([ [0, np.sqrt(3)/2,   -np.sqrt(3)/2],
                    [-1,        1/2,    1/2],
                    [d,         d,      d]])

    z1 = ca.transpose(state_error_[0,:3]); z2 = ca.transpose(state_error_[0,3:])
    S = H@z2 + lamda@z1
    c2 = 2; c3 = 2
    Ve = ca.transpose(S)@(c3@S) + ca.transpose(S)@H@(M_mat@(G_mat@ca.transpose(opt_controls[0,:])-B_mat@ca.transpose(opt_states[0,3:])) + (lamda + H_inv@H_dot)@z2)

    #### boundrary and control conditions
    # boundary and control conditions
    opti.subject_to(opti.bounded(-math.inf, x, math.inf))
    opti.subject_to(opti.bounded(-math.inf, y, math.inf))
    opti.subject_to(opti.bounded(-math.inf, theta, math.inf))
    opti.subject_to(opti.bounded(-v_max, vx, v_max))
    opti.subject_to(opti.bounded(-v_max, vy, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))
    # opti.subject_to(opti.bounded(-u_max, u1, u_max))
    # opti.subject_to(opti.bounded(-u_max, u2, u_max))
    # opti.subject_to(opti.bounded(-u_max, u3, u_max))
    # opti.subject_to(opti.bounded(-math.inf, Ve, 0))

    opts_setting = {'ipopt.max_iter':2000,
                    'ipopt.print_level':0,
                    'print_time':0,
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6}
    opti.solver('ipopt', opts_setting)

    t0 = 0
    u0 = np.zeros((N, 3))
    next_controls = np.zeros((N, 3))
    next_states = np.zeros((N+1, 6))

    while not rospy.is_shutdown():
        ## get the current state
        current_state = np.array([odom.pose.pose.position.x,
                                odom.pose.pose.position.y,
                                quaternion2Yaw(odom.pose.pose.orientation),
                                odom.twist.twist.linear.x,
                                odom.twist.twist.linear.y,
                                odom.twist.twist.angular.z])
        
        ## estimate the new desired trajectories and controls
        next_trajectories = desired_command_and_trajectory(t0, T, current_state, N)
        next_trajectories = correct_state(next_states, next_trajectories)

        # print(current_state)
        # print(next_states)

        ## set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x_ref, next_trajectories)
        # opti.set_value(opt_u_ref, next_controls)

        ## provide the initial guess of the optimization targets
        opti.set_initial(opt_controls, u0.reshape(N, 3))# (N, 3)
        opti.set_initial(opt_states, next_states) # (N+1, 6)

        ## solve the problem once again
        sol = opti.solve()

        ## obtain the control input
        u_res = sol.value(opt_controls)
        x_m = sol.value(opt_states)
        # v = predict_velocity(x_m[0], u_res[0])

        t0, current_state, u0, next_states = shift(T, t0, current_state, u_res, x_m, f_np)
        print(u0[0,:])
        # Publish velocity
        vel = Twist()
        vel.linear.x = current_state[3]
        vel.linear.y = current_state[4]
        vel.angular.z = current_state[5]
        pub_vel.publish(vel)

        # Publish predict path
        path = Path()
        path.header = odom.header
        for s in next_states:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = s[0]
            pose.pose.position.y = s[1]
            path.poses.append(pose)
        pub_pre_path.publish(path)

        r.sleep()

if __name__ == '__main__':    
    try:
        nmpc_node()
    except rospy.ROSInterruptException:
        pass