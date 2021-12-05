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

# Reference path callback
path = Path()
def path_callback(data):
    global path
    path = data

def quaternion2Yaw(orientation):
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w

    yaw = math.atan2(2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1*q1 + q2*q2))
    return yaw

def shift(u, x_n):
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return u_end, x_n

def desired_command_and_trajectory(state, reference, N, count, dt):
    # initial state / last state
    x_ = np.zeros((N+1, 6))
    x_[0] = state
    
    length = len(reference.poses)
    # states for the next N trajectories
    for i in range(N):
        idx = i + count
        if idx >= length:
            idx = length - 1
        x_ref = reference.poses[idx].pose.position.x
        y_ref = reference.poses[idx].pose.position.y
        theta_ref = quaternion2Yaw(reference.poses[idx].pose.orientation)

        pidx = idx-1
        if pidx >= length:
            pidx = length - 1
        elif pidx < 0:
            pidx = 0
        px_ref = reference.poses[pidx].pose.position.x
        py_ref = reference.poses[pidx].pose.position.y
        ptheta_ref = quaternion2Yaw(reference.poses[pidx].pose.orientation)

        theta_err = theta_ref - ptheta_ref
        theta_err = theta_err - np.floor((theta_err + np.pi)/(2*np.pi))*2*np.pi
        omega_ref = theta_err/dt

        dotx_ref = (x_ref - px_ref)/dt
        doty_ref = (y_ref - py_ref)/dt
        vx_ref = dotx_ref*np.cos(theta_ref) + doty_ref*np.sin(theta_ref)
        vy_ref = -dotx_ref*np.sin(theta_err) + doty_ref*np.cos(theta_ref)
        x_[i+1] = np.array([x_ref, y_ref, theta_ref, vx_ref, vy_ref, omega_ref])

    return x_

def correct_state(states, tracjectories):
    error = tracjectories - states
    error[:,2] = error[:,2] - np.floor((error[:,2] + np.pi)/(2*np.pi))*2*np.pi
    tracjectories = states + error
    return tracjectories

# The main node
def nmpc_node():
    rospy.init_node("nmpc_node", anonymous=True)

    # Subscriber
    rospy.Subscriber("/odom", Odometry, odom_callback)
    rospy.Subscriber("/path", Path, path_callback)
    
    # Publisher
    pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=50)
    pub_pre_path = rospy.Publisher('/predict_path', Path, queue_size=50)
    
    rate = 50
    r = rospy.Rate(rate)

    print("[INFO] Init Node...")
    while(odom.header.frame_id == "" or path.header.frame_id == ""):
        r.sleep()
        continue
    print("[INFO] NMPC Node is ready!!!")

    # Robot configuration
    d = 0.2
    M = 10; J = 2
    Bx = 0.05; By = 0.05; Bw = 0.06
    Cx = 0.5; Cy = 0.5; Cw = 0.6

    T = 1/rate              # time step
    N = 30                  # horizon length
    v_max = 3.0             # linear velocity max
    omega_max = np.pi/3.0   # angular velocity max
    u_max = 20              # force max of each direction

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
    # opt_u_ref = opti.parameter(N, 3)
    opt_x_ref = opti.parameter(N+1, 6)

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x_ref[0, :])
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # weight matrix
    Q = np.diag([30.0, 30.0, 1.0, 5.0, 5.0, 0.1])
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
    c2 = 2; c3 = 0.5
    Ve = ca.transpose(S)@(c3@S) + ca.transpose(S)@H@(M_mat@(G_mat@ca.transpose(opt_controls[0,:])-B_mat@ca.transpose(opt_states[0,3:])) + (lamda + H_inv@H_dot)@z2)

    #### boundrary and control conditions
    # boundary and control conditions
    opti.subject_to(opti.bounded(-math.inf, x, math.inf))
    opti.subject_to(opti.bounded(-math.inf, y, math.inf))
    opti.subject_to(opti.bounded(-math.inf, theta, math.inf))
    opti.subject_to(opti.bounded(-v_max, vx, v_max))
    opti.subject_to(opti.bounded(-v_max, vy, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))
    opti.subject_to(opti.bounded(-u_max, u1, u_max))
    opti.subject_to(opti.bounded(-u_max, u2, u_max))
    opti.subject_to(opti.bounded(-u_max, u3, u_max))
    opti.subject_to(opti.bounded(-math.inf, Ve, 0))

    opts_setting = {'ipopt.max_iter':2000,
                    'ipopt.print_level':0,
                    'print_time':0,
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6}
    opti.solver('ipopt', opts_setting)

    u0 = np.zeros((N, 3))
    next_states = np.zeros((N+1, 6))

    count = 0
    while not rospy.is_shutdown():
        ## get the current state
        current_state = np.array([odom.pose.pose.position.x,
                                odom.pose.pose.position.y,
                                quaternion2Yaw(odom.pose.pose.orientation),
                                odom.twist.twist.linear.x,
                                odom.twist.twist.linear.y,
                                odom.twist.twist.angular.z])
        
        ## estimate the new desired trajectories and controls
        next_trajectories = desired_command_and_trajectory(current_state, path, N, count, T)
        next_states[0,:] = current_state
        next_trajectories = correct_state(next_states, next_trajectories)

        # print(current_state)
        # print(next_states)

        ## set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x_ref, next_trajectories)
        # opti.set_value(opt_u_ref, next_controls)

        ## provide the initial guess of the optimization targets
        opti.set_initial(opt_controls, u0.reshape(N, 3))# (N, 3)
        opti.set_initial(opt_states, next_states) # (N+1, 3)

        ## solve the problem once again
        sol = opti.solve()

        ## obtain the control input
        u_res = sol.value(opt_controls)
        x_m = sol.value(opt_states)

        u0, next_states = shift(u_res, x_m)
        print(u0[0,:])
        # Publish velocity
        vel = Twist()
        vel.linear.x = next_states[0,3]
        vel.linear.y = next_states[0,4]
        vel.angular.z = next_states[0,5]
        pub_vel.publish(vel)

        # Publish predict path
        pre_path = Path()
        pre_path.header = odom.header
        for s in next_states:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = s[0]
            pose.pose.position.y = s[1]
            pre_path.poses.append(pose)
        pub_pre_path.publish(pre_path)
        count += 1
        r.sleep()

if __name__ == '__main__':    
    try:
        nmpc_node()
    except rospy.ROSInterruptException:
        pass