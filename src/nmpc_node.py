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

def desired_command_and_trajectory(state, reference, N, count):
    # initial state / last state
    x_ = np.zeros((N+1, 3))
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
        x_[i+1] = np.array([x_ref, y_ref, theta_ref])

    return x_

def correct_state(states, tracjectories):
    error = tracjectories - states
    error[:,2] = error[:,2] - np.floor((error[:,2] + np.pi)/(2*np.pi))*2*np.pi
    tracjectories = states + error
    return tracjectories

def update_next_states(current_state, next_states):
    error = next_states[0] - current_state
    for i in range(len(next_states)):
        next_states[i] = next_states[i] - error
    return next_states
        

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
    T = 1/rate
    N = 30                  # horizon length
    v_max = 3.0             # linear velocity max
    omega_max = np.pi/3.0   # angular velocity max

    opti = ca.Opti()
    # control variables, liear velocity and angular velocity
    opt_controls = opti.variable(N, 3)
    vx = opt_controls[:, 0]
    vy = opt_controls[:, 1]
    omega = opt_controls[:, 2]
    
    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    # parameters, these parameters are the reference trajectories of the pose and inputs
    opt_x_ref = opti.parameter(N+1, 3)

    # create model
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]) - u_[1]*ca.sin(x_[2]),
                                    u_[0]*ca.sin(x_[2]) + u_[1]*ca.cos(x_[2]),
                                    u_[2]])
    f_np = lambda x_, u_: np.array([u_[0]*ca.cos(x_[2]) - u_[1]*ca.sin(x_[2]),
                                    u_[0]*ca.sin(x_[2]) + u_[1]*ca.cos(x_[2]),
                                    u_[2]])

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x_ref[0, :])
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # weight matrix
    Q = np.diag([30.0, 30.0, 1.0])
    R = np.diag([1.0, 1.0, 0.1])

    # cost function
    obj = 0
    for i in range(N):
        state_error_ = opt_states[i, :] - opt_x_ref[i+1, :]
        obj = obj + ca.mtimes([state_error_, Q, state_error_.T]) +\
                    ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
    opti.minimize(obj)

    # boundrary and control conditions
    opti.subject_to(opti.bounded(-math.inf, x, math.inf))
    opti.subject_to(opti.bounded(-math.inf, y, math.inf))
    opti.subject_to(opti.bounded(-math.inf, theta, math.inf))
    opti.subject_to(opti.bounded(-v_max, vx, v_max))
    opti.subject_to(opti.bounded(-v_max, vy, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

    opts_setting = {'ipopt.max_iter':2000,
                    'ipopt.print_level':0,
                    'print_time':0,
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6}
    opti.solver('ipopt', opts_setting)

    u0 = np.zeros((N, 3))
    next_states = np.zeros((N+1, 3))  #np.tile(current_state, N+1).reshape(N+1, -1)
    
    count = 0
    while not rospy.is_shutdown():

        ## get the current state
        current_state = np.array([odom.pose.pose.position.x,
                                odom.pose.pose.position.y,
                                quaternion2Yaw(odom.pose.pose.orientation)])
        next_states = update_next_states(current_state, next_states)

        ## estimate the new desired trajectories and controls
        next_trajectories = desired_command_and_trajectory(current_state, path, N, count)
        next_trajectories = correct_state(next_states, next_trajectories)

        ## set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x_ref, next_trajectories)

        ## provide the initial guess of the optimization targets
        opti.set_initial(opt_controls, u0.reshape(N, 3))# (N, 3)
        opti.set_initial(opt_states, next_states) # (N+1, 3)

        ## solve the problem once again
        sol = opti.solve()

        ## obtain the control input
        u_res = sol.value(opt_controls)
        x_m = sol.value(opt_states)

        u0, next_states = shift(u_res, x_m)
    
        # Publish velocity
        vel = Twist()
        vel.linear.x = u_res[0,0]
        vel.linear.y = u_res[0,1]
        vel.angular.z = u_res[0,2]
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