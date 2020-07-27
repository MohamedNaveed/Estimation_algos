#HW3 Aero 626

import math as m
import pylab
import numpy as np
import sympy as sp
import time

import filter_functions as funcs
from numpy import reshape as rs
from numpy import matrix as mat

sp.init_printing()

sys = funcs.robot()
#sys = funcs.particle()
n = int((sys.tf - sys.t0)/sys.h)

def part1_1_EKF():

    sys = funcs.particle()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est = np.zeros((sys.X0.shape[0],n+1))
    X_act = np.zeros((sys.X0.shape[0],n+1))
    P = np.zeros((sys.X0.shape[0]*sys.X0.shape[0],n+1))

    X_est[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))
    X_act[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))

    P[:,0] = sys.P0.reshape((9,))

    for i in range(n):

        X_est[:,i+1], X_act[:,i+1], P[:,i+1] = funcs.EKF(sys, X_est[:,i], X_act[:,i],P[:,i])

    print "X_est:", X_est
    print "X_act:", X_act

    return X_est, X_act, P

def part2_1_EKF():

    sys = funcs.robot()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est = np.zeros((sys.X0.shape[0],n+1))
    X_act = np.zeros((sys.X0.shape[0],n+1))
    P = np.zeros((sys.X0.shape[0]*sys.X0.shape[0],n+1))

    X_est[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))
    X_act[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))

    P[:,0] = sys.P0.reshape((9,))

    for i in range(n):

        X_est[:,i+1], X_act[:,i+1], P[:,i+1] = funcs.EKF(sys, X_est[:,i], X_act[:,i],P[:,i],i+1)

    print "X_est:", X_est
    print "X_act:", X_act

    return X_est, X_act, P

def part2_1_UKF():

    sys = funcs.robot()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est = np.zeros((sys.X0.shape[0],n+1))
    X_act = np.zeros((sys.X0.shape[0],n+1))

    X_est[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))
    X_act[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))
    P = np.zeros((sys.X0.shape[0]*sys.X0.shape[0],n+1))

    P[:,0] = sys.P0.reshape((9,))


    for i in range(n):

        X_est[:,i+1], X_act[:,i+1], P[:,i+1] = funcs.UKF(sys, X_est[:,i], X_act[:,i],P[:,i],i+1)

    #print "X_est:", X_est
    #print "X_act:", X_act

    return X_est, X_act, P

def part1_1_UKF():

    sys = funcs.particle()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est = np.zeros((sys.X0.shape[0],n+1))
    X_act = np.zeros((sys.X0.shape[0],n+1))

    X_est[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))
    X_act[:,0] = np.reshape(sys.X0,(sys.X0.shape[0],))
    P = np.zeros((sys.X0.shape[0]*sys.X0.shape[0],n+1))

    P[:,0] = sys.P0.reshape((9,))


    for i in range(n):

        X_est[:,i+1], X_act[:,i+1], P[:,i+1] = funcs.UKF(sys, X_est[:,i], X_act[:,i],P[:,i])

    #print "X_est:", X_est
    #print "X_act:", X_act

    return X_est, X_act, P

def part1_1_EnKF():

    sys = funcs.particle()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est, X_act, P = funcs.EnKF(sys,n)

    return X_est, X_act, P

def part2_1_EnKF():

    sys = funcs.robot()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est, X_act, P = funcs.EnKF(sys,n)

    return X_est, X_act, P

def part1_1_ParticleF():

    sys = funcs.particle()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est, X_act, P = funcs.ParticleF(sys,n)

    return X_est, X_act, P

def part2_1_ParticleF():

    sys = funcs.robot()

    n = int((sys.tf - sys.t0)/sys.h)

    X_est, X_act, P = funcs.ParticleF(sys,n)

    return X_est, X_act, P

def plot_func(X_est,X_act,P):

    t = np.linspace(0,100,101)
    #t = np.linspace(0,0.5,6)
    fig, ax = pylab.subplots(3,1)
    ax[0].plot(t,X_act[0,:],'bo',markersize=2,linewidth=1,label='Actual x1')
    ax[0].errorbar(t,X_est[0,:],yerr=3*np.sqrt(P[0,:]),fmt='r*',markersize=2,linewidth=1,label='Estimated x1')
    pylab.xlabel('time')
    pylab.ylabel('x1')
    ax[0].legend()

    #ax[0].set_xlim(-0.1,0.7)
    ax[1].plot(t,X_act[1,:],'bo',markersize=2,linewidth=1,label='Actual x2')
    ax[1].errorbar(t,X_est[1,:],yerr=3*np.sqrt(P[4,:]),fmt='r*',markersize=2,linewidth=1,label='Estimated x2')
    ax[1].legend()
    pylab.xlabel('time')
    pylab.ylabel('x2')

    #ax[1].set_xlim(-0.1,0.7)
    ax[2].plot(t,X_act[2,:],'bo',markersize=2,linewidth=1,label='Actual x3')
    ax[2].errorbar(t,X_est[2,:],yerr=3*np.sqrt(P[8,:]),fmt='r*',markersize=2,linewidth=1,label='Estimated x3')
    ax[2].legend()
    pylab.xlabel('time')
    pylab.ylabel('x3')

    #ax[2].set_xlim(-0.1,0.7)
    #pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_PF_5.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)

    # pylab.figure(2)
    #
    # pylab.plot(np.zeros((6,1)),X_act[0,:],'bo',markersize=5,linewidth=2,label='Actual x1')
    # pylab.errorbar(np.zeros((6,1)),X_est[0,:],yerr=3*np.sqrt(P[0,:]),fmt='r*',markersize=5,linewidth=2,label='Estimated x1')
    # pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'1_1_EKF_x1.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    # pylab.legend()

    pylab.show()

def plot_traj(X_est,X_act,P):

    t = np.linspace(0,100,101)
    #t = np.linspace(0,0.5,6)
    pylab.plot(X_est[0,:],X_est[1,:],'-*',markersize=2,linewidth=1,label='Estimated',color='r')
    pylab.plot(X_act[0,:],X_act[1,:],'-o',markersize=2,linewidth=1,label='Actual',color='b')
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.legend()
    pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_PF_traj.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    #pylab.show()


def Monte_carlo_runs(filter,no_runs):

    np.random.seed(1)
    start = time.time()
    error_vec = np.zeros(no_runs)
    for i in range(no_runs):
        X_est, X_act, P = filter()

        error_sum = 0.0
        for j in range(1,n+1): #ignoring 1st step
            err = X_act[:,j] - X_est[:,j]
            norm_err = np.linalg.norm(err)
            error_sum = error_sum + norm_err

        error_vec[i] = error_sum/n #avg norm of error per time step

    end = time.time()
    filter_error = np.mean(error_vec) #avg norm of error per time step
    print "filter error:", filter_error
    print "error var:", np.var(error_vec)
    print "time taken:",end - start
    print "error vec:", error_vec

def Monte_carlo_runs_NEES(filter,no_runs):

    np.random.seed(1)

    nees_err_vec = np.zeros(n)
    for i in range(no_runs):
        X_est, X_act, P = filter()

        for j in range(1,n+1): #ignoring 1st step

            err = mat((X_act[:,j] - X_est[:,j]).reshape((sys.state_dim,1))).astype(np.float64)
            P_j = mat(P[:,j].reshape((sys.state_dim,sys.state_dim))).astype(np.float64)

            print "P:", P_j
            nees_err = np.matmul(np.matmul(err.T,P_j.I),err)
            #print "nees_error:",nees_err
            if nees_err > 10**3:
                nees_err = 0

            nees_err_vec[j-1] = nees_err_vec[j-1] + nees_err
        print "Nees err vec:", nees_err_vec

    nees_err_vec = np.true_divide(nees_err_vec,no_runs)

    print "nees:", nees_err_vec
    plot_nees(nees_err_vec)

def plot_nees(err_vec):

    #t = np.linspace(0.1,0.5,5)
    t = np.linspace(1,100,100)
    lb = 2.36*np.ones(n)
    ub = 3.72*np.ones(n)
    pylab.figure(2)
    pylab.plot(t,err_vec,'.-',markersize=5,linewidth=2,color='b')
    pylab.plot(t,lb)
    pylab.plot(t,ub)
    pylab.xlabel('time')
    pylab.ylabel('Average NEES over 50 runs')
    pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_NEES_EnKF.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    #pylab.show()

if __name__ == '__main__':

    np.random.seed(1)
    X_est, X_act, P = part2_1_ParticleF()

    #Monte_carlo_runs(part2_1_ParticleF,50)
    #Monte_carlo_runs_NEES(part2_1_ParticleF,50)
    #plot_func(X_est,X_act,P)
    plot_traj(X_est,X_act,P)

    #t, X = funcs.ode_integrate(funcs.dydt,X0,t0,tf,h)
    #print(X)
    #print(t)
    #pylab.plot(t,X[2,:],'-o',markersize=5,linewidth=2,label='RK4')
    #
    # pylab.legend()
    # pylab.show()
