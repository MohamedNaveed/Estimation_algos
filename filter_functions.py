import math as m
import numpy as np
import pylab
import sympy as sp
from numpy import reshape as rs
from numpy import matrix as mat
from matplotlib.patches import Ellipse

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

class robot(object):

    t0 = 0
    state_dim = 3
    meas_dim = 3
    X0 = np.array([[0],[0],[m.pi/2]])
    P0 = np.matrix([[.1,0,0],[0,.1,0],[0,0,.4]])
    h = 1
    tf = 100
    G = np.matrix([[h,0,0],[0,h,0],[0,0,h]])
    M = np.matrix([[1,0,0],[0,1,0],[0,0,0]])
    Q = (1.0/h)*np.matrix([[.01,0,0],[0,.01,0],[0,0,0.2]])
    R = (1.0/h)*np.matrix([[.2,0,0],[0,0.2,0],[0,0,0]])

    def kinematics(self,t):

        x1, x2, x3 = sp.symbols('x1 x2 x3')

        v = abs(m.sin(t))
        if t<=50:
            w = 0.1
        elif t<=80 and t>50:
            w = 0.2
        elif t>80:
            w = -0.1

        F = sp.Matrix([x1 + v*sp.cos(x3)*self.h, x2 + v*sp.sin(x3)*self.h, x3 + w*self.h])

        return F, x1, x2, x3

    def jacobian(self,X,t):

        F, x1, x2, x3 = self.kinematics(t)

        J = F.jacobian([x1,x2,x3])

        return J.subs([(x1,X[0]),(x2,X[1]),(x3,X[2])])

    def state_propagate(self,X0, Q,t):

        F, x1, x2, x3 = self.kinematics(t)


        w = np.random.multivariate_normal(np.zeros((self.state_dim,)),Q)
        w = mat(w.reshape((self.state_dim,1))).astype(np.float64)
        X = np.reshape(F.subs([(x1,X0[0]),(x2,X0[1]),(x3,X0[2])]) + np.matmul(self.G,w),(self.state_dim,))

        return X



def lin_obs_model():

    x1, x2, x3 = sp.symbols('x1 x2 x3')

    hx = sp.Matrix([x1,x2,0])

    return hx, x1, x2, x3

def observation(M,X, R):

    hx, x1, x2, x3 = lin_obs_model()
    nu = np.random.multivariate_normal(np.zeros((3,)),R)
    nu = mat(nu.reshape((3,1))).astype(np.float64)
    Y = hx.subs([(x1,X[0]),(x2,X[1]),(x3,X[2])]) + np.matmul(M,nu)

    return np.matrix(Y).astype(np.float64)

def obs_jacobian(X):

    hx, x1, x2, x3 = lin_obs_model()

    H = hx.jacobian([x1,x2,x3])

    return H.subs([(x1,X[0]),(x2,X[1]),(x3,X[2])])

def EKF(system, X_prev_est, X_prev_act, P_prev,t):

    #prediction steps
    P_prev = mat(P_prev.reshape((3,3))).astype(np.float64)
    X_prior = system.state_propagate(X_prev_est,np.zeros((system.state_dim,system.state_dim)),t)
    print "X_prior", X_prior
    A = np.matrix(system.jacobian(X_prev_est,t)).astype(np.float64)

    print "A:",A
    P_prior = np.matmul(np.matmul(A,P_prev),A.T) + np.matmul(np.matmul(system.G,system.Q),system.G.T)

    print "P prior:",P_prior#, " Pinv:", P_prior.I

    X_act = np.reshape(system.state_propagate(X_prev_act,system.Q,t),(system.state_dim,))
    #
    # if t == 5:
    #
    #     vals, vecs = eigsorted(P_prior[0:2,0:2])
    #     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    #     ax = pylab.gca()
    #     for sigma in xrange(1, 4):
    #         w, h = 2 * sigma * np.sqrt(vals)
    #         ell = Ellipse(xy=X_prior[0:2],width=w, height=h,angle=theta,fill=None,color='r')
    #         ell.set_facecolor('none')
    #         ax.add_artist(ell)
    #     #ellipse = Ellipse(xy=X_prior[0:2],width=lambda_*2, height=ell_radius_y*2,fill=None,color='r')
    #
    #
    #     pylab.plot(X_prior[0],X_prior[1],'ro',markersize=2,linewidth=1,label='predicted EKF')
    #     pylab.plot(X_act[0],X_act[1],'bo',markersize=2,linewidth=1,label='Actual')
    #     pylab.legend()
    #     pylab.xlabel('x')
    #     pylab.ylabel('y')
    #     pylab.xlim(-7,7)
    #     pylab.ylim(1,8)
    #     #pylab.show()
    #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_EKF_t5.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)


    if t%5 == 0:
    #update

        Y_act = observation(system.M,X_act, system.R)
        ##print "Yact:",Y_act
        Y_est = observation(system.M,X_prior,np.zeros((system.meas_dim,system.meas_dim)))
        #print "Y est:",Y_est
        H = np.matrix(obs_jacobian(X_prior)).astype(np.float64)
        print "H:",H
        S = np.matmul(np.matmul(H,P_prior),H.T) + system.R


        #since S is singular and only 1 measurement is received
        #K_gain = np.matmul(np.matmul(P_prior,H),S.I)
        K_gain = np.zeros((system.state_dim,system.state_dim))

        if S[1,1] == 0 :
            S[1,1] = 10**(-9)#adding to make it non-singular

        if S[2,2] == 0:
            S[2,2] = 10**(-9)
        #K_gain[0,0] = np.matmul(P_prior[0,:],H[:,0])/S[0,0]
        #K_gain[1,0] = np.matmul(P_prior[1,:],H[:,0])/S[0,0]
        #K_gain[2,0] = np.matmul(P_prior[2,:],H[:,0])/S[0,0]
        print "S:",S.I
        K_gain = np.matmul(np.matmul(P_prior,H.T),S.I)
        print "K:",K_gain
        print "H.TSI", np.matmul(H.T,S.I)
        X_est = np.reshape(X_prior,(3,1)) + np.matmul(K_gain, Y_act - Y_est)
        print "X est:", X_est
        print "Correction:", np.matmul(K_gain, Y_act - Y_est)
        X_est = np.reshape(X_est,(3,))
        print "I -KH",np.eye(3) - np.matmul(K_gain,H)
        P_post = np.matmul(np.eye(3) - np.matmul(K_gain,H), P_prior)
        print "P_post:", P_post

        # if t == 5:
        #
        #     vals, vecs = eigsorted(P_post[0:2,0:2])
        #     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        #     ax = pylab.gca()
        #     for sigma in xrange(1, 4):
        #         w, h = 2 * sigma * np.sqrt(vals)
        #         ell = Ellipse(xy=(X_est[0,0],X_est[0,1]),width=w, height=h,angle=theta,fill=None,color='r')
        #         ell.set_facecolor('none')
        #         ax.add_artist(ell)
        #
        #     pylab.plot(X_est[0,0],X_est[0,1],'ro',markersize=2,linewidth=1,label='updated EKF')
        #     pylab.plot(X_act[0],X_act[1],'bo',markersize=2,linewidth=1,label='Actual')
        #     pylab.legend()
        #     pylab.xlabel('x')
        #     pylab.ylabel('y')
        #     pylab.xlim(-7,7)
        #     pylab.ylim(1,8)
        #     #pylab.show()
        #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_EKF_t5_updated.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)

        return X_est, X_act, P_post.reshape((system.state_dim**2,))

    else:
        return X_prior.reshape((system.state_dim,)), X_act, P_prior.reshape((system.state_dim**2,))

def UKF(system, X_prev_est, X_prev_act, P_prev,t):

    P_prev = mat(P_prev.reshape((3,3))).astype(np.float64)
    print "P_prev:",P_prev
    X_prev_est = np.reshape(X_prev_est,(3,1))

    n = 3

    X_sigma = np.zeros((n,2*n+1))
    W = np.zeros(2*n+1)

    #choosing sigma points and weights
    X_sigma[:,0] = np.reshape(X_prev_est,(3,))
    W[0] = 0.1

    S = np.linalg.cholesky(P_prev)

    for i in range(1,n+1,1):

        X_sigma[:,i] = np.reshape(X_prev_est + np.sqrt(n/(1-W[0]))*S[:,i-1],(n,))
        X_sigma[:,i+n] = np.reshape(X_prev_est - np.sqrt(n/(1-W[0]))*S[:,i-1],(n,))
        W[i] = (1 - W[0])/(2*n)
        W[i+n] = (1 - W[0])/(2*n)

    #print "X_sigma:",X_sigma
    #print "W:", W

    #prediction
    X_prior = np.zeros(3)

    #calculating X_prior
    for i in range(2*n+1):

        X_sigma[:,i] = np.reshape(system.state_propagate(X_sigma[:,i],np.zeros((system.state_dim,system.state_dim)),t),(n,))
        X_prior = X_prior + W[i]*X_sigma[:,i]

    P_prior = np.zeros((3,3))
    #calculating P_prior
    for i in range(2*n+1):
        X_error = np.matrix(X_sigma[:,i] - X_prior).astype(np.float64)
        P_prior = P_prior + W[i]*np.matmul(X_error.T,X_error)

    P_prior = P_prior + np.matmul(np.matmul(system.G,system.Q),system.G.T)
    print "X_prior:",X_prior
    print "P_prior:",P_prior, " Eigen:",np.linalg.eig(P_prior)

    #update
    X_act = np.reshape(system.state_propagate(X_prev_act,system.Q,t),(3,))
    # if t == 5:
    #
    #     vals, vecs = eigsorted(P_prior[0:2,0:2])
    #     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    #     ax = pylab.gca()
    #     for sigma in xrange(1, 4):
    #         w, h = 2 * sigma * np.sqrt(vals)
    #         ell = Ellipse(xy=X_prior[0:2],width=w, height=h,angle=theta,fill=None,color='r')
    #         ell.set_facecolor('none')
    #         ax.add_artist(ell)
    #     #ellipse = Ellipse(xy=X_prior[0:2],width=lambda_*2, height=ell_radius_y*2,fill=None,color='r')
    #
    #
    #     pylab.plot(X_prior[0],X_prior[1],'ro',markersize=2,linewidth=1,label='predicted UKF')
    #     pylab.plot(X_act[0],X_act[1],'bo',markersize=2,linewidth=1,label='Actual')
    #     pylab.legend()
    #     pylab.xlabel('x')
    #     pylab.ylabel('y')
    #     pylab.xlim(-7,7)
    #     pylab.ylim(1,8)
    #     #pylab.show()
    #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_UKF_t5.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)

    if t%5 == 0:
        Y_act = observation(system.M,X_act, system.R)

        #passing sigma points through observation
        Y_est_sigma = np.zeros((n,2*n+1))
        Y_est = np.zeros(3)

        for i in range(2*n+1):
            Y_est_sigma[:,i] = np.reshape(observation(system.M,X_sigma[:,i], np.zeros((system.meas_dim,system.meas_dim))),(3,))
            Y_est = Y_est + W[i]*Y_est_sigma[:,i]

        print "Y_est:",Y_est, " Y_act:", Y_act
        #calculating Pyy
        P_yy = np.zeros((3,3))

        for i in range(2*n+1):
            Y_error = np.matrix(Y_est_sigma[:,i] - Y_est).astype(np.float64)
            P_yy = P_yy + W[i]*np.matmul(Y_error.T,Y_error)

        P_yy = P_yy + system.R

        #calculating Pxy
        P_xy = np.zeros((3,3))
        for i in range(2*n+1):
            X_error = np.matrix(X_sigma[:,i] - X_prior).astype(np.float64)
            Y_error = np.matrix(Y_est_sigma[:,i] - Y_est).astype(np.float64)

            P_xy = P_xy + W[i]*np.matmul(X_error.T,Y_error)

        #Kalman gain
        K_gain = np.zeros((3,3))

        if P_yy[1,1] == 0:
            P_yy[1,1] = 10**(-6)

        if P_yy[2,2] == 0:
            P_yy[2,2] = 10**(-6)
        print "Pxy:",P_xy
        print "Pyy:",P_yy
        K_gain = np.matmul(P_xy,P_yy.I)
        print "K-gain",K_gain

        #state update
        X_est = np.reshape(X_prior,(3,1)) + np.matmul(K_gain, Y_act - np.reshape(Y_est,(3,1)))
        P_post = P_prior - np.matmul(np.matmul(K_gain,P_yy),K_gain.T)

        #print "Cov corr:",np.matmul(np.matmul(K_gain,P_yy),K_gain.T)
        print "X_est:",X_est, "X_act:",X_act
        print "P_post:",P_post," Eigen:",np.linalg.eig(P_post)


        X_est = np.reshape(X_est,(3,))
        X_act = np.reshape(X_act,(3,))
        # if t == 5:
        #
        #     vals, vecs = eigsorted(P_post[0:2,0:2])
        #     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        #     ax = pylab.gca()
        #     for sigma in xrange(1, 4):
        #         w, h = 2 * sigma * np.sqrt(vals)
        #         ell = Ellipse(xy=(X_est[0,0],X_est[0,1]),width=w, height=h,angle=theta,fill=None,color='r')
        #         ell.set_facecolor('none')
        #         ax.add_artist(ell)
        #
        #     pylab.plot(X_est[0,0],X_est[0,1],'ro',markersize=2,linewidth=1,label='updated UKF')
        #     pylab.plot(X_act[0],X_act[1],'bo',markersize=2,linewidth=1,label='Actual')
        #     pylab.legend()
        #     pylab.xlabel('x')
        #     pylab.ylabel('y')
        #     pylab.xlim(-7,7)
        #     pylab.ylim(1,8)
        #     #pylab.show()
        #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_UKF_t5_updated.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)

        return X_est,X_act,P_post.reshape((9,))

    else:
        return X_prior.reshape((system.state_dim,)), X_act, P_prior.reshape((system.state_dim**2,))

def EnKF(system, n):


    X_est = np.zeros((system.X0.shape[0],n+1))
    X_act = np.zeros((system.X0.shape[0],n+1))

    X_est[:,0] = np.reshape(system.X0,(system.X0.shape[0],))
    X_act[:,0] = np.reshape(system.X0,(system.X0.shape[0],))

    P = np.zeros((system.X0.shape[0]*system.X0.shape[0],n+1))

    P[:,0] = system.P0.reshape((9,))

    N = 100 #ensemble size
    X_en = np.random.multivariate_normal(np.reshape(system.X0,(3,)),system.P0,N) #en - ensemble

    X_en = X_en.T #3x100 every column is a random vector.
    np.random.seed(1)
    #print "X_ensemble:",X_en

    Y_en = np.zeros((3,N))


    #fig, ax = pylab.subplots(1,2)
    for t in range(n):


        X_en = np.array(X_en) #matrix to array (after 1st iter)
        #print "X_en:", X_en
        """
        #Visualising particles

        ax[0].plot(t*np.ones((N,1)), X_en[0,:],'ro',markersize=2,linewidth=2)
        ax[0].plot(t, X_act[0,t],'bo',markersize=5,linewidth=2)
        ax[0].legend()
        #ax[0].set_xlim(-0.1,0.7)

        ax[1].plot(t*np.ones((N,1)), X_en[1,:],'ro',markersize=2,linewidth=2)
        ax[1].plot(t, X_act[1,t],'bo',markersize=5,linewidth=2)
        ax[1].legend()
        #ax[1].set_xlim(-0.1,0.7)
        """

        X_act[:,t+1] = np.reshape(system.state_propagate(X_act[:,t],system.Q,t+1),(3,))
        Y_act = observation(system.M,X_act[:,t+1], system.R)

        for i in range(N):
            X_en[:,i] = system.state_propagate(X_en[:,i],system.Q,t+1).reshape((3,)) #propagate dynamics of ensemble

            Y_en[:,i] = (Y_act + np.random.multivariate_normal(np.zeros(3),system.R,1).reshape((3,1))).reshape((3,)) #perturb observations.

        #print "X_en predict:",X_en
        #calculating prior covariance from ensemble
        X_en_bar = np.matmul(mat(X_en).astype(np.float64),(1.0/N)*np.ones((N,N)))
        #print "X_en_bar:", X_en_bar[:,0]
        X_err = mat(X_en - X_en_bar).astype(np.float64)
        P_prior = (1.0/(N-1))*np.matmul(X_err,X_err.T)

        # if t+1 == 5:
        #     X_prior = X_en_bar[:,0]
        #     vals, vecs = eigsorted(P_prior[0:2,0:2])
        #     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        #     ax = pylab.gca()
        #     for sigma in xrange(1, 4):
        #         w, h = 2 * sigma * np.sqrt(vals)
        #         ell = Ellipse(xy=X_prior[0:2],width=w, height=h,angle=theta,fill=None,color='r')
        #         ell.set_facecolor('none')
        #         ax.add_artist(ell)
        #     #ellipse = Ellipse(xy=X_prior[0:2],width=lambda_*2, height=ell_radius_y*2,fill=None,color='r')
        #
        #
        #     pylab.plot(X_prior[0],X_prior[1],'ro',markersize=2,linewidth=1,label='predicted EnKF')
        #     pylab.plot(X_act[0,t+1],X_act[1,t+1],'bo',markersize=2,linewidth=1,label='Actual')
        #     pylab.legend()
        #     pylab.xlabel('x')
        #     pylab.ylabel('y')
        #     pylab.xlim(-7,7)
        #
        #     pylab.ylim(-2,6)
        #     #pylab.show()
        #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_EnKF_t5.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)

        if (t+1)%5 == 0:
            #calculating measurement covariance from ensemble
            Y_err = Y_en - np.matmul(Y_act,np.ones((1,N)))

            Cov_e = (1.0/(N-1))*np.matmul(Y_err,Y_err.T)

            #Kalman update
            H = np.matrix(obs_jacobian(np.mean(X_en,axis=1))).astype(np.float64)

            S = np.matmul(np.matmul(H,P_prior),H.T) + Cov_e

            if S[1,1] == 0.0:
                S[1,1] = 10**(-9)

            if S[2,2] == 0.0:
                S[2,2] = 10**(-9)

            #print "S:",S
            K_gain = np.matmul(np.matmul(P_prior,H.T),S.I)
            X_en = X_en + np.matmul(K_gain,Y_en - np.matmul(H,X_en))

            #calculating post covariance from ensemble
            X_en_bar = np.matmul(mat(X_en).astype(np.float64),(1.0/N)*np.ones((N,N)))
            X_err = mat(X_en - X_en_bar).astype(np.float64)
            P_post = (1.0/(N-1))*np.matmul(X_err,X_err.T)

            X_est[:,t+1] = np.mean(X_en,axis=1).reshape((3,))
            P[:,t+1] = P_post.reshape((9,))

            # if t+1 == 5:
            #
            #     vals, vecs = eigsorted(P_post[0:2,0:2])
            #     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            #     ax = pylab.gca()
            #     for sigma in xrange(1, 4):
            #         w, h = 2 * sigma * np.sqrt(vals)
            #         ell = Ellipse(xy=X_est[0:2,t+1],width=w, height=h,angle=theta,fill=None,color='r')
            #         ell.set_facecolor('none')
            #         ax.add_artist(ell)
            #     #ellipse = Ellipse(xy=X_prior[0:2],width=lambda_*2, height=ell_radius_y*2,fill=None,color='r')
            #
            #
            #     pylab.plot(X_est[0,t+1],X_est[1,t+1],'ro',markersize=2,linewidth=1,label='updated EnKF')
            #     pylab.plot(X_act[0,t+1],X_act[1,t+1],'bo',markersize=2,linewidth=1,label='Actual')
            #     pylab.legend()
            #     pylab.xlabel('x')
            #     pylab.ylabel('y')
            #     pylab.xlim(-7,7)
            #
            #     pylab.ylim(-2,6)
            #     #pylab.show()
            #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_EnKF_t5_updated.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)


        else:
            X_est[:,t+1] = np.mean(X_en,axis=1).reshape((3,))
            P[:,t+1] = P_prior.reshape((9,))

        #print "X_en update:",X_en
        # print "X_mean:",np.mean(X_en,axis=1)
        # print "P_prior:", P_prior
        # print "P post:", P_post
        # print "Y_act:",Y_act
        # print "Cov_e:", Cov_e
        # print "K_gain:",K_gain
        # print "X_est:",X_est[:,t+1]
    #pylab.show()
    #pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_EnKF_ens.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    return X_est, X_act, P

def ParticleF(system,n):

    X_est = np.zeros((system.X0.shape[0],n+1))
    X_act = np.zeros((system.X0.shape[0],n+1))

    X_est[:,0] = np.reshape(system.X0,(system.X0.shape[0],))
    X_act[:,0] = np.reshape(system.X0,(system.X0.shape[0],))

    P = np.zeros((system.X0.shape[0]*system.X0.shape[0],n+1))

    P[:,0] = system.P0.reshape((9,))

    #Sampling
    N = 200#no. of particles
    X_hyps = np.random.multivariate_normal(np.reshape(system.X0,(3,)),system.P0,N) #sampling from prior
    X_hyps = X_hyps.T
    #print "Hyposthesis:",X_hyps
    #w = (1.0/N)*np.ones(N) #weights
    w = np.zeros(N)
    for i in range(N):
        X_err = mat((X_hyps[:,i] - rs(system.X0,(system.state_dim,))).reshape((system.state_dim,1))).astype(np.float64)
        w[i] = m.exp(-0.5*np.matmul(np.matmul(X_err.T,system.P0.I),X_err))/(m.sqrt((2*m.pi)**system.state_dim)*np.sqrt(np.linalg.det(system.P0))) #calculating l

    w = np.true_divide(w,np.sum(w))
    np.random.seed(1)
    """
    #Visualising particles
    pylab.plot(X_hyps[1,:],w,'ro',markersize=2,linewidth=1,label='Particles x1')
    pylab.legend()
    pylab.show()
    """

    for t in range(n):

        X_act[:,t+1] = np.reshape(system.state_propagate(X_act[:,t],system.Q,t+1),(3,))
        Y_act = observation(system.M,X_act[:,t+1], system.R)

        for i in range(N):
            X_hyps[:,i] = system.state_propagate(X_hyps[:,i],system.Q,t+1).reshape((3,)) #propagate dynamics of particles

            if (t+1)%5 == 0:
                observ_err = Y_act - observation(system.M,X_hyps[:,i],np.zeros((system.meas_dim,system.meas_dim))) #observation error

                if system.R[2,2] == 0:
                    system.R[2,2] = 10**(-6)

                #likelihood
                w[i] = w[i]*m.exp(-0.5*np.matmul(np.matmul(observ_err.T,system.R.I),observ_err))/(m.sqrt((2*m.pi)**system.meas_dim)*np.sqrt(np.linalg.det(system.R))) #calculating likelihood and updating weight

        w = np.true_divide(w,np.sum(w)) #normalising weights

        # pylab.figure(1)
        # pylab.plot(X_hyps[1,:],w,'ro',markersize=2,linewidth=1,label='Particles x1')
        # pylab.legend()

        #plot particles
        # if (t+1) == 5:
        #
        #     pylab.plot(X_hyps[0,:],X_hyps[1,:],'ro',markersize=2,linewidth=1,label='Predicted PF')
        #     pylab.plot(X_act[0,t+1],X_act[1,t+1],'bo',markersize=2,linewidth=1,label='Actual')
        #     pylab.legend()
        #     pylab.xlabel('x')
        #     pylab.ylabel('y')
        #     #pylab.show()
        #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_PF_t5.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)

        #resampling
        c = np.zeros(N)
        c[0] = 0
        for i in range(1,N):
            c[i] = c[i-1] + w[i]

        u = np.zeros(N)
        u[0] = np.random.uniform(0,1.0/N)
        i = 0 #starting at bottom of cdf
        for j in range(N):

            u[j] = u[0] + (1.0/N)*j

            while u[j] > c[i]:
                i = i + 1
                i = min(N-1,i)
                if i == N-1:
                    break
                #print "j:",j,"i:",i

            X_hyps[:,j] = X_hyps[:,i]
            w[j] = 1.0/N

        #print "w:",w
        #print "X_hyps:",X_hyps[0,:]
        # if (t+1) == 5:
        #     pylab.plot(X_hyps[0,:],X_hyps[1,:],'ro',markersize=2,linewidth=1,label='updated PF')
        #     pylab.plot(X_act[0,t+1],X_act[1,t+1],'bo',markersize=2,linewidth=1,label='Actual')
        #     pylab.legend()
        #     pylab.xlabel('x')
        #     pylab.ylabel('y')
        #     pylab.xlim(-4,4)
        #     pylab.ylim(-1,4)
        #     #pylab.show()
        #     pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'2_1_PF_t5_update.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
        #calculating estimate
        X_temp = np.zeros(3)
        for i in range(N):
            X_temp = X_temp + w[i]*X_hyps[:,i]

        X_est[:,t+1] = X_temp.reshape((3,))

        #calculating variance
        P_temp = np.zeros((system.state_dim,system.state_dim))
        for i in range(N):
            X_err = mat((X_hyps[:,i] - X_est[:,t+1]).reshape(3,1)).astype(np.float64)
            P_temp = P_temp + np.matmul(X_err,X_err.T)

        P[:,t+1] = (1.0/(N-1))*P_temp.reshape((9,))
        #print "P:", P[:,t+1]
        # pylab.figure(2)
        # pylab.plot(X_hyps[1,:],w,'ro',markersize=2,linewidth=1,label='Particles x1')
        # pylab.legend()
        # pylab.show()

    return X_est, X_act, P
