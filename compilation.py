# -*- coding: utf-8 -*-
"""
Created on Mon May 23 09:30:23 2022

@author: Charlie
"""


import numpy as np
from scipy.linalg import expm
import cirq
import pickle

class Compilation():
    
    def __init__(self):
        
        self.make_gates()
        
    def make_gates(self):
        
        #define gates unitaries
        self.I = np.matrix([[1,0],[0,1]]) #identity
        self.X = np.matrix([[0,1],[1,0]]) #pi x
        self.X2 = expm(-1j*self.X*np.pi/4) #pi/2 x
        self.mX2 = expm(1j*self.X*np.pi/4) #-pi/2 x
        self.Y = np.matrix([[0,-1j],[1j,0]]) #pi y
        self.Y2 = expm(-1j*self.Y*np.pi/4) #pi/2 y
        self.mY2 = expm(1j*self.Y*np.pi/4) #-pi/2 y
        self.Z = np.matrix([[1,0],[0,-1]]) #pi z
        self.CNOT = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        self.SWAP = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        self.iSWAP = np.matrix([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]])
        
        I = self.I 
        X = self.X
        X2 = self.X2 
        mX2 = self.mX2 
        Y = self.Y 
        Y2 = self.Y2 
        mY2 = self.mY2
        Z = self.Z 

        
        #define cirq gates
        q1,q2 = cirq.LineQubit.range(2)
        X_1 = cirq.X(q1)
        X2_1 = cirq.X(q1)**0.5
        mX2_1 = cirq.X(q1)**-0.5
        Y_1 = cirq.Y(q1)
        Y2_1 = cirq.Y(q1)**0.5
        mY2_1 = cirq.Y(q1)**-0.5
        X_2 = cirq.X(q2)
        X2_2 = cirq.X(q2)**0.5
        mX2_2 = cirq.X(q2)**-0.5
        Y_2 = cirq.Y(q2)
        Y2_2 = cirq.Y(q2)**0.5
        mY2_2 = cirq.Y(q2)**-0.5
        self.CNOT_12_XZ = [cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=0.5,z_exponent=-1)(q1),
                   cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=1)(q2),
                   cirq.ISWAP(q1,q2)**0.5,
                   cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=1,z_exponent=0)(q1),
                   cirq.ISWAP(q1,q2)**0.5,
                   cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=0.5,z_exponent=0.5)(q1),
                   cirq.PhasedXZGate(axis_phase_exponent=-1,x_exponent=0.5,z_exponent=1)(q2)
                   ] #compilation of CNOT in terms of phased XZ and sqiSWAP
        
        self.iSWAP_12_XZ = [cirq.ISWAP(q1,q2)**0.5,cirq.ISWAP(q1,q2)**0.5] #compilation of iSWAP in terms of phased XZ and sqiSWAP
        
        self.SWAP_12_XZ = [cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0.5,z_exponent=0.5)(q1),
                   cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=0.5,z_exponent=0)(q2),
                   cirq.ISWAP(q1,q2)**0.5,
                   cirq.PhasedXZGate(axis_phase_exponent=-1,x_exponent=0.5,z_exponent=1)(q1),
                   cirq.PhasedXZGate(axis_phase_exponent=-1,x_exponent=0.5,z_exponent=1)(q2),
                   cirq.ISWAP(q1,q2)**0.5,
                   cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=0.5,z_exponent=0)(q1),
                   cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=0.5,z_exponent=0)(q2),
                   cirq.ISWAP(q1,q2)**0.5,
                   cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=-0.5)(q1),
                   cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=1)(q2)
                   ] #compilation of SWAP in terms of phased XZ and sqiSWAP    
        
        # reduced set of single qubit Clifford unitaries
        self.C1_reduced_u = [I,
                      mX2,
                      mY2@X,
                      mY2@mX2,
                      X2@Y2,
                      X2@Y2@mX2] 
        
        # reduced set of single qubit Clifford Cirq definitions on qubit 1
        self.C1_reduced_q1_XZ = [[],
                         cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=-0.5,z_exponent=0)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=-0.5,z_exponent=1)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=-0.5,z_exponent=-0.5)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0.5,z_exponent=0.5)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=0.5)(q1)]
        
        # reduced set of single qubit Clifford Cirq definitions on qubit 2
        self.C1_reduced_q2_XZ = [[],
                         cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=-0.5,z_exponent=0)(q2),
                         cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=-0.5,z_exponent=1)(q2),
                         cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=-0.5,z_exponent=-0.5)(q2),
                         cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0.5,z_exponent=0.5)(q2),
                         cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=0.5)(q2)]
        
        # full single qubit Clifford group unitaries
        self.C1_full_u = [I,
                   X,
                   Y,
                   X@Y,
                   Y2@X2,
                   mY2@X2,
                   Y2@mX2,
                   mY2@mX2,
                   X2@Y2,
                   mX2@Y2,
                   X2@mY2,
                   mX2@mY2,
                   X2,
                   mX2,
                   Y2,
                   mY2,
                   X2@Y2@mX2,
                   X2@mY2@mX2,
                   Y2@X,
                   mY2@X,
                   X2@Y,
                   mX2@Y,
                   X2@Y2@X2,
                   mX2@Y2@mX2]
        
        # S1 unitaries, see https://www.nature.com/articles/nature13171
        self.S1_u = [I,
              X2@Y2,
              mY2@mX2]
        
        # S1 Cirq definitions on qubit 1
        self.S1_q1_XZ = [[],
                cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0.5,z_exponent=0.5)(q1),
                cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=-0.5,z_exponent=-0.5)(q1)]
        
        # S1 Cirq definitions on qubit 2
        self.S1_q2_XZ = [[],
                cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0.5,z_exponent=0.5)(q2),
                cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=-0.5,z_exponent=-0.5)(q2)]
        
        self.paulis = [I,X,Y,Z]
        self.two_qubit_paulis = [np.kron(i,j) for i in self.paulis for j in self.paulis]
        
        # ordered list of two qubit Pauli operator labels
        self.pauli_table = ['II',
                            'IX',
                            'IY',
                            'IZ',
                            'XI',
                            'XX',
                            'XY',
                            'XZ',
                            'YI',
                            'YX',
                            'YY',
                            'YZ',
                            'ZI',
                            'ZX',
                            'ZY',
                            'ZZ']
        
        # ordered list of Clifford tableau columns for each Pauli product
        self.symplectic_table = [[0,0,0,0],
                                 [0,0,1,0],
                                 [0,0,1,1],
                                 [0,0,0,1],
                                 [1,0,0,0],
                                 [1,0,1,0],
                                 [1,0,1,1],
                                 [1,0,0,1],
                                 [1,1,0,0],
                                 [1,1,1,0],
                                 [1,1,1,1],
                                 [1,1,0,1],
                                 [0,1,0,0],
                                 [0,1,1,0],
                                 [0,1,1,1],
                                 [0,1,0,1]]
        
    def get_pauli_prod(self,m):
    #input: tensor product of two Pauli matrices
    #output: tableau column of m
    
        for i,p in enumerate(self.two_qubit_paulis):
            prod = m@p
            if np.trace(prod)>3.9:
                return self.symplectic_table[i],0
            if np.trace(prod)<-3.9:
                return self.symplectic_table[i],1

    # The following functions output unitaries for two-qubit Cliffords using the reduced
    # single qubit set. These will be used to make symplectic matrices.
    def make_C1_r_u(self,i,j):
        return np.matrix(np.kron(self.C1_reduced_u[i],self.C1_reduced_u[j]))
    
    def make_CNOT_r_u(self,i,j,k,l):
        return np.matrix(np.kron(self.S1_u[k],self.S1_u[l])@self.CNOT@np.kron(self.C1_reduced_u[i],self.C1_reduced_u[j]))
    
    def make_iSWAP_r_u(self,i,j,k,l):
        return np.matrix(np.kron(self.S1_u[k],self.S1_u[l])@self.iSWAP@np.kron(self.C1_reduced_u[i],self.C1_reduced_u[j]))
    
    def make_SWAP_r_u(self,i,j):
        return np.matrix(self.SWAP@np.kron(self.C1_reduced_u[i],self.C1_reduced_u[j]))
    
    # The following functions output unitaries for two-qubit Cliffords using the full
    # single qubit set. These are used to verify the full set is a unitary 2-design
    def make_C1_f_u(self,i,j):
        return np.matrix(np.kron(self.C1_full_u[i],self.C1_full_u[j]))
    
    def make_CNOT_f_u(self,i,j,k,l):
        return np.matrix(np.kron(self.S1_u[k],self.S1_u[l])@self.CNOT@np.kron(self.C1_full_u[i],self.C1_full_u[j]))
    
    def make_iSWAP_f_u(self,i,j,k,l):
        return np.matrix(np.kron(self.S1_u[k],self.S1_u[l])@self.iSWAP@np.kron(self.C1_full_u[i],self.C1_full_u[j]))
    
    def make_SWAP_f_u(self,i,j):
        return np.matrix(self.SWAP@np.kron(self.C1_full_u[i],self.C1_full_u[j]))
    
    # The following functions output Cirq objects for two-qubit Cliffords using the reduced
    # single qubit set. 
    def make_C1_r_XZ(self,i,j):
        return cirq.Circuit([self.C1_reduced_q1_XZ[i],self.C1_reduced_q2_XZ[j]])
    
    def make_CNOT_XZ(self,i,j,k,l):
        return cirq.Circuit([self.C1_reduced_q1_XZ[i],self.C1_reduced_q2_XZ[j],
                self.CNOT_12_XZ,
                self.S1_q1_XZ[k],self.S1_q2_XZ[l]])
    
    def make_iSWAP_XZ(self,i,j,k,l):
        return cirq.Circuit([self.C1_reduced_q1_XZ[i],self.C1_reduced_q2_XZ[j],
                self.iSWAP_12_XZ,
                self.S1_q1_XZ[k],self.S1_q2_XZ[l]])
    
    def make_SWAP_XZ(self,i,j):
        return cirq.Circuit([self.C1_reduced_q1_XZ[i],self.C1_reduced_q2_XZ[j],
                self.SWAP_12_XZ])
    
    def get_symplectic_and_phase(self,m):
        #Turns a two-qubit unitary into the full tableau representation
        #input: two-qubit unitary m
        #outputs: s: symplectic matrix corresponding to m
        #         p: phase vector corresponding to m 
        s = np.zeros([4,4])
        p = np.zeros(4)
        
        s[:,0],p[0] = self.get_pauli_prod(m.H@np.kron(self.X,self.I)@m)
        s[:,1],p[1] = self.get_pauli_prod(m.H@np.kron(self.I,self.X)@m)
        s[:,2],p[2] = self.get_pauli_prod(m.H@np.kron(self.Z,self.I)@m)
        s[:,3],p[3] = self.get_pauli_prod(m.H@np.kron(self.I,self.Z)@m)
        
        return s,p
    
    def symplectic_to_int(m):
        #Turns a symplectic matrix into a unique integer. Used to verify all 720 symplectics
        #are made
        #input: symplectic matrix
        #output: 16 bit integer
        flat_m = m.ravel()
        num = 0
        
        for i in range(16):
            num += 2**i * flat_m[i]
        
        return num
    
    def phase_to_int(p):
        #Turns a phase vector into a unique integer
        #input: phase vector
        #output: 4 bit integer
        num = 0
        
        for i in range(4):
            num += 2**i * p[i]
            
        return num
    
    def full_matrix_to_int(m,p):
        #Turns a tableau into a unique integer
        #inputs: m: symplectic matrix
        #        p: phase vector
        #output: 20 bit integer
        flat_m = m.ravel()
        num = 0
        
        for i in range(16):
            num += 2**i * flat_m[i]
            
        for i in range(4):
            num += 2**(i+16) * p[i]
            
        return num
    
    def is_symplectic(m):
        #Checks if a matrix is symlectic
        #input: matrix
        #output: Boolean, True if m is symplectic
        Omega = np.matrix([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
        test = m.T@Omega@m
        
        if test.all() == Omega.all():
            return True
        else:
            return False
        
    def make_symplectic_cliffords(self):
    #makes full set of synplectic matrices using reduced single qubit Clifford set
    # output: cliffords: list of unitaries for reduced two-qubit Clifford group
    #         commands: list of instructions to make gates, used for debug
    #         circuits: list of Cirq objects for reduced two-qubit Clifford group 
        cliffords = []
        circuits = []
        
        # single qubit class
        for i in range(6):
            for j in range(6):
                cliffords.append(self.make_C1_r(i,j))
                circuits.append(self.make_C1_r_XZ(i,j))
        
        # CNOT class
        for i in range(6):
            for j in range(6):
                for k in range(3):
                    for l in range(3):
                        cliffords.append(self.make_CNOT_r(i,j,k,l))
                        circuits.append(self.make_CNOT_r_XZ(i,j,k,l))
                            
        # iSWAP class
        for i in range(6):
            for j in range(6):
                for k in range(3):
                    for l in range(3):
                        cliffords.append(self.make_iSWAP_r(i,j,k,l))
                        circuits.append(self.make_iSWAP_r_XZ(i,j,k,l))
                        
        # SWAP class
        for i in range(6):
            for j in range(6):
                cliffords.append(self.make_SWAP_r(i,j))
                circuits.append(self.make_SWAP_r_XZ(i,j))
                
        self.cliffords_reduced = cliffords
        self.circuits_reduced = circuits
    
    def make_all_cliffords(self):
        # makes full two-qubit clifford group
        # output: cliffords: list of unitaries for two-qubit Clifford group
        #         commands: list of instructions to make gates, used for debug
        cliffords = []
        
        # single qubit class
        for i in range(24):
            for j in range(24):
                cliffords.append(self.make_C1_f(i,j))
        
        # CNOT class
        for i in range(24):
            for j in range(24):
                for k in range(3):
                    for l in range(3):
                        cliffords.append(self.make_CNOT_f(i,j,k,l))
                            
        # iSWAP class
        for i in range(24):
            for j in range(24):
                for k in range(3):
                    for l in range(3):
                        cliffords.append(self.make_iSWAP_f(i,j,k,l))
                        
        # SWAP class
        for i in range(24):
            for j in range(24):
                cliffords.append(self.make_SWAP_f(i,j))
                
        self.cliffords_full = cliffords
        
        