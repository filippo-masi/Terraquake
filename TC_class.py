import numpy as np
from scipy import optimize
from scipy.integrate import solve_ivp

class terracotta:
    def __init__(self, K, G, M, ω, λ, pI, c, r, η, Γ):
        """
        Initialize the terracotta class with model parameters.
        Arguments:
        - K: Bulk modulus-like parameter.
        - G: Shear modulus-like parameter.
        - M: Critical stress ratio.
        - ω: Position of the critical state line.
        - λ: Slope of the isotropic compression line.
        - pI: Constant defining the isotropic compression line.
        - c: Deviatoric plastic transport coefficient.
        - r: Ratio of volumetric to deviatoric plastic transport coefficients.
        - η: Constant parameter specifying the energy sink.
        - Γ: Fixed coefficient that sets the unit of the meso-temperature.
        """
        
        self.K = K
        self.G = G
        self.M = M
        self.ω = ω
        self.λ = λ
        self.pI = pI
        self.r = r
        self.c = c
        self.η = η
        self.Γ = Γ

    def Voigt_to_Tensor(self,v, strain=False):
        """
        Convert a Voigt vector representation to a tensor.
        Arguments:
        - v (np.3darray): Input vector in Voigt notation.
        - strain (bool): Whether the input represents strain (affects scaling).
        Returns:
            np.3x3darray: Tensor representation of the input.
        """
        m = 1.
        if strain == True: m = .5
        tensor=np.asarray([v[0],   m*v[5],  m*v[4],
                           m*v[5],   v[1],  m*v[3],
                           m*v[4],  m*v[3], v[2]], dtype=np.float64)
        return tensor.reshape((-1,3))

    def Tensor_to_Voigt(self,T, strain=False):
        """
        Convert a tensor to a Voigt vector representation.
        Arguments:
        - T (np.3x3darray): Input tensor.
        - strain (bool): Whether the tensor represents strain (affects scaling).
        Returns:
            np.3darray: Vector in Voigt notation.
        """
        m = 1.
        if strain == True: m = 2.
        vector=np.asarray([T[0,0], T[1,1], T[2,2], m*T[1,2], m*T[0,2], m*T[0,1]], dtype=np.float64)
        return vector

    def find_volumetric(self,tensor,strain=False):
        """
        Compute the volumetric component (trace) of a tensor.
        Arguments:
        - tensor (np.3x3darray): Input tensor.
        - strain (bool): Whether the tensor represents strain (affects scaling).
        Returns:
            float: Volumetric component.
        """
        m = 1/3
        if strain == True: m = 1
        return m*np.trace(tensor)

    def find_deviatoric_tensor(self,tensor):
        """
        Compute the deviatoric part of a tensor.
        Arguments:
        - tensor (np.3x3darray): Input tensor.
        Returns:
            np.3x3darray: Deviatoric tensor (trace removed).
        """
        return tensor - np.trace(tensor)*np.eye(3)/3

    def find_dev(self,tensor,strain=False):
        """
        Compute the deviatoric magnitude of a tensor.
        Arguments:
        - tensor (np.3x3darray): Input tensor.
        - strain (np.6darray): Whether the tensor represents strain (affects scaling).
        Returns:
            float: Deviatoric magnitude.
        """
        m = 2/3
        if strain == False: m = 3/2
        dev_tensor = self.find_deviatoric_tensor(tensor)
        dev = np.sum(np.multiply(dev_tensor,dev_tensor))         
        return np.sqrt(m*dev)


    def find_Cijkl(self,phi,eps_e):
        """
        Compute the fourth-order stiffness tensor Cijkl based on the model.
        Arguments:
        - phi (float): Solid fraction.
        - eps_e (np.ndarray): Elastic strain in Voigt notation.
        Returns:
            np.ndarray: Stiffness tensor in Voigt notation.
        """
        eps_e_tensor = self.Voigt_to_Tensor(eps_e, strain=True)
        eps_ev = self.find_volumetric(eps_e_tensor, strain=True)

        # Components of the stiffness tensor based on the model equations.
        C1111 = self.K*eps_ev+4*self.G*eps_e_tensor[0,0]
        C1122 = (self.K-2*self.G)*eps_ev+2*self.G*(eps_e_tensor[0,0]+eps_e_tensor[1,1])
        C1133 = (self.K-2*self.G)*eps_ev+2*self.G*(eps_e_tensor[0,0]+eps_e_tensor[2,2])
        C2211 = (self.K-2*self.G)*eps_ev+2*self.G*(eps_e_tensor[0,0]+eps_e_tensor[1,1])
        C2222 = self.K*eps_ev+4*self.G*eps_e_tensor[1,1]
        C2233 = (self.K-2*self.G)*eps_ev+2*self.G*(eps_e_tensor[1,1]+eps_e_tensor[2,2])
        C3311 = (self.K-2*self.G)*eps_ev+2*self.G*(eps_e_tensor[0,0]+eps_e_tensor[2,2])
        C3322 = (self.K-2*self.G)*eps_ev+2*self.G*(eps_e_tensor[1,1]+eps_e_tensor[2,2])
        C3333 = self.K*eps_ev+4*self.G*eps_e_tensor[2,2]
        C1123 = 2*self.G*eps_e_tensor[1,2]
        C1131 = 2*self.G*eps_e_tensor[0,2]
        C1112 = 2*self.G*eps_e_tensor[0,1]
        C2223 = 2*self.G*eps_e_tensor[1,2]
        C2231 = 2*self.G*eps_e_tensor[0,2]
        C2212 = 2*self.G*eps_e_tensor[0,1]
        C3323 = 2*self.G*eps_e_tensor[1,2]
        C3331 = 2*self.G*eps_e_tensor[0,2]
        C3312 = 2*self.G*eps_e_tensor[0,1]
        C2323 = 2*self.G*eps_ev
        C3131 = 2*self.G*eps_ev
        C1212 = 2*self.G*eps_ev
        
        tensor = np.asarray([[C1111, C1122, C1133, C1123, C1131, C1112],
                             [C2211, C2222, C2233, C2223, C2231, C2212],
                             [C3311, C3322, C3333, C3323, C3331, C3312],
                             [C1123, C2223, C3323, C2323, 0.0,   0.0  ],
                             [C1131, C2231, C3331, 0.0,   C3131,   0.0],
                             [C1112, C2212, C3312, 0.0,   0.0,   C1212]],dtype=np.float64)
        return tensor*phi**6
    
    
    def find_Fij(self,phi,sig_e):
        """
        Compute derivative of the elastic stress tensor wrt the solid fraction.
        Arguments:
        - phi (float): Solid fraction.
        - sig_e (np.ndarray): Elastic stress in Voigt notation.
        Returns:
            np.ndarray: Derivative F_ij.
        """
        return 6*sig_e/phi

    def find_pc(self,phi):
        """
        Compute isotropic compression line.
        Arguments:
        - phi (float): Solid fraction.
        Returns:
            float: Isotropic compression pressure.
        """
        pc = self.pI*phi**self.λ
        return pc
        
    def sigmaD(self,Tm,dot_eps):
        """
        Compute (dissipative) viscous stress.
        Arguments:
        - Tm (float): Meso-temperature.
        - dot_eps (np.ndarray): Strain rate in Voigt notation.
        Returns:
            np.ndarray: Viscous stress in Voigt notation.
        """
        return Tm*self.bar_sigmaD(dot_eps)
        
    def bar_sigmaD(self,dot_eps):
        """
        Compute rate component of the (dissipative) viscous stress.
        Arguments:
        - Tm (float): Meso-temperature.
        - dot_eps (np.ndarray): Strain rate in Voigt notation.
        Returns:
            np.ndarray: Rate component of the viscous stress in Voigt notation.
        """
        dot_e_tensor = self.Voigt_to_Tensor(dot_eps,strain=True)
        dot_ev = self.find_volumetric(dot_e_tensor,strain=True)
        dot_edev = self.find_deviatoric_tensor(dot_e_tensor)
        deltaij = np.array([1,1,1,0,0,0])
        sigmaD_tensor = (2*self.c/self.Γ)*(self.r*dot_ev*np.eye(3,3)+2/3*dot_edev)
        sigmaD = self.Tensor_to_Voigt(sigmaD_tensor)
        return sigmaD

    def evol_sigmaD(self,Tm, dot_Tm, dot_eps, ddot_eps):
        """
        Compute rate of change of the viscous stress.
        Arguments:
        - Tm (float): Meso-temperature.
        - dot_Tm (float): Rate of change of Tm.
        - dot_eps (np.ndarray): Strain rate in Voigt notation.
        - ddot_eps (np.ndarray): Second derivative of strain rate.
        Returns:
            np.ndarray: Rate of change of the viscous stress in Voigt notation.
        """
        ddot_e_tensor = self.Voigt_to_Tensor(ddot_eps,strain=True)
        ddot_ev = self.find_volumetric(ddot_e_tensor,strain=True)
        ddot_edev = self.find_deviatoric_tensor(ddot_e_tensor)
        
        dsDde = (self.r-2/9)*ddot_ev*np.eye(3,3)+2/3*ddot_e_tensor
        dsigmaDddote = (2*self.c/self.Γ)* Tm * self.Tensor_to_Voigt(dsDde)
        return dsigmaDddote + self.bar_sigmaD(dot_eps)*dot_Tm  

    def evol_sigmaE(self, sigma_el, phi, dot_phi, epsilon_el, dot_epsilon_el):
        """
        Compute rate of change of the elastic stress.
        Arguments:
        - sigma_el (ndarray): Elastic stress in Voigt notation.
        - phi (float): Solid fraction.
        - dot_phi (float): Rate of change of phi.
        - epsilon_el (ndarray): Elastic strain in Voigt notation.
        - dot_epsilon_el (ndarray): Rate of change of epsilon_el.
        Returns:
            ndarray: Rate of change of the elastic stress in Voigt notation.
        """
        return self.find_Cijkl(phi, epsilon_el) @ dot_epsilon_el + self.find_Fij(phi, sigma_el) * dot_phi

    def evol_pT(self, Tm, dot_Tm):
        """
        Compute rate of change of the thermodynamic pressure.
        Arguments:
        - Tm (float): Meso-temperature.
        - dot_Tm (float): Rate of change of Tm.
        Returns: 
            float: Rate of change of pT.
        """
        return 2*Tm/self.Γ*dot_Tm
        
    def evol_eps_p(self,phi,Tm,sig_e):
        """
        Compute the evolution of plastic strain.
        Arguments:
        - phi (float): Solid fraction.
        - Tm (float): Meso-temperature.
        - sig_e (np.ndarray): Elastic stress in Voigt notation.
        Returns:
            np.ndarray: Plastic strain rate in Voigt notation.
        """
        pc = self.find_pc(phi)
        sig_e_tensor = self.Voigt_to_Tensor(sig_e)
        pe = self.find_volumetric(sig_e_tensor)
        qe = self.find_dev(sig_e_tensor)   
        sedev = self.find_deviatoric_tensor(sig_e_tensor)
        B = Tm/(self.M**2 * pc)*np.sqrt(self.η/self.c)
        dot_epl_ij = B*(self.M**2*pe-qe**2/pe)*np.eye(3,3)/(3*np.sqrt(self.r))+3/2*B*self.M/self.ω*sedev
        dot_eps_pl = self.Tensor_to_Voigt(dot_epl_ij,strain=True)
        return dot_eps_pl

    def evol_eps_e(self,phi,Tm,sig_e,dot_eps):
        """
        Compute the evolution of elastic strain.
        Arguments:
        - phi (float): Solid fraction.
        - Tm (float): Meso-temperature.
        - sig_e (np.ndarray): Elastic stress in Voigt notation.
        - dot_eps (np.ndarray): Total strain rate in Voigt notation.
        Returns:
            np.ndarray: Elastic strain rate in Voigt notation.
        """
        return dot_eps - self.evol_eps_p(phi,Tm,sig_e)

    def evol_phi(self,phi,dot_eps):
        """
        Compute the evolution of solid fraction.
        Arguments:
        - phi (float): Solid fraction.
        - dot_eps (np.ndarray): Total strain rate in Voigt notation.
        Returns:
            float: Rate of change of solid fraction.
        """
        dot_eps_tensor = self.Voigt_to_Tensor(dot_eps, strain=True)
        dot_eps_v = self.find_volumetric(dot_eps_tensor, strain=True)
        return phi*dot_eps_v

    def evol_Tm(self,Tm,dot_eps):
        """
        Compute the evolution of meso-temperature.
        Arguments:
        - Tm (float): Meso-temperature.
        - dot_eps (np.ndarray): Total strain rate in Voigt notation.
        Returns: 
            float: Rate of change of Tm.
        """
        dot_e_tensor = self.Voigt_to_Tensor(dot_eps,strain=True)
        dot_ev = self.find_volumetric(dot_e_tensor,strain=True)
        dot_es = self.find_dev(dot_e_tensor,strain=True)
        return self.c*(self.r*dot_ev**2+dot_es**2)-self.η*Tm**2
            
    def compute_stress(self,sig_e,Tm,dot_eps):
        """
        Computes the total stress tensor, decomposing it into contributions
        from elastic, thermodynamic, and deviatoric stresses.
    
        Parameters:
            sig_e (np.ndarray): Elastic stress in Voigt notation.
            Tm (float): Meso-temperature parameter.
            dot_eps (np.ndarray): Strain rate in Voigt notation.
    
        Returns:
            np.ndarray: Concatenated stress components in Voigt notation, including:
                - Total stress
                - Viscous stress
                - Thermodynamic pressure (pT).
        """
        sig_e_tensor = self.Voigt_to_Tensor(sig_e)
        sig_D_tensor = self.Voigt_to_Tensor(self.sigmaD(Tm,dot_eps))
        pT = Tm**2/self.Γ
        sigma = sig_e_tensor+pT*np.eye(3,3)+sig_D_tensor
        return np.hstack((self.Tensor_to_Voigt(sigma),self.Tensor_to_Voigt(sig_D_tensor),self.Tensor_to_Voigt(pT*np.eye(3,3))))

    def compute_total_stress(self,sig_e,Tm,dot_eps):
        """
        Computes the total stress tensor, including elastic,
        viscous stresses and thermodynamic pressure in Voigt notation.
    
        Parameters:
            sig_e (np.ndarray): Elastic stress in Voigt notation.
            Tm (float): Meso-temperature.
            dot_eps (np.ndarray): Strain rate in Voigt notation.
    
        Returns:
            np.ndarray: Total stress in Voigt notation.
        """
        sig_e_tensor = self.Voigt_to_Tensor(sig_e)
        sig_D_tensor = self.Voigt_to_Tensor(self.sigmaD(Tm,dot_eps))
        pT = Tm**2/self.Γ
        sigma = sig_e_tensor+pT*np.eye(3,3)+sig_D_tensor
        return self.Tensor_to_Voigt(sigma)

    def find_phi0_(self,pe):
        """
        Computes solid fraction at a given elastic pressure `pe`.
    
        Parameters:
            pe (float): Elastic pressure.
    
        Returns:
            float: Porosity `phi`.
        """
        return (pe/self.pI)**(1/self.λ)
    
    def find_phi0(self, p, O):
        """
        Computes the initial porosity `phi` based on pressure and Overconsolidation ratio.
    
        Parameters:
            p (float): Applied pressure.
            O (float): Overconsolidation ratio.
    
        Returns:
            float: Solid fraction.
        """
        return (p*O/self.pI)**(1/self.λ)
        
    def find_ev0(self,pe,phi):
        """
        Computes the initial volumetric strain rate.
    
        Parameters:
            pe (float): Elastic isotropic stress.
            phi (float): Solid fraction.
    
        Returns:
            float: Initial volumetric elastic strain.
        """
        return np.sqrt(2*pe/(self.K*phi**6))
    

    def evol_state_eps_driven(self,t,statet,dot_eps,eps_dot_n1,dt):
        """
        Governs the evolution of the state variables for a strain-driven system.
    
        Parameters:
            t (float): Current time.
            statet (np.ndarray): Current state variables [phi, eps_e, Tm, sig_e, eps, sig].
            dot_eps (np.ndarray): Applied strain rate in Voigt notation.
            eps_dot_n1 (np.ndarray): Previous strain rate in Voigt notation.
            dt (float): Time step size.
    
        Returns:
            np.ndarray: Rate of change of all state variables.
        """
        phi = statet[0]
        eps_e = statet[1:7]
        Tm = statet[7]
        sig_e = statet[8:14]
        eps = statet[14:20]
        sig = statet[20:26]
        
        dot_Tm = self.evol_Tm(Tm,dot_eps)
        dot_phi = self.evol_phi(phi,dot_eps)
        dot_eps_e = self.evol_eps_e(phi,Tm,sig_e,dot_eps)

        ddot_eps = (dot_eps-eps_dot_n1)/dt
        
        dot_sigma_el = TC.evol_sigmaE(sigma_el, phi, dot_phi, epsilon_el, dot_epsilon_el) # rate of the elastic stress
        dot_sigmaD = TC.evol_sigmaD(Tm, dot_Tm, dot_eps_update, ddot_epsilon) # rate of the viscous stress
        dot_pT = self.evol_pT(Tm, dot_Tm)
        
        dot_sig = dot_sig_e+dot_sigmaD
        dot_sig[:3] += dot_pT
        
        return np.hstack((dot_phi,dot_eps_e,dot_Tm,dot_sig_e,dot_eps,dot_sig))
    
    def solve(self,state_t,deps,dt,eps_dot_n1):
        """
        Solves the evolution of the state variables over a time step using ODE integration.
    
        Parameters:
            state_t (np.ndarray): Current state variables [phi, eps_e, Tm, sig_e, eps, sig].
            deps (np.ndarray): Incremental strain in Voigt notation.
            dt (float): Time step size.
            eps_dot_n1 (np.ndarray): Previous strain rate in Voigt notation.
    
        Returns:
            np.ndarray: Updated state variables after time step `dt`.
        """
        eps_dot = deps/dt
        sol = solve_ivp(self.evol_state_eps_driven,
                    args=[eps_dot,eps_dot_n1,dt],
                        t_span=[0.,dt],
                        t_eval=[dt],
                        y0=state_t,
                        vectorized=False,
                        atol = 1.e-12,
                        rtol=1e-11,
                        method='LSODA')
        return sol.y[:,-1]