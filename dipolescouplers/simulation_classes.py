#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# simulation_classes.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import special
import random

random.seed(42)

class DipoleSimulation:
    def __init__(self, Nb_particule=100, W=3, L=1, k_in=25, N_x=100, N_y=300):
        self.Nb_particule = Nb_particule  # nombre de dipôles
        self.W = W                        # largeur du milieu
        self.L = L                        # longueur du milieu
        self.k_in = k_in                  # nombre d'onde incident
        self.N_x = N_x                    # résolution de la grille pour les particules
        self.N_y = N_y
        # ici, alpha est défini selon votre code
        self.alpha = 4 * 1j / (k_in**2)
        self.pas_x = L / N_x
        self.pas_y = W / N_y
        
        # Génère la configuration initiale des dipôles
        self.X_random, self.Y_random = self.creation_grille_avec_points_aleat()
        # Construit la matrice d'interaction A entre dipôles
        self.A = self.compute_interaction_matrix()
        
    def creation_grille_avec_points_aleat(self):
        """
        Création d'une grille sur [0,L]x[0,W] et sélection aléatoire de Nb_particule positions.
        """
        x = np.linspace(0, self.L, self.N_x)
        y = np.linspace(0, self.W, self.N_y)
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        total_points = len(X_flat)
        indices = np.random.choice(total_points, self.Nb_particule, replace=False)
        X_random = X_flat[indices]
        Y_random = Y_flat[indices]
        return X_random, Y_random

    def green_function(self, r, r_prime, k0, eps=1e-9):
        R = np.linalg.norm(np.array(r) - np.array(r_prime))
        if R < eps:
            return 0
        return 1j * special.hankel1(0, k0 * R) / 4

    def champ_incident(self, x, y, thet):
        """
        Retourne le champ incident pour une onde plane d'angle d'incidence thet.
        """
        return np.exp(1j * self.k_in * (np.cos(thet)*x + np.sin(thet)*y))
    
    def compute_interaction_matrix(self):
        """
        Construit la matrice A d'interaction entre dipôles.
        """
        A = np.zeros((self.Nb_particule, self.Nb_particule), dtype=complex)
        for j in range(self.Nb_particule):
            for k in range(self.Nb_particule):
                if j == k:
                    A[j, k] = 0
                else:
                    A[j, k] = self.k_in**2 * self.alpha * self.green_function(
                        (self.X_random[j], self.Y_random[j]),
                        (self.X_random[k], self.Y_random[k]),
                        self.k_in)
        return A
    
    def compute_field(self, thet, N_obs_x=100, N_obs_y=30):
        """
        Calcule le champ total dans le milieu pour une incidence d'angle thet.
        Retourne (x_obs, y_obs, E_total) où E_total est la carte du champ.
        """
        # Calcul du champ incident sur chaque dipôle
        E0 = np.array([self.champ_incident(x, y, thet) 
                       for x, y in zip(self.X_random, self.Y_random)])
        I_mat = np.eye(self.Nb_particule, dtype=complex)
        M = I_mat - self.A
        E = np.linalg.solve(M, E0)
        
        # Définition de la grille d'observation (pour couvrir la zone réfléchie et transmise)
        x_obs = np.linspace(-self.L, 2*self.L, N_obs_x)
        y_obs = np.linspace(-self.L, self.W+self.L, N_obs_y)
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs)
        E_total = np.zeros_like(X_obs, dtype=complex)
        
        # Calcul du champ total point par point
        for i in range(N_obs_y):
            for j in range(N_obs_x):
                r_obs = (X_obs[i, j], Y_obs[i, j])
                E0_r = self.champ_incident(r_obs[0], r_obs[1], thet)
                somme = 0.0 + 0.0j
                for idx in range(self.Nb_particule):
                    r_part = (self.X_random[idx], self.Y_random[idx])
                    somme += self.green_function(r_obs, r_part, self.k_in) * E[idx]
                E_total[i, j] = E0_r + self.k_in**2 * self.alpha * somme
        return x_obs, y_obs, E_total

    def build_TM(self, theta_vals, N_obs_x=100, N_obs_y=30):
        """
        Pour chaque angle dans theta_vals, extrait le champ en sortie (à x=2L) 
        et construit la matrice de transmission pré-transposée.
        """
        TM_pretransposee = []
        for idx, th in enumerate(theta_vals):
            progress = (idx+1) / len(theta_vals) * 100
            print(f"Traitement de theta {idx+1}/{len(theta_vals)} : {progress:.1f}% terminé")
            x_obs, _, E_total = self.compute_field(th, N_obs_x, N_obs_y)
            # Extraction de la colonne à x = 2L
            indx = np.argmin(np.abs(x_obs - 2*self.L))
            field_line = E_total[:, indx]
            TM_pretransposee.append(field_line)
        return np.array(TM_pretransposee)  # forme (N_input, N_obs_y)


class PhaseConjugation:
    def __init__(self, TM_pretransposee):
        """
        TM_pretransposee : matrice de transmission pré-transposée de forme (N_input, N_obs_y)
        """
        self.TM_pretransposee = TM_pretransposee
        # Calcul de l'adjoint (conjugué-transposé) de TM_pretransposee
        self.TM_dagger = TM_pretransposee.conj()  # forme (N_obs_y, N_input)
        # Pour la propagation, nous définissons la TM dans la convention (N_obs_y, N_input)
        self.TM = TM_pretransposee.T
        print("TM shape est" , self.TM.shape)

    def compute_input_field(self, desired_output):
        """
        Calcule le champ d'entrée optimal via la multiplication par l'adjoint de la TM.
        desired_output : vecteur cible de taille (N_obs_y,)
        Retourne input_field de taille (N_input,)
        """
        return np.dot(self.TM_dagger, desired_output)

    def phase_only(self, input_field):
        """
        Conserve uniquement la phase du champ d'entrée.
        """
        return np.exp(1j * np.angle(input_field))

    def compute_focused_output(self, phase_input):
        """
        Propagation du champ d'entrée modifié par la TM pour obtenir le champ de sortie focalisé.
        """
        return np.dot(self.TM, phase_input)

