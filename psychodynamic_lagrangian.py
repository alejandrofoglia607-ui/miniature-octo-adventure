"""Lagrangiano y formulación variacional completa con SymPy.

Este módulo implementa una versión simbólica consistente del sistema
psicodinámico cuántico mostrado por el usuario.
"""

import sympy as sp
from sympy import Derivative, Function, Matrix, integrate, symbols


# Símbolos globales
T = symbols("t", real=True)
X, Y, Z = symbols("x y z", real=True)
HBAR, C, M = symbols("hbar c m", positive=True)

# Parámetros del modelo
R1, R2, K1, K2 = symbols("r1 r2 K1 K2", positive=True)
BETA1, BETA2 = symbols("beta1 beta2", real=True)
GAMMA1, GAMMA2 = symbols("gamma1 gamma2", real=True)
D_PHI, ETA_PHI, THETA_PHI, XI_PHI = symbols(
    "D_phi eta_phi theta_phi xi_phi", positive=True
)


class PsychodynamicLagrangian:
    """Lagrangiano simbólico del sistema psicodinámico-cuántico."""

    def __init__(self):
        # Campos escalares dependientes de tiempo
        self.a = Function("a")(T)
        self.b = Function("b")(T)

        # Espinor cuántico mínimo de 2 componentes
        self.psi1 = Function("psi_1")(T)
        self.psi2 = Function("psi_2")(T)
        self.psi = Matrix([self.psi1, self.psi2])
        self.psi_dagger = Matrix([[sp.conjugate(self.psi1), sp.conjugate(self.psi2)]])

        # Campo de materialización en 3+1
        self.phi = Function("phi")(X, Y, Z, T)

    def lagrangian_density_1d(self):
        """Densidad lagrangiana en 1+1 (t, x)."""
        a_dot = Derivative(self.a, T)
        b_dot = Derivative(self.b, T)
        psi_dot = Matrix([Derivative(self.psi1, T), Derivative(self.psi2, T)])

        phi_t = Derivative(self.phi, T)
        phi_x = Derivative(self.phi, X)

        kinetic_actions = sp.Rational(1, 2) * (a_dot**2 + b_dot**2)
        quantum_term = sp.I * HBAR * (self.psi_dagger * psi_dot)[0]
        kinetic_phi = sp.Rational(1, 2) * (phi_t**2 - D_PHI * phi_x**2)

        potential_actions = (
            -R1 * self.a**2 * (1 - self.a / (3 * K1))
            - R2 * self.b**2 * (1 - self.b / (3 * K2))
        )
        interaction_term = -BETA1 * self.a * self.b**2 - BETA2 * self.a**2 * self.b

        sigma_z = Matrix([[1, 0], [0, -1]])
        observable = (self.psi_dagger * sigma_z * self.psi)[0]
        coupling_term = -GAMMA1 * observable * self.a

        potential_phi = (
            -sp.Rational(1, 2)
            * ETA_PHI
            * (BETA1 * self.a * self.b**2 - THETA_PHI)
            * self.phi**2
        )

        return (
            kinetic_actions
            + quantum_term
            + kinetic_phi
            + potential_actions
            + interaction_term
            + coupling_term
            + potential_phi
        )

    @staticmethod
    def action_functional(lagrangian, domain="1+1"):
        """Funcional S = ∫L d^n x para dominios 1+1 o 3+1."""
        t_i, t_f = symbols("t_i t_f", real=True)
        x_min, x_max = symbols("x_min x_max", real=True)

        if domain == "1+1":
            return integrate(lagrangian, (T, t_i, t_f), (X, x_min, x_max))

        if domain == "3+1":
            y_min, y_max = symbols("y_min y_max", real=True)
            z_min, z_max = symbols("z_min z_max", real=True)
            return integrate(
                lagrangian,
                (T, t_i, t_f),
                (X, x_min, x_max),
                (Y, y_min, y_max),
                (Z, z_min, z_max),
            )

        raise ValueError("domain debe ser '1+1' o '3+1'")

    def euler_lagrange_equations(self, lagrangian):
        """Deriva ecuaciones de Euler-Lagrange para a, b y phi(t,x)."""
        a_dot = Derivative(self.a, T)
        b_dot = Derivative(self.b, T)
        phi_t = Derivative(self.phi, T)
        phi_x = Derivative(self.phi, X)

        equations = {
            "a": sp.simplify(sp.diff(lagrangian, self.a) - sp.diff(sp.diff(lagrangian, a_dot), T)),
            "b": sp.simplify(sp.diff(lagrangian, self.b) - sp.diff(sp.diff(lagrangian, b_dot), T)),
            "phi": sp.simplify(
                sp.diff(lagrangian, self.phi)
                - sp.diff(sp.diff(lagrangian, phi_t), T)
                - sp.diff(sp.diff(lagrangian, phi_x), X)
            ),
        }

        # Para el espinor complejo: variación independiente de psi_i y psi_i*.
        psi_dot = Matrix([Derivative(self.psi1, T), Derivative(self.psi2, T)])
        equations["psi"] = [
            sp.simplify(sp.diff(lagrangian, self.psi[i]) - sp.diff(sp.diff(lagrangian, psi_dot[i]), T))
            for i in range(2)
        ]
        equations["psi_dagger"] = [
            sp.simplify(sp.diff(lagrangian, sp.conjugate(self.psi[i]))) for i in range(2)
        ]

        return equations


if __name__ == "__main__":
    model = PsychodynamicLagrangian()
    L = model.lagrangian_density_1d()
    eqs = model.euler_lagrange_equations(L)

    print("Lagrangiano (1+1):")
    print(L)
    print("\nEcuaciones de Euler-Lagrange disponibles:")
    for key in eqs:
        print(f"- {key}")
