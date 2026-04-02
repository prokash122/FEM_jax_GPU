"""
3D Piezoelectric FEA Solver using JAX
======================================
Replicates Abaqus BaTiO3 piezoelectric model:
  - 3D hex8 elements (C3D8E equivalent)
  - Coupled electromechanical formulation (e-form)
  - Pressure load on top surface
  - Electric potential (EPOT) output

Material: BaTiO3 single crystal (tetragonal, polarization along Z)
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

jax.config.update("jax_enable_x64", True)

# ============================================================
# 1. MATERIAL PROPERTIES (BaTiO3 - SI units)
# ============================================================

# Elastic stiffness at constant E-field [Pa]
# Voigt: C11,C12,C13,C33,C44,C66
C11 = 275.1e9;  C12 = 178.9e9;  C13 = 151.6e9
C33 = 164.8e9;  C44 = 54.3e9;   C66 = 113.1e9

cE = jnp.array([
    [C11, C12, C13, 0,   0,   0  ],
    [C12, C11, C13, 0,   0,   0  ],
    [C13, C13, C33, 0,   0,   0  ],
    [0,   0,   0,   C44, 0,   0  ],
    [0,   0,   0,   0,   C44, 0  ],
    [0,   0,   0,   0,   0,   C66],
])

# Piezoelectric stress constants [C/m^2]
# e-form: 3x6 matrix  (row = electric direction, col = Voigt strain)
e15 = 21.3;  e31 = -0.7;  e33 = 6.7

e_piezo = jnp.array([
    [0,   0,   0,   0,   e15, 0  ],
    [0,   0,   0,   e15, 0,   0  ],
    [e31, e31, e33, 0,   0,   0  ],
])

# Dielectric permittivity at constant strain [F/m]
eps0 = 8.854187817e-12
eps11_S = 2920.0 * eps0
eps33_S = 168.0 * eps0

eps_S = jnp.array([
    [eps11_S, 0,       0      ],
    [0,       eps11_S, 0      ],
    [0,       0,       eps33_S],
])

# Density [kg/m^3]
rho = 6020.0

print("=" * 60)
print("  3D Piezoelectric FEA Solver (JAX)")
print("  Material: BaTiO3 (tetragonal)")
print("=" * 60)

# ============================================================
# 2. MESH GENERATION (Rectangular block with hex8 elements)
# ============================================================

# Geometry [m]
Lx = 0.06   # 60 mm
Ly = 0.06   # 60 mm
Lz = 0.01   # 10 mm (thickness along polarization)

# Mesh divisions
nx, ny, nz = 12, 12, 4

print(f"\nGeometry: {Lx*1e3:.1f} x {Ly*1e3:.1f} x {Lz*1e3:.1f} mm")
print(f"Mesh: {nx} x {ny} x {nz} = {nx*ny*nz} elements")

# Node coordinates
n_nodes = (nx+1) * (ny+1) * (nz+1)
n_elems = nx * ny * nz
n_dof_per_node = 4  # u1, u2, u3, phi
n_total_dof = n_nodes * n_dof_per_node

coords = np.zeros((n_nodes, 3))
node_id = 0
for k in range(nz+1):
    for j in range(ny+1):
        for i in range(nx+1):
            coords[node_id] = [i*Lx/nx, j*Ly/ny, k*Lz/nz]
            node_id += 1

def node_index(i, j, k):
    return k*(ny+1)*(nx+1) + j*(nx+1) + i

# Element connectivity (hex8)
connectivity = np.zeros((n_elems, 8), dtype=int)
elem_id = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            n0 = node_index(i, j, k)
            n1 = node_index(i+1, j, k)
            n2 = node_index(i+1, j+1, k)
            n3 = node_index(i, j+1, k)
            n4 = node_index(i, j, k+1)
            n5 = node_index(i+1, j, k+1)
            n6 = node_index(i+1, j+1, k+1)
            n7 = node_index(i, j+1, k+1)
            connectivity[elem_id] = [n0, n1, n2, n3, n4, n5, n6, n7]
            elem_id += 1

coords_jax = jnp.array(coords)
conn_jax = jnp.array(connectivity)

print(f"Nodes: {n_nodes}, DOFs: {n_total_dof}")

# ============================================================
# 3. HEX8 ELEMENT FORMULATION
# ============================================================

# 2x2x2 Gauss quadrature
gp = 1.0 / jnp.sqrt(3.0)
gauss_pts = jnp.array([
    [-gp, -gp, -gp], [+gp, -gp, -gp],
    [+gp, +gp, -gp], [-gp, +gp, -gp],
    [-gp, -gp, +gp], [+gp, -gp, +gp],
    [+gp, +gp, +gp], [-gp, +gp, +gp],
])
gauss_wts = jnp.ones(8)


def shape_functions(xi, eta, zeta):
    """Hex8 shape functions and derivatives in natural coords."""
    N = 0.125 * jnp.array([
        (1-xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1-zeta),
        (1+xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1-zeta),
        (1-xi)*(1-eta)*(1+zeta), (1+xi)*(1-eta)*(1+zeta),
        (1+xi)*(1+eta)*(1+zeta), (1-xi)*(1+eta)*(1+zeta),
    ])
    dN = 0.125 * jnp.array([
        [-(1-eta)*(1-zeta), +(1-eta)*(1-zeta), +(1+eta)*(1-zeta), -(1+eta)*(1-zeta),
         -(1-eta)*(1+zeta), +(1-eta)*(1+zeta), +(1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
        [-(1-xi)*(1-zeta), -(1+xi)*(1-zeta), +(1+xi)*(1-zeta), +(1-xi)*(1-zeta),
         -(1-xi)*(1+zeta), -(1+xi)*(1+zeta), +(1+xi)*(1+zeta), +(1-xi)*(1+zeta)],
        [-(1-xi)*(1-eta), -(1+xi)*(1-eta), -(1+xi)*(1+eta), -(1-xi)*(1+eta),
         +(1-xi)*(1-eta), +(1+xi)*(1-eta), +(1+xi)*(1+eta), +(1-xi)*(1+eta)],
    ])
    return N, dN


def compute_element_matrices(elem_coords):
    """Compute element stiffness matrix for piezoelectric hex8."""
    Ke = jnp.zeros((32, 32))

    def gauss_loop(carry, gp_idx):
        Ke = carry
        xi, eta, zeta = gauss_pts[gp_idx]
        w = gauss_wts[gp_idx]
        N, dN_nat = shape_functions(xi, eta, zeta)

        # Jacobian
        J = dN_nat @ elem_coords  # 3x3
        detJ = jnp.linalg.det(J)
        invJ = jnp.linalg.inv(J)
        dN = invJ @ dN_nat  # 3x8 (derivatives in physical coords)

        # Strain-displacement matrix Bu (6x24)
        Bu = jnp.zeros((6, 24))
        for a in range(8):
            col = a * 3
            Bu = Bu.at[0, col+0].set(dN[0, a])  # eps_11
            Bu = Bu.at[1, col+1].set(dN[1, a])  # eps_22
            Bu = Bu.at[2, col+2].set(dN[2, a])  # eps_33
            Bu = Bu.at[3, col+1].set(dN[2, a])  # 2*eps_23
            Bu = Bu.at[3, col+2].set(dN[1, a])
            Bu = Bu.at[4, col+0].set(dN[2, a])  # 2*eps_13
            Bu = Bu.at[4, col+2].set(dN[0, a])
            Bu = Bu.at[5, col+0].set(dN[1, a])  # 2*eps_12
            Bu = Bu.at[5, col+1].set(dN[0, a])

        # Electric field-potential matrix Bphi (3x8)
        Bphi = -dN  # E = -grad(phi), so Bphi = -dN

        # Sub-matrices
        Kuu = Bu.T @ cE @ Bu * detJ * w               # 24x24
        Kuphi = Bu.T @ e_piezo.T @ Bphi * detJ * w    # 24x8
        Kphiphi = -Bphi.T @ eps_S @ Bphi * detJ * w   # 8x8

        # Assemble into 32x32 element matrix
        # DOF order: [u1_1,u2_1,u3_1, u1_2,..., u1_8,...,u3_8, phi_1,...,phi_8]
        # i.e., first 24 = displacement, last 8 = potential
        Ke = Ke.at[:24, :24].add(Kuu)
        Ke = Ke.at[:24, 24:].add(Kuphi)
        Ke = Ke.at[24:, :24].add(Kuphi.T)
        Ke = Ke.at[24:, 24:].add(Kphiphi)

        return Ke, None

    Ke, _ = jax.lax.scan(gauss_loop, Ke, jnp.arange(8))
    return Ke


compute_element_matrices_jit = jit(compute_element_matrices)

# ============================================================
# 4. GLOBAL ASSEMBLY
# ============================================================

print("\nAssembling global system...")
t0 = time.time()

# DOF mapping: node n -> global DOFs [4n, 4n+1, 4n+2, 4n+3] = [u1, u2, u3, phi]
K_global = np.zeros((n_total_dof, n_total_dof))
F_global = np.zeros(n_total_dof)

for e in range(n_elems):
    nodes = connectivity[e]
    elem_coords = coords_jax[nodes]

    Ke = np.array(compute_element_matrices_jit(elem_coords))

    # Build global DOF indices for this element
    # Element DOF order: u1_0,u2_0,u3_0, u1_1,u2_1,u3_1,..., phi_0,...,phi_7
    dof_u = []
    dof_phi = []
    for local_n in range(8):
        gn = nodes[local_n]
        dof_u.extend([4*gn, 4*gn+1, 4*gn+2])
        dof_phi.append(4*gn+3)

    elem_dofs = dof_u + dof_phi  # 32 DOFs total

    for i_local in range(32):
        for j_local in range(32):
            K_global[elem_dofs[i_local], elem_dofs[j_local]] += Ke[i_local, j_local]

    if (e+1) % 200 == 0 or e == n_elems-1:
        print(f"  Elements assembled: {e+1}/{n_elems}")

t_assemble = time.time() - t0
print(f"Assembly time: {t_assemble:.2f} s")

# ============================================================
# 5. BOUNDARY CONDITIONS & LOADS
# ============================================================

print("\nApplying BCs and loads...")

# Identify node sets
tol = 1e-10
bottom_nodes = np.where(coords[:, 2] < tol)[0]
top_nodes = np.where(coords[:, 2] > Lz - tol)[0]

# Fixed corner node to prevent rigid body motion
corner_node = node_index(0, 0, 0)

# --- Mechanical BCs: simply supported bottom ---
bc_dofs = []
bc_vals = []

# Fix Z-displacement on bottom face
for n in bottom_nodes:
    bc_dofs.append(4*n + 2)  # u3 = 0
    bc_vals.append(0.0)

# Fix one corner to prevent rigid body translation in X, Y
bc_dofs.append(4*corner_node + 0)  # u1 = 0
bc_vals.append(0.0)
bc_dofs.append(4*corner_node + 1)  # u2 = 0
bc_vals.append(0.0)

# Fix another node in Y to prevent rotation
corner2 = node_index(nx, 0, 0)
bc_dofs.append(4*corner2 + 1)  # u2 = 0
bc_vals.append(0.0)

# --- Electrical BC: ground bottom surface (phi = 0) ---
for n in bottom_nodes:
    bc_dofs.append(4*n + 3)  # phi = 0
    bc_vals.append(0.0)

bc_dofs = np.array(bc_dofs)
bc_vals = np.array(bc_vals)

# --- Pressure load on top surface (200 kPa in -Z) ---
pressure = 200000.0  # Pa

# Compute consistent nodal forces from pressure on top face
# For each element face on top surface, distribute pressure
for e in range(n_elems):
    nodes_e = connectivity[e]
    z_top_mask = coords[nodes_e, 2] > Lz - tol
    # Top face of hex: nodes 4,5,6,7
    top_face_nodes = nodes_e[4:8]

    if np.all(coords[top_face_nodes, 2] > Lz - tol):
        # Compute face area (approximate for regular mesh)
        dx = Lx / nx
        dy = Ly / ny
        face_area = dx * dy
        # Distribute pressure equally to 4 face nodes
        force_per_node = -pressure * face_area / 4.0  # negative Z
        for fn in top_face_nodes:
            F_global[4*fn + 2] += force_per_node

print(f"  Bottom nodes (grounded): {len(bottom_nodes)}")
print(f"  Top nodes (loaded): {len(top_nodes)}")
print(f"  Total force: {np.sum(F_global[2::4]):.4f} N")

# ============================================================
# 6. SOLVE
# ============================================================

print("\nSolving coupled system...")
t0 = time.time()

# Apply BCs using penalty method
penalty = 1e30
for i, dof in enumerate(bc_dofs):
    K_global[dof, dof] += penalty
    F_global[dof] += penalty * bc_vals[i]

# Solve with JAX
K_jax = jnp.array(K_global)
F_jax = jnp.array(F_global)

solution = jnp.linalg.solve(K_jax, F_jax)
solution = np.array(solution)

t_solve = time.time() - t0
print(f"Solve time: {t_solve:.2f} s")

# Extract results
u_disp = np.zeros((n_nodes, 3))
phi = np.zeros(n_nodes)

for n in range(n_nodes):
    u_disp[n, 0] = solution[4*n]
    u_disp[n, 1] = solution[4*n+1]
    u_disp[n, 2] = solution[4*n+2]
    phi[n] = solution[4*n+3]

# ============================================================
# 7. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\n  Max displacement (Z): {np.max(u_disp[:,2])*1e6:.4f} um")
print(f"  Min displacement (Z): {np.min(u_disp[:,2])*1e6:.4f} um")
print(f"\n  Max electric potential: {np.max(phi):.4f} V")
print(f"  Min electric potential: {np.min(phi):.4f} V")
print(f"  Potential difference:   {np.max(phi) - np.min(phi):.4f} V")

# Potential at top surface
phi_top = phi[top_nodes]
phi_bot = phi[bottom_nodes]
print(f"\n  Avg potential (top):    {np.mean(phi_top):.4f} V")
print(f"  Avg potential (bottom): {np.mean(phi_bot):.6f} V (grounded)")

# ============================================================
# 8. VISUALIZATION
# ============================================================

print("\nGenerating plots...")

fig = plt.figure(figsize=(18, 6))

# --- Plot 1: EPOT contour (3D) ---
ax1 = fig.add_subplot(131, projection='3d')
sc1 = ax1.scatter(coords[:, 0]*1e3, coords[:, 1]*1e3, coords[:, 2]*1e3,
                  c=phi, cmap='jet', s=8, edgecolors='none')
ax1.set_xlabel('X [mm]')
ax1.set_ylabel('Y [mm]')
ax1.set_zlabel('Z [mm]')
ax1.set_title('Electric Potential (EPOT) [V]')
plt.colorbar(sc1, ax=ax1, shrink=0.6, label='Potential [V]')

# --- Plot 2: EPOT through thickness ---
ax2 = fig.add_subplot(132)
center_col = []
for k in range(nz+1):
    nid = node_index(nx//2, ny//2, k)
    center_col.append((coords[nid, 2]*1e3, phi[nid]))
center_col = np.array(center_col)
ax2.plot(center_col[:, 1], center_col[:, 0], 'b-o', linewidth=2, markersize=6)
ax2.set_xlabel('Electric Potential [V]')
ax2.set_ylabel('Z position [mm]')
ax2.set_title('EPOT through thickness\n(center of plate)')
ax2.grid(True, alpha=0.3)

# --- Plot 3: Z-displacement contour ---
ax3 = fig.add_subplot(133, projection='3d')
sc3 = ax3.scatter(coords[:, 0]*1e3, coords[:, 1]*1e3, coords[:, 2]*1e3,
                  c=u_disp[:, 2]*1e6, cmap='coolwarm', s=8, edgecolors='none')
ax3.set_xlabel('X [mm]')
ax3.set_ylabel('Y [mm]')
ax3.set_zlabel('Z [mm]')
ax3.set_title('Z-Displacement [μm]')
plt.colorbar(sc3, ax=ax3, shrink=0.6, label='Uz [μm]')

plt.tight_layout()
plt.savefig('piezo_results.png', dpi=150, bbox_inches='tight')
print("  Saved: piezo_results.png")

# --- Plot 4: Comparison with Abaqus ---
fig2, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(center_col[:, 1], center_col[:, 0], 'b-o', linewidth=2.5,
         markersize=8, label='JAX FEA')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(y=Lz*1e3, color='gray', linestyle='--', alpha=0.5)
ax4.annotate('Bottom (grounded)', xy=(0, 0), fontsize=9, color='gray')
ax4.annotate('Top (loaded)', xy=(0, Lz*1e3), fontsize=9, color='gray')
ax4.set_xlabel('Electric Potential [V]', fontsize=12)
ax4.set_ylabel('Z position [mm]', fontsize=12)
ax4.set_title('BaTiO₃ Piezoelectric Response\n200 kPa pressure → Electric potential', fontsize=13)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('piezo_epot_profile.png', dpi=150, bbox_inches='tight')
print("  Saved: piezo_epot_profile.png")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
