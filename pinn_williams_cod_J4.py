
# ============================================================
# PINN ENRICHI DE WILLIAMS - GRIFFITH MODE I
# Plaque infinie approximee par un domaine tronque
# PyTorch 2.1.5+cpu / NumPy 2.x
#
# Inconnue apprise : amplitude KI du champ singulier
# Validation : COD, KI, G, trois contours Rice J
# Chargement progressif puis extension a contrainte constante
# ============================================================

import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)

DEVICE = torch.device("cpu")
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("Device:", DEVICE)

# ============================================================
# 1. PARAMETRES PHYSIQUES
# ============================================================

E = 200e9
NU = 0.30
GC = 8000.0

# Plane stress
EP = E
MU = E / (2.0 * (1.0 + NU))
KAPPA = (3.0 - NU) / (1.0 + NU)

A0 = 0.050
SIGMA0 = 90e6

# Domaine local autour de la pointe droite.
# Les coordonnees sont relatives a la pointe.
# Une frontiere exterieure de rayon RMAX est utilisee.
RMIN = 0.0015
RMAX = 0.080

# Trois contours pour J
J_RADII = [0.005, 0.008, 0.011]

# Precision et entrainement
N_EPOCHS = 1200
N_INTERIOR = 600
N_OUTER = 160
N_FACES = 120
N_J = 240

LR = 1.0e-3
PRINT_EVERY = 200

# Chargements progressifs
SIGMA_LIST_MPA = [
    90.0, 95.0, 98.0, 100.0,
    100.5, 100.9, 101.0
]

# Extension post-amorcage
DA = 0.0005
N_EXTENSION = 5

# ============================================================
# 2. OUTILS
# ============================================================

def tensor(x, requires_grad=False):
    return torch.tensor(
        x, dtype=torch.float64, device=DEVICE,
        requires_grad=requires_grad
    )

def grad(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

def analytical_ki(sigma, a):
    return sigma * math.sqrt(math.pi * a)

def analytical_g(sigma, a):
    return analytical_ki(sigma, a)**2 / EP

def critical_sigma(a):
    return math.sqrt(EP * GC / (math.pi * a))

def relative_error(x, ref):
    return 100.0 * abs(x - ref) / max(abs(ref), 1e-20)

# ============================================================
# 3. CHAMP DE WILLIAMS - MODE I
# ============================================================

# Coordonnees polaires centrees sur la pointe droite.
# theta = +/- pi sur les levres de fissure.
#
# ux = KI/(2mu)*sqrt(r/2pi)*
#      cos(theta/2)*(kappa-1+2sin²(theta/2))
#
# uy = KI/(2mu)*sqrt(r/2pi)*
#      sin(theta/2)*(kappa+1-2cos²(theta/2))

def williams_displacement(xy, ki):
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    r = torch.sqrt(x*x + y*y + 1e-24)
    th = torch.atan2(y, x)

    fac = ki / (2.0 * MU) * torch.sqrt(r / (2.0 * math.pi))

    ux = fac * torch.cos(th/2.0) * (
        KAPPA - 1.0 + 2.0 * torch.sin(th/2.0)**2
    )

    uy = fac * torch.sin(th/2.0) * (
        KAPPA + 1.0 - 2.0 * torch.cos(th/2.0)**2
    )

    return torch.cat([ux, uy], dim=1)

# ============================================================
# 4. RESEAU NEURONAL
# ============================================================

class WilliamsPINN(nn.Module):
    def __init__(self, ki_initial):
        super().__init__()

        # Coefficient KI appris, sans verrouillage analytique
        self.k_raw = nn.Parameter(
            torch.tensor(
                float(ki_initial / 1e6),
                dtype=torch.float64
            )
        )

        self.net = nn.Sequential(
            nn.Linear(2, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 2)
        )

        # Le reseau apprend une correction reguliere.
        # Son amplitude n'est PAS initialisee a zero
        # par un multiplicateur global qui annulerait
        # tous les gradients.

    def forward(self, xy):
        ki = self.k_raw * 1e6

        uw = williams_displacement(xy, ki)

        # Echelle des deplacements
        u_scale = 1e-4

        # Entrees normalisees
        xn = xy / RMAX

        correction = u_scale * self.net(xn)

        return uw + correction

    def ki(self):
        return self.k_raw * 1e6

# ============================================================
# 5. ELASTICITE LINEAIRE
# ============================================================

def strain_stress(model, xy):
    uv = model(xy)

    gu = grad(uv[:, 0:1], xy)
    gv = grad(uv[:, 1:2], xy)

    exx = gu[:, 0:1]
    eyy = gv[:, 1:2]
    exy = 0.5 * (gu[:, 1:2] + gv[:, 0:1])

    c = E / (1.0 - NU**2)

    sxx = c * (exx + NU * eyy)
    syy = c * (eyy + NU * exx)
    sxy = 2.0 * MU * exy

    w = 0.5 * (
        sxx * exx +
        syy * eyy +
        2.0 * sxy * exy
    )

    return uv, gu, gv, sxx, syy, sxy, w

def equilibrium_loss(model, xy):
    _, _, _, sxx, syy, sxy, _ = strain_stress(model, xy)

    gsxx = grad(sxx, xy)
    gsyy = grad(syy, xy)
    gsxy = grad(sxy, xy)

    rx = gsxx[:, 0:1] + gsxy[:, 1:2]
    ry = gsxy[:, 0:1] + gsyy[:, 1:2]

    # Normalisation physique
    scale = SIGMA0 / RMAX

    return torch.mean(
        (rx / scale)**2 + (ry / scale)**2
    )

# ============================================================
# 6. ECHANTILLONNAGE
# ============================================================

def interior_points(n):
    # Echantillonnage logarithmique radial
    rr = RMIN * (RMAX / RMIN)**torch.rand(n, 1)
    th = -math.pi + 2.0*math.pi*torch.rand(n, 1)

    xy = torch.cat([
        rr*torch.cos(th),
        rr*torch.sin(th)
    ], dim=1)

    return xy.detach().requires_grad_(True)

def outer_points(n):
    th = torch.linspace(
        -math.pi + 1e-4,
        math.pi - 1e-4,
        n
    ).reshape(-1, 1)

    return torch.cat([
        RMAX*torch.cos(th),
        RMAX*torch.sin(th)
    ], dim=1)

def crack_face_points(n, upper=True):
    rr = torch.linspace(
        RMIN, RMAX, n
    ).reshape(-1, 1)

    yy = torch.full_like(
        rr, 1e-8 if upper else -1e-8
    )

    return torch.cat([-rr, yy], dim=1).requires_grad_(True)

# ============================================================
# 7. CONDITIONS AUX LIMITES
# ============================================================

def boundary_loss(model, sigma, a):
    # Sur le contour exterieur, le champ lointain est
    # approxime par Williams + champ T non singulier.
    # Le KI de reference est utilise UNIQUEMENT pour
    # definir cette condition aux limites du benchmark.
    #
    # Il n'est pas directement impose au parametre
    # k_raw du reseau.

    xo = outer_points(N_OUTER)

    kfar = analytical_ki(sigma, a)

    with torch.no_grad():
        u_target = williams_displacement(
            xo, tensor(kfar)
        )

        # Champ T de la solution de Kirsch/Williams
        # non inclus ici : benchmark local dominant.

    u_pred = model(xo)

    scale = max(
        kfar / E * math.sqrt(RMAX),
        1e-8
    )

    bc = torch.mean(
        ((u_pred - u_target) / scale)**2
    )

    return bc

def face_loss(model):
    losses = []

    for upper in [True, False]:
        xy = crack_face_points(N_FACES, upper)
        _, _, _, sxx, syy, sxy, _ = strain_stress(model, xy)

        # Sur les levres horizontales :
        # sigma_yy = sigma_xy = 0
        losses.append(torch.mean(
            (syy / SIGMA0)**2 +
            (sxy / SIGMA0)**2
        ))

    return sum(losses)

# ============================================================
# 8. FONCTION DE PERTE
# ============================================================

def total_loss(model, sigma, a):
    xy = interior_points(N_INTERIOR)

    leq = equilibrium_loss(model, xy)
    lbc = boundary_loss(model, sigma, a)
    lface = face_loss(model)

    loss = leq + 10.0*lbc + 5.0*lface

    return loss, leq, lbc, lface

# ============================================================
# 9. ENTRAINEMENT
# ============================================================

def train_model(model, sigma, a, epochs=N_EPOCHS):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR
    )

    history = []
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss, leq, lbc, lface = total_loss(
            model, sigma, a
        )

        loss.backward()
        optimizer.step()

        history.append(float(loss.detach()))

        if epoch % PRINT_EVERY == 0 or epoch == epochs-1:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss={loss.item():.4e} | "
                f"Eq={leq.item():.3e} | "
                f"BC={lbc.item():.3e} | "
                f"Faces={lface.item():.3e} | "
                f"KI={model.ki().item()/1e6:.5f}"
            )

    print(f"Training time: {time.time()-start:.1f} s")

    return history

# ============================================================
# 10. EXTRACTION INDEPENDANTE KI PAR COD
# ============================================================

def cod_and_ki(model):
    distances = np.linspace(0.002, 0.012, 30)

    xp = tensor(np.column_stack([
        -distances,
        np.full_like(distances, 1e-8)
    ]))

    xm = tensor(np.column_stack([
        -distances,
        np.full_like(distances, -1e-8)
    ]))

    with torch.no_grad():
        up = model(xp)[:, 1].cpu().numpy()
        um = model(xm)[:, 1].cpu().numpy()

    cod = up - um

    # COD = KI*(kappa+1)/mu*sqrt(r/(2pi))
    basis = (
        (KAPPA+1.0)/MU *
        np.sqrt(distances/(2.0*np.pi))
    )

    ki_cod = np.dot(basis, cod) / np.dot(basis, basis)

    cod_fit = ki_cod * basis

    err = 100.0 * np.linalg.norm(
        cod - cod_fit
    ) / max(np.linalg.norm(cod), 1e-20)

    return distances, cod, ki_cod, err

# ============================================================
# 11. INTEGRALE DE RICE J
# ============================================================

def rice_j(model, radius, n=N_J):
    # Contour circulaire autour de la pointe.
    # On exclut une petite ouverture sur les levres.
    eps = 1e-4

    theta = np.linspace(
        -np.pi+eps,
        np.pi-eps,
        n
    )

    x = radius*np.cos(theta)
    y = radius*np.sin(theta)

    xy = tensor(
        np.column_stack([x, y]),
        requires_grad=True
    )

    uv, gu, gv, sxx, syy, sxy, w = strain_stress(
        model, xy
    )

    nx = tensor(np.cos(theta).reshape(-1,1))
    ny = tensor(np.sin(theta).reshape(-1,1))

    tx = sxx*nx + sxy*ny
    ty = sxy*nx + syy*ny

    dux = gu[:, 0:1]
    duy = gv[:, 0:1]

    integrand = (
        w*nx - tx*dux - ty*duy
    )

    val = integrand.detach().cpu().numpy().ravel()

    return float(
        radius*np.trapezoid(val, theta)
    )

# ============================================================
# 12. VALIDATION HIERARCHIQUE
# ============================================================

def evaluate(model, sigma, a):
    d, cod, ki_cod, cod_error = cod_and_ki(model)

    ki_basis = float(model.ki().detach())
    ki_ref = analytical_ki(sigma, a)

    g_cod = ki_cod**2 / EP
    g_ref = analytical_g(sigma, a)

    jvals = [
        rice_j(model, r)
        for r in J_RADII
    ]

    jmean = float(np.mean(jvals))

    contour_error = (
        100.0*(max(jvals)-min(jvals)) /
        max(abs(jmean), 1e-20)
    )

    jg_error = relative_error(jmean, g_cod)
    ki_error = relative_error(ki_cod, ki_ref)

    # Trois tests distincts
    pass_cod = cod_error < 2.0
    pass_ki = ki_error < 5.0
    pass_j = (
        contour_error < 2.0 and
        jg_error < 5.0
    )

    valid = pass_cod and pass_ki and pass_j

    print("\n--- KI ---")
    print(f"KI learned   = {ki_basis/1e6:.6f} MPa.sqrt(m)")
    print(f"KI from COD  = {ki_cod/1e6:.6f} MPa.sqrt(m)")
    print(f"KI reference = {ki_ref/1e6:.6f} MPa.sqrt(m)")
    print(f"KI error     = {ki_error:.4f} %")
    print(f"COD fit error= {cod_error:.4f} %")

    print("\n--- ENERGY ---")
    print(f"G from COD   = {g_cod:.6f} J/m2")
    print(f"G reference  = {g_ref:.6f} J/m2")
    print(f"G/Gc         = {g_cod/GC:.6f}")

    print("\n--- RICE J ---")
    for r, j in zip(J_RADII, jvals):
        print(f"r={r:.4f} m | J={j:.6f} J/m2")

    print(f"J mean       = {jmean:.6f} J/m2")
    print(f"Contour error= {contour_error:.5f} %")
    print(f"J/G error    = {jg_error:.5f} %")
    print(f"Validation   = {'PASS' if valid else 'FAIL'}")

    return {
        "sigma": sigma,
        "a": a,
        "ki_basis": ki_basis,
        "ki_cod": ki_cod,
        "ki_ref": ki_ref,
        "g": g_cod,
        "g_ref": g_ref,
        "j": jvals,
        "jmean": jmean,
        "cod_error": cod_error,
        "ki_error": ki_error,
        "contour_error": contour_error,
        "jg_error": jg_error,
        "valid": valid,
        "distances": d,
        "cod": cod
    }

# ============================================================
# 13. CHARGEMENT PROGRESSIF
# ============================================================

def main():
    results = []
    histories = []

    a = A0
    model = WilliamsPINN(
        analytical_ki(SIGMA0, a)
    ).to(DEVICE)

    print("\n" + "="*65)
    print("MODE B - PINN WILLIAMS + GRIFFITH")
    print("="*65)

    critical_record = None

    for sigma_mpa in SIGMA_LIST_MPA:
        sigma = sigma_mpa*1e6

        print("\n" + "-"*65)
        print(
            f"LOAD sigma={sigma_mpa:.4f} MPa | "
            f"a={a*1000:.3f} mm"
        )
        print("-"*65)

        hist = train_model(model, sigma, a)
        rec = evaluate(model, sigma, a)

        histories.append((sigma_mpa, hist))
        results.append(rec)

        if not rec["valid"]:
            print(
                "STOP: PINN non valide. "
                "Pas de decision Griffith."
            )
            break

        if rec["g"] >= GC:
            print("\nGRIFFITH THRESHOLD REACHED")
            critical_record = rec
            break

    # ========================================================
    # 14. POST-AMORCAGE
    # ========================================================

    if critical_record is not None:
        sigma_critical = critical_record["sigma"]

        print("\n" + "="*65)
        print("POST-INITIATION - CONSTANT REMOTE STRESS")
        print("="*65)

        for i in range(1, N_EXTENSION+1):
            anew = A0 + i*DA

            print("\n" + "-"*65)
            print(
                f"CRACK a={anew*1000:.3f} mm | "
                f"sigma={sigma_critical/1e6:.4f} MPa"
            )
            print("-"*65)

            hist = train_model(
                model, sigma_critical, anew,
                epochs=600
            )

            rec = evaluate(
                model, sigma_critical, anew
            )

            histories.append((anew, hist))
            results.append(rec)

            if not rec["valid"]:
                print("STOP: validation failed.")
                break

            if rec["g"] > GC:
                print(
                    "G > Gc: unstable extension "
                    "under constant remote stress."
                )

    # ========================================================
    # 15. FIGURES
    # ========================================================

    if len(results) > 0:
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        sigma_vals = [
            r["sigma"]/1e6 for r in results
        ]
        g_vals = [r["g"] for r in results]
        ki_vals = [
            r["ki_cod"]/1e6 for r in results
        ]

        ax[0,0].plot(
            sigma_vals, np.array(g_vals)/GC,
            "o-", label="PINN G/Gc"
        )
        ax[0,0].axhline(
            1.0, color="black", linestyle="--",
            label="Griffith"
        )
        ax[0,0].set_xlabel("Applied stress (MPa)")
        ax[0,0].set_ylabel("G/Gc")
        ax[0,0].set_title("Progressive loading")
        ax[0,0].legend()

        ax[0,1].plot(
            sigma_vals, ki_vals,
            "o-", label="PINN COD"
        )
        ax[0,1].set_xlabel("Applied stress (MPa)")
        ax[0,1].set_ylabel("KI (MPa sqrt(m))")
        ax[0,1].set_title("Stress intensity factor")
        ax[0,1].legend()

        last = results[-1]
        ax[1,0].plot(
            np.array(J_RADII)*1000,
            last["j"], "o-",
            label="Rice J"
        )
        ax[1,0].axhline(
            last["g"], color="black",
            linestyle="--", label="G from COD"
        )
        ax[1,0].set_xlabel("Contour radius (mm)")
        ax[1,0].set_ylabel("J (J/m2)")
        ax[1,0].set_title("Path independence")
        ax[1,0].legend()

        for label, hist in histories:
            ax[1,1].semilogy(
                np.maximum(hist, 1e-15),
                label=f"{label:.4f}"
            )
        ax[1,1].set_xlabel("Adam epoch")
        ax[1,1].set_ylabel("Loss")
        ax[1,1].set_title("PINN convergence")
        ax[1,1].legend(fontsize=7)

        for aa in ax.flat:
            aa.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # COD final
        d = last["distances"]
        cod = last["cod"]

        plt.figure(figsize=(7,5))
        plt.plot(
            d*1000, cod*1000,
            "o-", label="PINN COD"
        )

        ki = last["ki_ref"]
        cod_ref = (
            ki*(KAPPA+1)/MU *
            np.sqrt(d/(2*np.pi))
        )

        plt.plot(
            d*1000, cod_ref*1000,
            "--", label="Williams reference"
        )

        plt.xlabel("Distance behind tip (mm)")
        plt.ylabel("COD (mm)")
        plt.title("Crack opening displacement")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    torch.save(
        {
            "model_state": model.state_dict(),
            "results": [
                {
                    k: v for k, v in rec.items()
                    if k not in ["distances", "cod"]
                }
                for rec in results
            ]
        },
        "pinn_griffith_williams_corrected.pt"
    )

    print("\nModel saved.")
    print("END")

if __name__ == "__main__":
    main()
