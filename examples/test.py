import numpy as np
import plotly.graph_objects as go

def plot_3d_histogram_bars(
    data: np.ndarray,
    bins=(20, 20),
    bar_scale=0.9,
    colorscale="Viridis",
    renderer=None,          # e.g. "browser", "vscode", "notebook_connected"
    save_html=None          # e.g. "3d_hist.html"
):
    """
    Plot a 3D bar-chart histogram for 2D points using Plotly Mesh3d.

    data: (n,2) array of (x,y) points
    bins: (bx, by) or [edges_x, edges_y]
    bar_scale: 0..1, shrink bars inside each bin so gaps are visible
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must have shape (n, 2)")

    # 2D histogram
    H, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=bins)

    # bin centers and widths (per-bin to be safe)
    x_cent = 0.5*(xedges[:-1] + xedges[1:])
    y_cent = 0.5*(yedges[:-1] + yedges[1:])
    x_w = np.diff(xedges)
    y_w = np.diff(yedges)

    # Collect vertices and faces for all cuboids
    X, Y, Z = [], [], []
    I, J, K = [], [], []
    intensity = []

    def add_bar(x0, x1, y0, y1, z):
        """Add one cuboid [x0,x1]x[y0,y1]x[0,z] as 8 verts + 12 triangles."""
        base = len(X)
        # order: 0..3 bottom, 4..7 top (see diagram in code comments)
        xs = [x0, x1, x1, x0, x0, x1, x1, x0]
        ys = [y0, y0, y1, y1, y0, y0, y1, y1]
        zs = [0, 0, 0, 0, z, z, z, z]
        X.extend(xs); Y.extend(ys); Z.extend(zs)
        intensity.extend([z]*8)

        # 12 triangles (two per face)
        faces = [
            (0,1,2),(0,2,3),       # bottom
            (4,6,5),(4,7,6),       # top
            (0,5,1),(0,4,5),       # side x+
            (1,6,2),(1,5,6),       # side y+
            (2,7,3),(2,6,7),       # side x-
            (3,4,0),(3,7,4)        # side y-
        ]
        for a,b,c in faces:
            I.append(base+a); J.append(base+b); K.append(base+c)

    # build bars
    for ix, xc in enumerate(x_cent):
        dx = x_w[ix]*bar_scale
        x0, x1 = xc - dx/2, xc + dx/2
        for iy, yc in enumerate(y_cent):
            count = int(H[ix, iy])  # H is (len(xedges)-1, len(yedges)-1)
            if count <= 0:
                continue
            dy = y_w[iy]*bar_scale
            y0, y1 = yc - dy/2, yc + dy/2
            add_bar(x0, x1, y0, y1, count)

    fig = go.Figure(go.Mesh3d(
        x=X, y=Y, z=Z,
        i=I, j=J, k=K,
        intensity=intensity, colorscale=colorscale, showscale=True,
        flatshading=True, opacity=1.0,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1),
        lightposition=dict(x=100, y=200, z=0)
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Count",
            aspectmode="data"
        ),
        title="3D Histogram (bars) from 2D data"
    )

    if renderer:
        import plotly.io as pio
        pio.renderers.default = renderer
    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn", auto_open=False)
    else:
        fig.show()

# Example
if __name__ == "__main__":
    np.random.seed(0)
    pts = np.random.randn(2000, 2)
    plot_3d_histogram_bars(pts, bins=(25, 25), bar_scale=0.9, renderer="browser", save_html='hist.html')