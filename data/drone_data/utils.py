import numpy as np

def polyfit_with_fixed_points(n, x, y, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x ** np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf ** np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]

def point_regen(points_np):
    x_new = []
    y_new = []
    seg_length = 1
    t = np.linspace(0, 1, len(points_np))
    x = points_np[:, 0]
    y = points_np[:, 1]

    n, d, f = len(t), 7, 2
    params_x = polyfit_with_fixed_points(d, t[1:-1], x[1:-1], np.asarray([t[0], t[-1]]), np.asarray([x[0], x[-1]]))
    poly_x = np.polynomial.Polynomial(params_x)

    params_y = polyfit_with_fixed_points(d, t[1:-1], y[1:-1], np.asarray([t[0], t[-1]]), np.asarray([y[0], y[-1]]))
    poly_y = np.polynomial.Polynomial(params_y)

    t_cand = np.linspace(0, 1, 1001)
    x_cands = poly_x(t_cand)
    y_cands = poly_y(t_cand)
    x_cands[0] = points_np[0, 0]
    x_cands[-1] = points_np[-1, 0]
    y_cands[0] = points_np[0, 1]
    y_cands[-1] = points_np[-1, 1]

    cnt = 0
    for i in range(len(t_cand)):
        if i == 0:
            x_new.append(x_cands[i])
            y_new.append(y_cands[i])
            cnt = cnt +1
        else:
            dist_cnt = np.sqrt(np.sum((x_cands[i] - x_new[-1]) ** 2 + (y_cands[i] - y_new[-1]) ** 2))
            if dist_cnt >= seg_length:
                x_new.append(x_cands[i])
                y_new.append(y_cands[i])
                cnt = cnt + 1
    x_new.append(x_cands[-1])
    y_new.append(y_cands[-1])
    return np.transpose(np.concatenate((np.array([x_new]), np.array([y_new])), axis=0))